"""Low-load, *abridged* Pruning-vs-XNOR comparison on existing ESC-10 checkpoints.

Run from the root of mohaimenz/pruning-vs-xnor (Python 3.10, Torch 2.9.1).
Requires esc50_repro_runner_v2.py next to this file and the previously trained
200-epoch Mini and XMini checkpoints. No 200/500/2000-epoch retraining.

For each selected held-out fold, this script:
  1. Evaluates the already-trained Mini-ACDNet and XMini-ACDNet fairly.
  2. One-shot L2-prunes a fraction of TFEB convolution channels in Mini.
  3. Optionally fine-tunes the resulting smaller Mini for a few tiny batches.
  4. Performs REAL PyTorch eager INT8 post-training quantization on CPU.
  5. Evaluates all variants on exactly the same balanced validation clips.
  6. Saves a CSV summary and reusable compressed checkpoint/artifact.

IMPORTANT SCIENTIFIC LIMITATIONS:
  - L2 one-shot channel pruning is NOT the paper's iterative Taylor/hybrid pruning.
  - Pruned Mini is NOT the paper's Micro/QMicro-ACDNet.
  - Binary-model PyTorch weights are not packed as MCU-ready 1-bit tensors.
  - Reduced validation clips/crops, short fine-tune, and 200-epoch source models
    do NOT reproduce the published paper's accuracies or compute requirements.
  - Best checkpoints were selected using the held-out folds: selection bias.
"""
import argparse
import copy
import csv
import json
import os
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from audio.nets import models
from audio.resources import esc_subsets
from audio.resources import utils as U
from common.binop_lib import BinOp
from esc50_repro_runner_v2 import (
    INPUT_LENGTH, SAMPLE_RATE, BCLearningBatches, load_folds,
    select_fold, set_seed,
)


def load_trusted_local(path):
    # Only use locally created experiment checkpoints: torch.load may unpickle.
    if not os.path.isfile(path):
        raise FileNotFoundError('Missing checkpoint: ' + path)
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def locate_checkpoint(model_name, fold):
    if fold == 1:
        folder = 'mini10_200epochs' if model_name == 'mini' else 'xmini10_200epochs'
    else:
        folder = 'mini10_cv200_folds2to5' if model_name == 'mini' else 'xmini10_cv200_folds2to5'
    return os.path.join('experiments', 'esc50', folder, 'checkpoints',
                        '{}_c10_fold{}.pt'.format(model_name, fold))


def load_checkpoint_model(model_name, fold, device):
    path = locate_checkpoint(model_name, fold)
    payload = load_trusted_local(path)
    if 'weight' not in payload or 'config' not in payload:
        raise ValueError('Unexpected checkpoint format: ' + path)
    meta = payload.get('meta', {})
    if meta and (int(meta.get('classes', 10)) != 10 or int(meta.get('fold', fold)) != fold):
        raise ValueError('Checkpoint metadata mismatch in {}: {}'.format(path, meta))
    config = list(payload['config'])
    factory = models.GetBINACDNetModel if model_name == 'xmini' else models.GetACDNetModel
    net = factory(INPUT_LENGTH, 10, SAMPLE_RATE, config).to(device)
    net.load_state_dict(payload['weight'], strict=True)
    net.eval()
    binop = BinOp(net) if model_name == 'xmini' else None
    return net, binop, path


def balanced_samples(records, per_class, seed):
    by_label = defaultdict(list)
    for audio, label in records:
        by_label[label].append((audio, int(label)))
    rng = random.Random(seed)
    selected = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        if per_class and len(items) < per_class:
            raise ValueError('Not enough validation clips in class {}'.format(label))
        selected.extend(items[:per_class] if per_class else items)
    rng.shuffle(selected)
    return selected


def crops(audio, n_crops):
    padded = U.padding(INPUT_LENGTH // 2)(audio)
    normalized = U.normalize(32768.0)(padded)
    if n_crops == 1:
        start = (len(normalized) - INPUT_LENGTH) // 2
        arr = normalized[start:start + INPUT_LENGTH][np.newaxis, :]
    else:
        arr = U.multi_crop(INPUT_LENGTH, n_crops)(normalized)
    return np.asarray(arr, dtype=np.float32)


def precompute_eval_batches(records, n_crops, recs_per_batch):
    result = []
    for i in range(0, len(records), recs_per_batch):
        group = records[i:i + recs_per_batch]
        batch = np.stack([crops(sound, n_crops) for sound, _ in group], axis=0)
        x = torch.from_numpy(batch.reshape(-1, 1, 1, INPUT_LENGTH).copy())
        y = torch.tensor([label for _, label in group], dtype=torch.long)
        result.append((x, y))
    return result


def score_model(model, batches, device, n_crops, binop=None):
    model.eval()
    correct = 0
    n = 0
    loss_total = 0.0
    if binop is not None:
        binop.binarization()
    try:
        with torch.inference_mode():
            for xb, yb in batches:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb).reshape(len(yb), n_crops, 10).mean(1)
                if out.shape != (len(yb), 10) or not torch.isfinite(out).all():
                    raise AssertionError('Invalid prediction tensor')
                correct += int((out.argmax(1) == yb).sum().item())
                loss_total += float(-out.gather(1, yb[:, None]).clamp(min=1e-8).log().sum().item())
                n += len(yb)
    finally:
        if binop is not None:
            binop.restore()
    return round(correct * 100.0 / n, 2), round(loss_total / n, 5)


def clone_conv(conv, new_in, new_out):
    if conv.groups != 1:
        raise ValueError('Grouped convolutions not supported')
    return nn.Conv2d(new_in, new_out, conv.kernel_size, stride=conv.stride,
                     padding=conv.padding, dilation=conv.dilation,
                     groups=conv.groups, bias=(conv.bias is not None),
                     padding_mode=conv.padding_mode)


def prune_tfeb_l2(model, fraction):
    """One-shot L2 ranking of each TFEB Conv; physical output/input rewiring."""
    if not 0 < fraction < 0.8:
        raise ValueError('--prune-fraction should be >0 and <0.8')
    net = copy.deepcopy(model).cpu().eval()
    conv_positions = [i for i, m in enumerate(net.tfeb) if isinstance(m, nn.Conv2d)]
    if len(conv_positions) != 10:
        raise RuntimeError('Expected 10 TFEB Conv2d layers, found {}'.format(len(conv_positions)))
    li = next((i for i, m in enumerate(net.tfeb) if isinstance(m, nn.Linear)), None)
    if li is None:
        raise RuntimeError('Missing TFEB final Linear layer')
    changes = []
    for j, ci in enumerate(conv_positions):
        layer = net.tfeb[ci]
        if ci + 1 >= len(net.tfeb) or not isinstance(net.tfeb[ci + 1], nn.BatchNorm2d):
            raise RuntimeError('Expected Conv-BN at index {}'.format(ci))
        bn = net.tfeb[ci + 1]
        count = layer.out_channels
        remove_n = int(round(count * fraction))
        if count >= 4:
            remove_n = max(1, remove_n)
        remove_n = min(remove_n, count - 2)
        if remove_n <= 0:
            continue
        keep_n = count - remove_n
        scores = layer.weight.detach().square().flatten(1).sum(1)
        keep = torch.topk(scores, keep_n, largest=True).indices.sort().values
        next_idx = conv_positions[j + 1] if j + 1 < len(conv_positions) else li
        next_module = net.tfeb[next_idx]
        if isinstance(next_module, nn.Conv2d):
            if next_module.in_channels != count:
                raise RuntimeError('Next conv shape mismatch: {} vs {}'.format(next_module.in_channels, count))
            new_next = clone_conv(next_module, keep_n, next_module.out_channels)
        elif isinstance(next_module, nn.Linear):
            if next_module.in_features != count:
                raise RuntimeError('Next linear input mismatch: {} vs {}'.format(next_module.in_features, count))
            new_next = nn.Linear(keep_n, next_module.out_features,
                                 bias=(next_module.bias is not None))
        else:
            raise RuntimeError('Unsupported downstream layer')
        new_layer = clone_conv(layer, layer.in_channels, keep_n)
        new_bn = nn.BatchNorm2d(keep_n, eps=bn.eps, momentum=bn.momentum,
                                affine=bn.affine, track_running_stats=bn.track_running_stats)
        with torch.no_grad():
            new_layer.weight.copy_(layer.weight[keep])
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias[keep])
            if bn.affine:
                new_bn.weight.copy_(bn.weight[keep])
                new_bn.bias.copy_(bn.bias[keep])
            if bn.track_running_stats:
                new_bn.running_mean.copy_(bn.running_mean[keep])
                new_bn.running_var.copy_(bn.running_var[keep])
                new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
            if isinstance(next_module, nn.Conv2d):
                new_next.weight.copy_(next_module.weight[:, keep])
                if next_module.bias is not None:
                    new_next.bias.copy_(next_module.bias)
            else:
                new_next.weight.copy_(next_module.weight[:, keep])
                if next_module.bias is not None:
                    new_next.bias.copy_(next_module.bias)
        net.tfeb[ci] = new_layer
        net.tfeb[ci + 1] = new_bn
        net.tfeb[next_idx] = new_next
        # Config indexes 2-11 correspond to TFEB Conv3-Conv12.
        net.ch_config = list(net.ch_config)
        net.ch_config[2 + j] = keep_n
        changes.append({'conv': j + 3, 'before': count, 'after': keep_n})
    if not changes:
        raise RuntimeError('No channels removed')
    return net, changes


def finetune_pruned(model, training, device, epochs, steps, batch_size):
    if epochs <= 0:
        return model
    model.to(device).train()
    batches = BCLearningBatches(training, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    kl = nn.KLDivLoss(reduction='batchmean')
    for epoch in range(epochs):
        losses = []
        for _ in range(steps):
            xx, yy = batches.batch(batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            prob = model(xx)
            loss = kl(prob.clamp(min=1e-8).log(), yy)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print('  Fine-tune {}/{}: train loss {:.4f}'.format(epoch + 1, epochs, sum(losses) / len(losses)), flush=True)
    return model.cpu().eval()


def fuse_conv_bn_relu(seq):
    fused = 0
    for i in range(len(seq) - 2):
        if (isinstance(seq[i], nn.Conv2d) and isinstance(seq[i + 1], nn.BatchNorm2d)
                and isinstance(seq[i + 2], nn.ReLU)):
            torch.ao.quantization.fuse_modules(seq, [str(i), str(i + 1), str(i + 2)], inplace=True)
            fused += 1
    return fused


def quantize_int8(pruned, calibration, calib_samples):
    candidate = models.GetACDNetModel(INPUT_LENGTH, 10, SAMPLE_RATE,
                                       list(pruned.ch_config), quantize=True).cpu().eval()
    candidate.load_state_dict(pruned.state_dict(), strict=True)
    fused = fuse_conv_bn_relu(candidate.sfeb) + fuse_conv_bn_relu(candidate.tfeb)
    if fused == 0:
        raise RuntimeError('Could not fuse Conv-BN-ReLU layers')
    supported = list(torch.backends.quantized.supported_engines)
    engine = next((e for e in ('x86', 'fbgemm', 'onednn', 'qnnpack') if e in supported), None)
    if engine is None:
        raise RuntimeError('No supported CPU INT8 backend: {}'.format(supported))
    torch.backends.quantized.engine = engine
    candidate.qconfig = torch.ao.quantization.get_default_qconfig(engine)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        torch.ao.quantization.prepare(candidate, inplace=True)
    samples = balanced_samples(calibration, 0, seed=42)
    with torch.inference_mode():
        for audio, _ in samples[:calib_samples]:
            one = torch.from_numpy(crops(audio, 1).reshape(1, 1, 1, INPUT_LENGTH).copy())
            candidate(one)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        torch.ao.quantization.convert(candidate, inplace=True)
    int8_count = sum(isinstance(m, torch.ao.nn.quantized.Conv2d) for m in candidate.modules())
    if int8_count < 1:
        raise AssertionError('No quantized convolutions were created')
    return candidate.eval(), engine, int8_count, fused


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser(description='Lightweight comparative reproduction of Pruning vs XNOR on ESC-10')
    ap.add_argument('--data-root', default=r'C:\AI_Projects\datasets\esc50')
    ap.add_argument('--folds', type=int, nargs='+', choices=[1, 2, 3, 4, 5], default=[1])
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--prune-fraction', type=float, default=0.20)
    ap.add_argument('--finetune-epochs', type=int, default=2)
    ap.add_argument('--finetune-steps', type=int, default=4)
    ap.add_argument('--finetune-batch-size', type=int, default=16)
    ap.add_argument('--calibration-samples', type=int, default=20)
    ap.add_argument('--val-per-class', type=int, default=4, help='4=40 clips per fold; 0=all 80 clips')
    ap.add_argument('--eval-crops', type=int, default=3, help='1=center crop; 10=original validation setting')
    ap.add_argument('--eval-batch', type=int, default=2)
    ap.add_argument('--output', default='experiments/esc50/lite_reproduction')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    if a.finetune_epochs < 0 or a.finetune_steps < 1 or a.finetune_batch_size < 1:
        raise ValueError('Invalid fine-tuning arguments')
    if a.eval_crops < 1 or a.calibration_samples < 1 or a.eval_batch < 1 or a.val_per_class < 0:
        raise ValueError('Invalid validation/calibration arguments')
    set_seed(a.seed)
    # Reduce CPU thread contention during repeated INT8 inference on laptops.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    gpu = 'cuda' if torch.cuda.is_available() and a.device != 'cpu' else 'cpu'
    if a.device == 'cuda':
        gpu = 'cuda'
    device = torch.device(gpu)
    os.makedirs(a.output, exist_ok=True)
    dataset = os.path.join(a.data_root, 'wav20.npz')
    folds_data = load_folds(dataset)
    classes = list(esc_subsets.esc10)
    results = []
    print('\nLIGHTWEIGHT REPRODUCTION | ESC-10 | PyTorch {}'.format(torch.__version__), flush=True)
    print('Device FP32/XMini/fine-tune: {}; INT8: CPU'.format(device), flush=True)
    print('Validation: {} clips/class, {} crop(s); pruning: {:.0f}% of TFEB channels'.format(
        a.val_per_class or 8, a.eval_crops, 100 * a.prune_fraction), flush=True)
    for fold in a.folds:
        set_seed(a.seed + fold)
        training, all_validation = select_fold(folds_data, fold, classes)
        validation = balanced_samples(all_validation, a.val_per_class, a.seed + fold)
        prepared = precompute_eval_batches(validation, a.eval_crops, a.eval_batch)
        print('\n==== FOLD {} | {} validation clips ===='.format(fold, len(validation)), flush=True)
        mini, _, mini_path = load_checkpoint_model('mini', fold, device)
        xmini, binop, xmini_path = load_checkpoint_model('xmini', fold, device)
        print('  Loaded:', mini_path, '|', xmini_path, flush=True)
        mini_acc, mini_loss = score_model(mini, prepared, device, a.eval_crops)
        xmini_acc, xmini_loss = score_model(xmini, prepared, device, a.eval_crops, binop=binop)
        mini_n = count_params(mini)
        xmini_n = count_params(xmini)
        pruned, changes = prune_tfeb_l2(mini.cpu(), a.prune_fraction)
        del mini, xmini, binop
        pruned_n = count_params(pruned)
        print('  TFEB pruned {} layers | {:,} -> {:,} FP32 params ({:.1f}% fewer)'.format(
            len(changes), mini_n, pruned_n, 100 * (1 - pruned_n / mini_n)), flush=True)
        # Make sure the physically smaller network still supports backprop.
        pruned.train()
        probe_x = prepared[0][0][:2].clone()
        pruned(probe_x).sum().backward()
        pruned.zero_grad(set_to_none=True)
        pruned.eval()
        pruned_acc, pruned_loss = score_model(pruned, prepared, torch.device('cpu'), a.eval_crops)
        pruned = finetune_pruned(pruned, training, device, a.finetune_epochs,
                                 a.finetune_steps, a.finetune_batch_size)
        tuned_acc, tuned_loss = score_model(pruned, prepared, torch.device('cpu'), a.eval_crops)
        qmodel, engine, int8_convs, fused = quantize_int8(pruned, training, a.calibration_samples)
        quant_acc, quant_loss = score_model(qmodel, prepared, torch.device('cpu'), a.eval_crops)
        folder = os.path.join(a.output, 'fold{}'.format(fold))
        os.makedirs(folder, exist_ok=True)
        pruned_ckpt = os.path.join(folder, 'pruned_mini_fp32.pt')
        torch.save({'weight': pruned.state_dict(), 'config': list(pruned.ch_config),
                    'description': 'L2 one-shot TFEB channel-pruned, optionally short-finetuned Mini-ACDNet'}, pruned_ckpt)
        quant_path = os.path.join(folder, 'pruned_mini_int8_traced.pt')
        export_x = prepared[0][0][:1].cpu()
        with torch.inference_mode():
            traced = torch.jit.trace(qmodel, export_x, check_trace=False)
            torch.jit.save(traced, quant_path)
            restored = torch.jit.load(quant_path, map_location='cpu').eval()
            orig = qmodel(export_x)
            check = restored(export_x)
            if not torch.allclose(orig, check, atol=1e-5, rtol=1e-5):
                raise AssertionError('INT8 export/reload mismatch')
        items = [
            ('Mini FP32 (existing)', mini_acc, mini_loss, mini_n, mini_path),
            ('XMini PyTorch (existing)', xmini_acc, xmini_loss, xmini_n, xmini_path),
            ('Mini L2-channel-pruned', pruned_acc, pruned_loss, pruned_n, ''),
            ('Mini pruned + short fine-tune', tuned_acc, tuned_loss, pruned_n, pruned_ckpt),
            ('Mini pruned + fine-tune + INT8', quant_acc, quant_loss, pruned_n, quant_path),
        ]
        for variant, acc, loss, npars, path in items:
            results.append({'fold': fold, 'variant': variant, 'validation_clips': len(validation),
                            'eval_crops': a.eval_crops, 'accuracy_percent': acc,
                            'mean_nll': loss, 'fp32_model_parameters': npars,
                            'artifact_mb': round(os.path.getsize(path) / 1048576, 3) if path else '',
                            'artifact': path})
            print('  {:36s} accuracy {:6.2f}%'.format(variant, acc), flush=True)
        with open(os.path.join(folder, 'compression_info.json'), 'w', encoding='utf-8') as f:
            json.dump({'fold': fold, 'pruning': changes, 'quantization_backend': engine,
                       'int8_convs': int8_convs, 'conv_bn_relu_fused': fused,
                       'calibration_samples': a.calibration_samples,
                       'val_per_class': a.val_per_class, 'eval_crops': a.eval_crops,
                       'finetune_epochs': a.finetune_epochs, 'finetune_steps': a.finetune_steps},
                      f, indent=2)
    out_path = os.path.join(a.output, 'comparison.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print('\nRESULTS SAVED: {}'.format(out_path), flush=True)
    if len(a.folds) > 1:
        grouped = defaultdict(list)
        for row in results:
            grouped[row['variant']].append(float(row['accuracy_percent']))
        print('\nMEAN ACROSS {} FOLDS:'.format(len(a.folds)))
        for name, nums in grouped.items():
            print('  {:36s} {:6.2f}%'.format(name, sum(nums)/len(nums)))
    print('\nNOTE: This is a LOW-LOAD illustration, not a reproduction of paper Table 3/6.')
    print('No full Taylor/hybrid pruning, Micro-ACDNet, matched-memory comparison,')
    print('bit-packed XNOR CPU kernels, or 2000-epoch full-class study was performed.')


if __name__ == '__main__':
    main()
