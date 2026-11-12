# Pruning vs XNOR-Net — Python Setup and Execution Guide

**Project:** Reproduction of *Pruning vs XNOR-Net: A Comprehensive Study of Deep Learning for Audio Classification on Edge-Devices*  
**Last updated:** 12 November 2026  
**Primary platform:** Windows / PowerShell, with a CUDA-enabled Python environment  
**Project directory:** `C:\AI_Projects\pruning-vs-xnor`

## 1. Purpose and current status

This README records how we located, prepared, and launched the Python code for the **audio-classification** portion of the pruning-vs-XNOR project. It separates the original authors' scripts from our locally used lightweight reproduction script, so that subsequent runs can be started consistently.

**What was confirmed in our session:**

- We inspected the original GitHub repository and its audio-related Python files.
- We located the original XNOR helper class `BinOp` in `common/binop_lib.py`.
- We identified the original data-preparation, training, and quantization entry points.
- We configured a GPU-capable environment that reported **PyTorch `2.9.1+cu128`**.
- We successfully started `esc50_lite_reproduction.py` with **all five ESC-50 folds**, a **5% pruning fraction**, and **no fine-tuning epochs**.
- The run reported **CUDA for FP32/XMini/fine-tuning paths** and **CPU for INT8**.

**Scope note:** The local script's console banner read `LIGHTWEIGHT REPRODUCTION | ESC-10`, even though the loader reported 400 samples in each of five ESC-50 folds. Therefore, *loading the ESC-50 dataset is not proof that the run evaluated all 50 classes*. This README does not claim full ESC-50 accuracy reproduction or give unverified numerical accuracy/compression results.

## 2. Project layout

The relevant files/folders discovered or used during the project are:

```text
C:\AI_Projects\pruning-vs-xnor\
├── audio\
│   ├── models\                 # model architecture definitions / model-related files
│   ├── nets\                   # network implementations
│   ├── resources\
│   │   └── prepare_dataset.py  # original audio data preparation
│   ├── trained_models\        # original repository's model checkpoint area
│   ├── trainer.py              # original full-precision / XNOR training entry point
│   └── quantization.py         # original quantization entry point
├── common\
│   └── binop_lib.py            # defines the original XNOR BinOp class
├── image\                    # separate image-classification portion (not covered here)
├── esc50_lite_reproduction.py  # local lightweight reproduction entry point
├── experiments\
│   └── esc50\                 # experiment output location used in our command
├── venv\                      # earlier Python virtual environment, if retained
└── venv_gpu\                  # GPU-enabled virtual environment used in the recorded run
```

`esc50_lite_reproduction.py` is part of our local reproduction workflow, **not a file verified to exist in the authors' original GitHub repository**. Some listed output/environment folders are created locally and should not be expected in a fresh clone.

## 3. First-time setup (PowerShell)

Run commands from a terminal in the project root. The following are **reusable setup instructions**; they are not a verbatim installation log of every package we installed.

### 3.1 Get the original repository

```powershell
cd C:\AI_Projects
git clone https://github.com/mohaimenz/pruning-vs-xnor.git
cd .\pruning-vs-xnor
```

If the project is already present, skip cloning:

```powershell
cd C:\AI_Projects\pruning-vs-xnor
```

### 3.2 Create or select a virtual environment

Our recorded successful command used the Python executable inside `venv_gpu` directly. To create an environment on a new machine:

```powershell
py -m venv venv_gpu
.\venv_gpu\Scripts\python.exe -m pip install --upgrade pip
```

To activate it (optional):

```powershell
.\venv_gpu\Scripts\Activate.ps1
```

> If PowerShell prevents activation, you can avoid changing the execution policy by using `.\venv_gpu\Scripts\python.exe` explicitly, as in our experiment command.

Install the required Python libraries for the scripts you are running. The original paper reports Python 3.7.4 / PyTorch 1.8.1, whereas our session used a newer PyTorch build. **Do not assume the legacy scripts are compatible with the new environment without changes.** No verified, complete pinned `requirements.txt` was established during our session, so this README does not invent a package lockfile.

Check the environment:

```powershell
.\venv_gpu\Scripts\python.exe -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

In the captured lightweight run, PyTorch identified itself as `2.9.1+cu128`.

## 4. Prepare and verify the audio dataset

Our working dataset root was:

```text
C:\AI_Projects\datasets\esc50
```

The lightweight reproduction was started with `--data-root` pointing to that directory. The console reported:

```text
Loaded ESC-50: fold1=400, fold2=400, fold3=400, fold4=400, fold5=400
```

ESC-50 contains 2,000 recordings in five predefined folds; the observed fold counts agree with that structure.

The original repository also provides `audio/resources/prepare_dataset.py`. From our code inspection, that legacy preparation script uses external command-line utilities such as `wget`, `unzip`, and `ffmpeg` and imports `wavio`. It may require adjustments on Windows. In our reproduction, we used the local ESC-50 dataset path above; **we have not established that the legacy preparation script itself completed successfully**.

To inspect the locally available files:

```powershell
Get-ChildItem "C:\AI_Projects\datasets\esc50" -Force
Get-ChildItem "C:\AI_Projects\datasets\esc50" -Recurse -File | Measure-Object
```

**Important:** The original ACDNet-style training workflow can depend on preprocessed files (for example, `.npz` files). Raw ESC-50 `.wav` files and a prepared `.npz` file are **not interchangeable**. Check each script's loader and expected data layout before passing a dataset path. We did not establish a universally correct `wav20.npz` file for every training command.

## 5. Python entry points: what each file is for

| Script | Function | Status in our project |
| --- | --- | --- |
| `audio/resources/prepare_dataset.py` | Original audio dataset downloading/preparation | Inspected; successful end-to-end execution not confirmed |
| `audio/trainer.py` | Original model training; `--xnor 0` for ordinary training, `--xnor 1` for XNOR | Identified; not the confirmed five-fold lightweight command |
| `audio/quantization.py` | Original quantization process | Identified; requires editing settings before launching |
| `common/binop_lib.py` | Contains `class BinOp` used in the original binary/XNOR implementation | Confirmed by file search |
| `esc50_lite_reproduction.py` | Local lightweight evaluation/reproduction workflow with pruning and optional fine-tuning settings | **Confirmed launched** |

The commands in the following sections must be run from the **project root**, unless stated otherwise.

## 6. Exact command used for our lightweight reproduction

This is the command we recorded on 11 November 2026, reformatted using PowerShell backticks for readability:

```powershell
cd C:\AI_Projects\pruning-vs-xnor

.\venv_gpu\Scripts\python.exe .\esc50_lite_reproduction.py `
  --data-root "C:\AI_Projects\datasets\esc50" `
  --folds 1 2 3 4 5 `
  --prune-fraction 0.05 `
  --finetune-epochs 0 `
  --finetune-steps 1 `
  --calibration-samples 12 `
  --val-per-class 4 `
  --eval-crops 3 `
  --output experiments/esc50/lite_prune5_no_finetune
```

**Recorded console output (excerpt):**

```text
Loaded ESC-50: fold1=400, fold2=400, fold3=400, fold4=400, fold5=400

LIGHTWEIGHT REPRODUCTION | ESC-10 | PyTorch 2.9.1+cu128
Device FP32/XMini/fine-tune: cuda; INT8: CPU
Validation: 4 clips ...
```

The captured transcript does not include the full final accuracy tables, final compression files, or the process exit code. This establishes that the code **started and loaded the data**, not that every metric was successfully finalized.

### Explanation of the arguments

| Argument | Recorded setting | Meaning / caution |
| --- | --- | --- |
| `--data-root` | `C:\AI_Projects\datasets\esc50` | Dataset folder used by the local script |
| `--folds` | `1 2 3 4 5` | Run across the five designated folds |
| `--prune-fraction` | `0.05` | Requested pruning fraction of 5% |
| `--finetune-epochs` | `0` | No fine-tuning epochs requested |
| `--finetune-steps` | `1` | Configured steps parameter; does not override zero fine-tuning epochs |
| `--calibration-samples` | `12` | Number requested for calibration |
| `--val-per-class` | `4` | Validation clips per class requested |
| `--eval-crops` | `3` | Evaluation crops requested |
| `--output` | `experiments/esc50/lite_prune5_no_finetune` | Destination for this run's output, if written by the script |

Because the local script was not attached here, descriptions reflect the argument names and observed invocation. Consult its parser/implementation for exact internal semantics, random seeds, and saved file names.

### After the run: inspect outputs

```powershell
Get-ChildItem .\experiments\esc50\lite_prune5_no_finetune -Recurse -File
```

Use the **actual output files** for any final five-fold accuracy table or compression comparison. Do not copy the paper's numerical results into the local experiment table as if we reproduced them.

## 7. Original repository's Python commands (reference only)

These commands follow the original GitHub README. They are **examples from the authors' workflow, not evidence that we ran them successfully in our present environment**. They may require prepared datasets, compatible package versions, and working model files.

### Full-precision Mini-ACDNet on ESC-50

```powershell
.\venv_gpu\Scripts\python.exe .\audio\trainer.py `
  --model_path "audio/models/mini_acdnet.pt" `
  --dataset esc50 `
  --data "PATH_TO_PREPARED_DATASET" `
  --xnor 0 `
  --nClasses 50 `
  --model_name mini_acdnet_esc50
```

### XNOR ACDNet on ESC-50

```powershell
.\venv_gpu\Scripts\python.exe .\audio\trainer.py `
  --model_path "" `
  --dataset esc50 `
  --data "PATH_TO_PREPARED_DATASET" `
  --xnor 1 `
  --nClasses 50 `
  --model_name xacdnet_esc50
```

The author-provided training flags include `--model_path`, `--dataset`, `--data`, `--xnor`, `--nClasses`, and `--model_name`. Replace `PATH_TO_PREPARED_DATASET` with the **format and path expected by `audio/trainer.py`**, not automatically the raw WAV directory.

### Quantization script

The original README specifies editing the following options in `audio/quantization.py` before running it:

```text
opt.dataset
opt.data
opt.model_path
opt.model_name
opt.split
```

Then run:

```powershell
.\venv_gpu\Scripts\python.exe .\audio\quantization.py
```

This is **not** the same command as our local lightweight INT8 evaluation. The README does not claim the original quantization script completed.

## 8. Code discovery and debugging checks we used

### Inspect the original audio folder

```powershell
Get-ChildItem .\audio -Force
```

This confirmed folders such as `models`, `nets`, `resources`, and `trained_models`, plus scripts including `trainer.py` and `quantization.py`.

### Find the project-defined `BinOp` class

```powershell
Get-ChildItem . -Recurse -File -Filter "*.py" |
  Select-String -Pattern "class BinOp"
```

The project-specific result was:

```text
common\binop_lib.py:4:class BinOp():
```

Other matches under `venv\Lib\site-packages\` were from installed libraries and should not be confused with the project's XNOR helper.

### Inspect the dataset preparation source

```powershell
Get-Content .\audio\resources\prepare_dataset.py
```

This helped us identify its external-tool prerequisites and directory assumptions before attempting to run legacy data preparation.

## 9. Compatibility notes and unresolved limitations

1. **Old vs new PyTorch:** The published experiments report Python 3.7.4 and PyTorch 1.8.1; our logged lightweight run used PyTorch 2.9.1+cu128. Results and API behavior may differ.
2. **Windows portability:** The original dataset preparation relies on commands such as `wget`, `unzip`, and `ffmpeg`, which might not be installed or configured in PowerShell. A pre-downloaded dataset can avoid dependence on that particular legacy script, provided the new script supports its format.
3. **Dataset format mismatch:** The raw ESC-50 dataset root and model-training `.npz` inputs must not be assumed equivalent. Validate sample rate, channels, label mapping, fold metadata, and preprocessing before comparing results.
4. **ESC-10 vs ESC-50:** The successful launch displayed `ESC-10`. Verify the evaluated class subset and class count in `esc50_lite_reproduction.py` before claiming 50-class ESC-50 reproduction.
5. **No fine-tuning:** `--finetune-epochs 0` intentionally removes fine-tuning from this recorded run; this can materially affect post-pruning performance.
6. **Not a full paper reproduction:** The recorded command uses lightweight settings (`12` calibration samples, `4` validation clips per class, `3` evaluation crops) and may differ from the original training, pruning, and quantization protocols.
7. **No invented performance numbers:** The visible command excerpt confirms dataset loading and execution configuration, but does not provide final fold-by-fold accuracies or model-size/compression measurements.

These are **documented compatibility concerns and verification gaps**; they should not all be described as confirmed software bugs in the authors' GitHub repository.

## 10. Restart checklist

1. Open **PowerShell** and `cd C:\AI_Projects\pruning-vs-xnor`.
2. Check that `.\venv_gpu\Scripts\python.exe` exists and can import PyTorch.
3. Confirm the local ESC-50 dataset path is present.
4. Ensure `esc50_lite_reproduction.py` is in the project root.
5. Copy the exact command in **Section 6** to resume the recorded lightweight configuration.
6. Examine the outputs in `experiments/esc50/lite_prune5_no_finetune` and record per-fold metrics only after verifying the actual evaluated classes and completed run.

## 11. References

- Official code repository: <https://github.com/mohaimenz/pruning-vs-xnor>
- Original repository instructions: <https://github.com/mohaimenz/pruning-vs-xnor/blob/main/README.md>
- Original ACDNet project (dataset-processing reference): <https://github.com/mohaimenz/acdnet>
- ESC-50 dataset: <https://github.com/karoldvl/ESC-50>
- Paper overview: <https://research.monash.edu/en/publications/pruning-vs-xnor-net-a-comprehensive-study-of-deep-learning-for-au/>

---

**Summary:** We examined the legacy Python modules, identified the model and XNOR helper paths, configured/used a CUDA-capable Python environment, pointed the local reproduction script at ESC-50 data, and launched a lightweight five-fold pruning experiment. The current README is an **execution and initialization record**; final scientific claims require the saved metrics, class-split confirmation, and comparison to the original paper's full protocol.
