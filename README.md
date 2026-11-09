**[AttCo: Attention-based Co-Learning Fusion of Deep Feature Representation for Medical Image Segmentation using Multimodality](https://www.sciencedirect.com/science/article/pii/S0893608026000353)**



## 📄 License
This project is released under the [Apache 2.0 license](LICENSE).

## 🎓 Citation

```
@misc{AttCo,
      title={AttCo: Attention-based Co-Learning Fusion of Deep Feature Representation for Medical Image Segmentation using Multimodality}, 
      author={Duy-Phuong Dao, Hyung-Jeong Yang*, Sae-Ryung Kang, Soo-Hyung Kim},
      year={2026},
      journal={Neural Networks},
}
```


## ACDC Wave 2.5D

### Params and GFLOPs

Training prints total/trainable Params and inference GFLOPs and saves
`complexity_fold_N.json` beside the checkpoints. Install `requirements.txt`
to include the fvcore profiler. `--is_thop false` disables FLOPs tracing while
retaining parameter counts; this legacy flag now correctly parses true/false.
Missing profiler dependencies or failed traces are recorded without stopping training.

Measure a model without loading any dataset or starting training:

```sh
python model_complexity.py --modelname Wave --image_size 224 224 --output complexity_Wave.json
python model_complexity.py --modelname WaveDe --device cuda:0
python model_complexity.py --modelname EfficientMedNeXt --volume_size 32 128 128
python model_complexity.py --modelname EffiDec3D --device cpu --volume_size 32 128 128 --output complexity_results/EffiDec3D.json
```

Input batch size is always 1: `[1,3,H,W]` for Wave/WaveDe (three adjacent
slices), `[1,1,H,W]` for 2D baselines, and `[1,1,D,H,W]` for 3D models.
The profiler measures an eval forward, excluding backward/optimizer steps.
Params includes every registered parameter, including frozen parameters;
trainable Params is also reported. fvcore counts one multiply-add as one FLOP;
compare models using the same input dimensions and counting convention.
Unsupported operations (e.g. FFT or custom CUDA operators) are listed in the JSON.
When present, `counted_gflops` is only a partial estimate and `gflops` is null.
Even a successful trace uses fvcore's supported-op coverage, not a count of
every scalar operation. The model's training/eval flags are restored afterwards.
For MedNeXt-derived 3D models, block checkpointing is temporarily disabled
during tracing to avoid checkpoint autograd/JIT conflicts, then restored even
if tracing fails. This affects only the model instance being measured.

Expected layout: `training/patient001/Info.cfg`, `patient001_frame01.nii.gz`
and `patient001_frame01_gt.nii.gz` (frame numbers come from ED/ES in Info.cfg).
Each sample contains three adjacent spatial slices from one phase, ordered
previous/center/next; only the center mask is supervised. Boundary slices repeat.
Labels remain 0=background, 1=RV, 2=MYO, 3=LV. Normalization uses each phase
volume's nonzero voxels. Images are resized in-plane to 224x224 by default;
this is pixel resizing, not physical-spacing resampling.

Install a compatible torch/torchvision pair for your CUDA environment, then
install the remaining dependencies from requirements.txt.

```sh
python main.py --path_image /path/to/ACDC/training --folds 0 --train_mode single --gpu_ids 0
python main.py --path_image /path/to/ACDC/training --folds 0 --train_mode multi --gpu_ids 0,1
python test.py --path_image /path/to/ACDC/training --fold 0 --pretrained checkpoint/ACDC/Wave/best_fold_0.pt --save_pred
```

`main_single.py` uses the same training entry point. Use `--train_mode cpu`
for CPU training and `--cpu` for CPU evaluation. DataLoader workers default to
0 for Windows compatibility; increase `--num_workers` if appropriate.
`--image_size H W` (multiples of 32, >=64) and `--base_channels` (default 16)
must match between training and evaluation. Checkpoints contain model state_dict
weights, not optimizer/resume state.

Training Dice is the mean per-slice diagnostic. Validation Dice is accumulated
per patient-phase on the resized grid; checkpoint selection uses validation
CE+Dice loss. test.py restores logits to the original H/W before argmax and
reports per patient-phase volume Dice against the original labels. Both-empty
structures score 1. Output NIfTI files preserve original spacing, origin and
direction, with separate files per ED/ES frame. CSV splits are patient-level;
ED and ES of one patient always stay together. These evaluation patients are
held-out folds of the labeled training data, not the unlabeled challenge test set.


### Model selection

`MODEL_REGISTRY` in main.py is shared by training and test.py. Use
`--modelname Wave` (default, Wave_ACDC.py) or `--modelname WaveDe`
(WaveDe_ACDC.py). Both accept the existing ACDC 2.5D loader and center-slice loss.
WaveDe requires base_channels >=8 and divisible by 4 (default 16). WaveDe keeps
its constructor initialization to preserve deformable-attention offset initialization.

```sh
python main.py --path_image /path/to/ACDC/training --modelname WaveDe --train_mode multi --gpu_ids 0,1
python test.py --path_image /path/to/ACDC/training --modelname WaveDe --fold 0 --pretrained checkpoint/ACDC/WaveDe/best_fold_0.pt --save_pred --output_path test_predictions/WaveDe/fold_0
```

The selected architecture and base_channels must match the checkpoint. Existing
Wave checkpoints remain supported. All folds run by default during training.
Weights/logs are separated under checkpoint/ACDC/<modelname>/.
Legacy models WaveCo, WaveCo_Constraint, AttCo_BraTS and AttCo_AutoPET are also
registered with their actual class names, dataset and 3D layout. The current
ACDC entry points reject those models before loading data: their original 3D
pipelines must be adapted separately before they can train on ACDC slices.
To add another compatible model, add its module/class/dataset/layout tuple to
MODEL_REGISTRY and keep its constructor compatible with inChannel/outChannel/baseChannel.


### Quantitative results for statistical analysis

Every test.py run writes two CSV files under --output_path, regardless of
--save_pred:

- test_results_fold_N.csv: one row per patient-phase, retaining original-grid
  3D Dice_RV, Dice_MYO, Dice_LV and Dice_mean.
- patient_results_fold_N.csv: one row per patient, with separate *_ED and *_ES
  scores. Unsuffixed Dice_RV/MYO/LV are the arithmetic ED/ES means; Dice_mean
  averages all three structures and both phases. Scores range from 0 to 1,
  not percentages. Both-empty structures follow the existing Dice=1 convention.

Both tables include model, fold, checkpoint and patient_id. The patient table
requires exactly one ED and one ES result per patient. Missing or duplicate
phases are errors. Keep the original scores for downstream analysis; p-values
and confidence intervals are not calculated automatically. For paired model
comparisons, align patient rows using patient_id and fold (with identical fold
splits); ED/ES and slices from one patient are correlated observations. Concatenate
the five held-out patient tables to obtain the full out-of-fold patient dataset.
Do not include repeated runs of the same patient as independent patients.

对比模型（原生 2D/3D 输入）的论文出处、具体版本、安装和训练测试命令见 [BASELINES.md](BASELINES.md)。

### EMIDEC (official NIfTI download)

Use the extracted dataset directory containing `Case*/Images/*.nii[.gz]` and
`Case*/Contours/*.nii[.gz]`. Wrapper directories are discovered recursively.
No conversion or MnM metadata CSV is required. Native labels are retained:
0 background, 1 cavity, 2 normal myocardium, 3 infarction, 4 no-reflow.
All registered models use five output classes for EMIDEC.

```bash
python main.py --dataname EMIDEC --path_image /data/EMIDEC --modelname Wave --folds 0
python test.py --dataname EMIDEC --path_image /data/EMIDEC --modelname Wave --fold 0 --pretrained checkpoint/EMIDEC/Wave/best_fold_0.pt --save_pred
```

The first run creates `EMIDEC_Training_5folds.csv` from labelled cases using
seed 100 and patient-level five-fold splits (balanced across N/P case suffixes).
Keep this file for reproducibility; `--csv_path` selects a custom split file.
Test evaluates held-out labelled patients, reports four foreground Dice scores,
and optionally exports predictions in original geometry with original labels.
The official challenge test download has no public contours and cannot be used
for Dice evaluation. Training and validation require contours.
Official dataset specification: https://emidec.com/dataset

EMIDEC fixed five-fold split is provided in `EMIDEC_Training_5folds.csv`, generated
from the 100 labelled official cases (33 normal / 67 pathological) with seed 100.
It uses the same schema as ACDC: `ID,Fold_0,Fold_1,Fold_2,Fold_3,Fold_4`, where
1 means training and 0 means held-out validation/test. Each fold has 80 training
and 20 held-out patients; each patient is held out exactly once. Normal/pathological
held-out counts are 7/13, 7/13, 7/13, 6/14 and 6/14. See `EMIDEC_5fold_summary.csv`.
This is a project-defined reproducible split, not an official challenge split.

Run all five folds on the local dataset:

```bash
python main.py --dataname EMIDEC --path_image D:/data_download/EMIDEC/EMIDEC --folds 0 1 2 3 4
```

### Test-time 3D Hausdorff distance

`test.py` also reports symmetric maximum Hausdorff distance (HD, not HD95)
for each foreground class on reconstructed original-resolution 3D volumes.
Distances use NIfTI spacing in millimetres (XYZ spacing reversed for ZYX arrays).
The volume CSV adds `HD_<class>_mm` and `HD_mean_mm`; the patient CSV retains
ED/ES values and their arithmetic means. EMIDEC has one LGE volume per patient.
The terminal reports patient-level class means and their macro mean.
Both masks empty yields HD=0; exactly one empty yields HD=inf, retained in
CSV and averages rather than dropped. This makes missing structures explicit.
The normal test command enables HD automatically; no additional dependencies
or command flags are required.

### WaveFour

WaveFour retains WaveDe's shared ResNet-50 image encoder, two-channel
MobileNetV2 difference encoder, image-decoder wavelet refinement and direct
additive encoder skips. It replaces all four difference-decoder deformable
transformers and the final difference transformer with Wave's `SpectralSCSE`:
FFT2 -> independent real/imaginary channel-SE residual enhancement -> IFFT2.
The existing difference transposed-convolution upsampling remains.
This reuses Wave's Fourier operator on decoder states, not Wave's encoder-skip
placement. No deformable attention remains; checkpoints are model-specific.

```bash
python main.py --modelname WaveFour --dataname EMIDEC --path_image /data/EMIDEC --folds 0 1 2 3 4
python test.py --modelname WaveFour --dataname EMIDEC --path_image /data/EMIDEC --fold 0 --pretrained checkpoint/EMIDEC/WaveFour/best_fold_0.pt --save_pred
```
