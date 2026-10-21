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
