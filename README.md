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
