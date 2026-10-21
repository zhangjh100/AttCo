# ACDC 对比模型接入说明

当前项目的训练数据集是 ACDC，标签为背景=0、RV=1、MYO=2、LV=3。全部模型使用同一份患者级五折 CSV；ED/ES 不会跨患者或跨阶段拼接。

## 模型、论文和具体版本

| `--modelname` | 原生输入 | 本项目采用的版本 | 论文 / 官方实现 |
|---|---|---|---|
| Mamba-UNet | `[B,1,H,W]` 单切片；官方内部重复为 RGB | 官方 MambaUnet，vmamba_tiny，depths=[2,2,2,2]，224² | [Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation, 2024](https://arxiv.org/abs/2402.05079) / [源码](https://github.com/ziyangwang007/Mamba-UNet) |
| EMCAD | `[B,1,H,W]`；官方学习式 1→3 通道 | PVTv2-B2 + EMCAD | [Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html) / [源码](https://github.com/SLDGroup/EMCAD) |
| CASCADE | `[B,1,H,W]`；官方学习式 1→3 通道 | PVT_CASCADE，PVTv2-B2 编码器 | [Medical Image Segmentation via Cascaded Attention Decoding, WACV 2023](https://github.com/SLDGroup/CASCADE) |
| EffiDec3D | `[B,1,D,H,W]` 真三维 | 官方 MedNeXt-M-K3 + EffiDec3D，32 基础通道 | [An Optimized Decoder for High-Performance and Efficient 3D Medical Image Segmentation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.html) / [源码](https://github.com/SLDGroup/EffiDec3D) |
| EfficientMedNeXt | `[B,1,D,H,W]` 真三维 | 官方 T 版本，32 基础通道，kernel_sizes=[1,3,5] | [Multi-Receptive Dilated Convolutions for Medical Image Segmentation, MICCAI 2025](https://papers.miccai.org/miccai-2025/0280-Paper4895.html) / [源码](https://github.com/SLDGroup/EfficientMedNeXt) |
| MT-UNet | `[B,1,H,W]`；官方内部重复为 RGB | 官方 MTUNet，224² | [Mixed Transformer U-Net for Medical Image Segmentation, ICASSP 2022](https://arxiv.org/abs/2111.04734) / [源码](https://github.com/Dootmaan/MT-UNet) |
| TransUNet | `[B,1,H,W]`；官方内部重复为 RGB | 官方 R50-ViT-B_16，n_skip=3，224² | [Transformers Make Strong Encoders for Medical Image Segmentation, 2021](https://arxiv.org/abs/2102.04306) / [源码](https://github.com/Beckschen/TransUNet) |
| U-Net | `[B,1,H,W]` 单切片 | 本地 PyTorch 实现，64/128/256/512/1024 通道 | [U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015](https://arxiv.org/abs/1505.04597) |

所有名称均已找到出处。EMCAD、CASCADE、EffiDec3D 是解码器方法，实验报告应写明上述编码器搭配。U-Net 按原论文采用 2D，使用 same padding 适配统一输出网格；它不是原始 valid-convolution 裁剪版的逐像素复刻。EfficientMedNeXt-T 与 EffiDec3D 的 MedNeXt-M 是已明确固定的版本，不代表其余规模版本。

Wave、WaveDe 继续使用 `[B,3,H,W]`，三个通道依次为前一张、中心张、后一张；所有 2D 基线只读中心张，内部 RGB 转换不会引入相邻切片。

## 安装与运行

在项目根目录执行：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-baselines.txt
```

Mamba-UNet 额外需要能运行 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` 的 CUDA 环境。当前 Windows 环境的 PyTorch 为 CPU 构建，无法完成该模型的实际前向/反向检查。请在兼容的 Linux/CUDA/PyTorch 环境按上游安装 mamba-ssm 及其所需的 causal-conv1d；不要直接使用本地 `.venv` 跨平台运行。代码会在缺少算子时明确报错，不会替换为其他架构。

2D 示例（把路径改成实际 ACDC training 目录）：

```powershell
python main.py --path_image D:/datasets/ACDC/training --modelname TransUNet --train_mode single --gpu_ids 0 --folds 0 --train_batch_size 4
python test.py --path_image D:/datasets/ACDC/training --modelname TransUNet --fold 0 --pretrained checkpoint/ACDC/TransUNet/best_fold_0.pt --save_pred --output_path test_predictions/TransUNet
```

3D 示例：

```powershell
python main.py --path_image D:/datasets/ACDC/training --modelname EfficientMedNeXt --volume_size 32 128 128 --train_mode single --gpu_ids 0 --folds 0 --train_batch_size 1
python test.py --path_image D:/datasets/ACDC/training --modelname EfficientMedNeXt --volume_size 32 128 128 --fold 0 --pretrained checkpoint/ACDC/EfficientMedNeXt/best_fold_0.pt --save_pred --output_path test_predictions/EfficientMedNeXt
```

`--folds 0 1 2 3 4` 运行所有折（也是默认值）。通过 `--modelname` 切换表中其他模型。多卡用 `--train_mode multi --gpu_ids 0,1`，batch size 为总量。`--base_channels` 仅控制 Wave/WaveDe；官方基线规模按上表固定。

## 输入与实验协议

- 2D 默认 224×224；MT-UNet、TransUNet、Mamba-UNet 当前固定为该尺寸，其他 2D 可用不小于 64 的 32 倍数尺寸。
- 3D 每个 ED/ES 阶段是独立样本，整幅体积重采样到 `--volume_size D H W`（默认 32×128×128，最小 32，每维为 16 倍数）。采用三维卷积，不做逐切片伪 3D。图像线性插值，标签最近邻。此处是适配 ACDC 的固定体积协议，未复现原论文在 BTCV/MSD 上的随机 patch / sliding-window 采样或物理间距预处理。
- 归一化沿用项目原有体积级非零前景 z-score。训练使用平面翻转和旋转；3D 对整个体积应用相同变换。
- 默认所有基线从零训练，保留官方构造器初始化，不自动下载或加载 ImageNet/VMamba 权重。`--pretrained` 仅用于严格加载本项目相同配置的完整模型 checkpoint，不是通用编码器权重接口。
- 共享现有 Adam、学习率、CE+foreground Dice 和 best validation loss 选择规则。这是统一对比协议，不等同于逐篇复现官方训练配方；论文中应披露预训练状态、输入分辨率、batch size、epoch 数及模型版本。2D 每 epoch 的样本数为切片数，3D 为阶段体积数，训练步数不相同。
- EMCAD/CASCADE 训练时对四个输出头分别计算损失后取均值，推理融合四头 logits；未复刻官方全部组合监督配方。3D 训练启用官方多尺度深监督，对各头插值到目标网格后计算平均损失；推理保留官方 `mode='test'` 提前输出分支，再插值恢复尺寸。
- 验证 Dice 在重采样网格按患者阶段累积；最终 test 在原始体积网格评估。先对连续 logits 做双线性/三线性恢复，再 argmax。保存的 NIfTI 保留原始 spacing/origin/direction。
- `test_results_fold_N.csv` 每个患者阶段一行；`patient_results_fold_N.csv` 每个患者一行，保留 ED、ES 和二者算术均值。三类均为空时 Dice=1，沿用原项目约定。进行显著性比较时使用患者级表。
- 每折保存 `config_fold_N.json` 记录输入大小、初始化方式、训练参数和上游提交；训练测试必须使用相同架构和输入设置。

## 源码与验证

`models/baselines.py` 是模型工厂和输出适配，`comparison_data.py` 是原生维度数据管线。`third_party/*/UPSTREAM.json` 记录下载的官方 commit，原许可证保留。

对上游架构的修改仅包括：隔离绝对 `lib`/`networks` 导入命名空间；CASCADE 支持关闭强制预训练权重加载并移除未使用可视化依赖；MT-UNet 的 `.cuda()` 和补零张量改为跟随输入设备。层结构及计算逻辑保留。`scripts/patch_baselines.py` 记录这些修改；下载脚本优先读取已记录的 commit。U-Net 为本项目实现。

```powershell
python -m unittest discover -s tests -v
python scripts/smoke_baselines.py
```

架构检查报告保存为 `output/baseline_smoke.json`。合成数据回归覆盖单切片与体积形状、ED/ES 分隔、2D/3D 一轮训练、checkpoint 加载、患者 CSV、NIfTI 原始几何恢复。Mamba CUDA 算子需在目标 GPU 环境另行检查；未执行真实 ACDC 完整收敛实验，也不能从这些检查推断论文指标。
