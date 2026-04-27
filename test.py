import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
from tqdm import tqdm
import pandas as pd

# 100%复用仓库现有模块，无需修改任何已有代码
import dataset
import transforms
import metrics
from models.AttCo_BraTS import AttCo  # 导入BraTS模型（AutoPET请替换为AttCo_AutoPET）
from models.WaveCo_BraTS import WaveCo


def save_nii(data, path, affine=np.eye(4)):
    """将3D数组保存为医学影像标准nii.gz格式，用于ITK-SNAP/3D Slicer查看"""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


if __name__ == "__main__":
    # -------------------------- 1. 命令行参数（完全兼容服务器调用） --------------------------
    parser = argparse.ArgumentParser(description="WaveCo 模型测试脚本")
    parser.add_argument('--modelname', type=str, default="WaveCo", help='模型名称')
    parser.add_argument('--dataname', type=str, default="BraTS2020", help='数据集名称')
    parser.add_argument('--fold', type=int, default=0, help='交叉验证折数(0-4)')
    parser.add_argument('--test_batch_size', type=int, default=1, help='3D数据固定为1')
    parser.add_argument('--path_image', type=str, required=True, help='测试数据集根目录')
    parser.add_argument('--csv_path', type=str, default="BraTS2020_Training_5folds.csv", help='5折划分CSV路径')
    parser.add_argument('--pretrained', type=str, required=True, help='训练好的模型权重(.pt)路径')
    parser.add_argument('--save_pred', action='store_true', default=False, help='是否保存预测结果')
    parser.add_argument('--output_path', type=str, default="./test_predictions", help='预测结果保存目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU卡号')

    args = parser.parse_args()

    # -------------------------- 2. 设备配置（服务器多卡适配） --------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # -------------------------- 3. 数据预处理（必须和训练时VAL的变换完全一致！） --------------------------
    # 绝对不能用train的随机增强，否则结果完全错误
    test_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.RandomCrop(margin=(0, 0, 0), target_size=(128, 128, 128), original_size=(155, 240, 240)),
        transforms.ToTensor()
    ])

    # -------------------------- 4. 加载测试数据集 --------------------------
    frame = pd.read_csv(args.csv_path)
    # 方式1：加载对应折的验证集（用于5折模型评估）
    listTestPatients = list(frame["ID"][frame[f"Fold_{args.fold}"] == 0])
    # 方式2：加载独立测试集（无CSV，直接遍历文件夹，取消注释即可用）
    # listTestPatients = sorted([d for d in os.listdir(args.path_image) if os.path.isdir(os.path.join(args.path_image, d))])

    test_set = dataset.MedDataset(args.path_image, listTestPatients, transforms=test_transforms, mode="val")
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = WaveCo(inChannel=2, outChannel=4, baseChannel=16)  # 训练时用24就改成24！
    model = model.to(device)

    print(f"加载模型权重: {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        # 如果加载的是模型实例，直接提取其state_dict
        model.load_state_dict(checkpoint.state_dict())
    else:
        # 原有逻辑：处理正常state_dict（含多卡module.前缀）
        if isinstance(checkpoint, dict) and 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

'''
    if isinstance(checkpoint, dict) and 'module.' in list(checkpoint.keys())[0]:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
'''

    model.eval()

    # -------------------------- 7. 初始化评估指标（和训练时完全一致） --------------------------
    dice_metric = metrics.DiceMetrics()
    total_dice = [0.0] * 4  # TC(肿瘤核心)、ED(水肿)、ET(增强肿瘤)、WT(全肿瘤)
    num_samples = len(test_loader)

    # -------------------------- 8. 测试循环（推理+评估+保存结果） --------------------------
    if args.save_pred:
        os.makedirs(args.output_path, exist_ok=True)
        print(f"预测结果将保存到: {args.output_path}")

    with torch.no_grad():  # 禁用梯度，节省显存+加速推理
        for idx, sample in enumerate(tqdm(test_loader, desc="测试进度")):
            # 加载数据
            input_img = sample["input"].to(device)
            target = sample["target"].type(torch.LongTensor).to(device)
            patient_id = sample["id"][0]  # 获取患者ID，用于结果命名

            # 模型推理
            output = model(input_img)
            # 后处理：取argmax得到最终分割掩码（4类→1类）
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).cpu().numpy()

            # 计算Dice指标
            dice = dice_metric(output, target)
            for i in range(4):
                total_dice[i] += dice[i].item()

            # 保存预测结果（nii.gz格式，医学影像标准）
            if args.save_pred:
                save_nii(pred, os.path.join(args.output_path, f"{patient_id}_pred.nii.gz"))
                # 可选：保存真实标签用于对比
                # save_nii(target_np, os.path.join(args.output_path, f"{patient_id}_gt.nii.gz"))

    # -------------------------- 9. 计算平均指标并输出 --------------------------
    avg_dice = [d / num_samples for d in total_dice]
    print("\n" + "=" * 60)
    print(f"【Fold {args.fold} 测试结果】")
    print(f"肿瘤核心(TC) 平均Dice: {avg_dice[0]:.4f}")
    print(f"水肿(ED) 平均Dice: {avg_dice[1]:.4f}")
    print(f"增强肿瘤(ET) 平均Dice: {avg_dice[2]:.4f}")
    print(f"全肿瘤(WT) 平均Dice: {avg_dice[3]:.4f}")
    print("=" * 60)

    # -------------------------- 10. 保存指标到CSV（方便后续统计） --------------------------
    result_df = pd.DataFrame({
        "Fold": [args.fold],
        "Dice_TC": [avg_dice[0]],
        "Dice_ED": [avg_dice[1]],
        "Dice_ET": [avg_dice[2]],
        "Dice_WT": [avg_dice[3]]
    })
    result_df.to_csv(f"test_results_fold_{args.fold}.csv", index=False)
    print(f"指标结果已保存到: test_results_fold_{args.fold}.csv")