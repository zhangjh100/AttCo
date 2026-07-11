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
from models.WaveCo_Constraint_BraTS import WaveCo_Constraint


def save_nii(data, path, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


if __name__ == "__main__":
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    test_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.RandomCrop(margin=(0, 0, 0), target_size=(128, 128, 128), original_size=(155, 240, 240)),
        transforms.ToTensor()
    ])

    frame = pd.read_csv(args.csv_path)
    listTestPatients = list(frame["ID"][frame[f"Fold_{args.fold}"] == 0])

    test_set = dataset.MedDataset(args.path_image, listTestPatients, transforms=test_transforms, mode="val")
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = WaveCo_Constraint(inChannel=2, outChannel=4, baseChannel=16)  # 训练时用24就改成24！
    model = model.to(device)

    print(f"加载模型权重: {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model.load_state_dict(checkpoint.state_dict())
    else:
        if isinstance(checkpoint, dict) and 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    # if isinstance(checkpoint, dict) and 'module.' in list(checkpoint.keys())[0]:
    #     from collections import OrderedDict
    #
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint.items():
    #         new_state_dict[k.replace('module.', '')] = v
    #     model.load_state_dict(new_state_dict)
    # else:
    #     model.load_state_dict(checkpoint)

    dice_metric = metrics.DiceMetrics()
    total_dice = [0.0] * 4
    num_samples = len(test_loader)

    if args.save_pred:
        os.makedirs(args.output_path, exist_ok=True)
        print(f"预测结果将保存到: {args.output_path}")

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_loader, desc="测试进度")):
            input_img = sample["input"].to(device)
            target = sample["target"].type(torch.LongTensor).to(device)
            patient_id = sample["id"][0]

            output = model(input_img)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).cpu().numpy()

            # 计算Dice指标
            dice = dice_metric(output, target)
            for i in range(4):
                total_dice[i] += dice[i].item()

            if args.save_pred:
                save_nii(pred, os.path.join(args.output_path, f"{patient_id}_pred.nii.gz"))

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