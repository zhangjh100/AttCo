import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import losses, transforms, dataset, metrics, utils
from torch.nn import init

# 导入你的小波损失
from models.WaveCo_Constraint_BraTS import WaveletLoss
losses.WaveletLoss = WaveletLoss


def param_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))


def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('GroupNorm') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


# ===================== 核心修复：正确计算小波损失（提取滤波器） =====================
def compute_wavelet_loss(model, wavelet_criterion, device):
    """
    自动提取模型中 fusion1, fusion2, fusion3 的小波滤波器并计算损失
    权重：fusion1=1.0, fusion2=0.5, fusion3=1/3
    """
    fusion_weights = [1.0, 0.5, 1/3]
    total_wave_loss = 0.0
    loss1 = loss2 = loss3 = 0.0

    # 处理多卡 DataParallel
    model_core = model.module if isinstance(model, nn.DataParallel) else model

    # 遍历3个fusion模块
    for idx, fusion_name in enumerate(['fusion1', 'fusion2', 'fusion3']):
        if hasattr(model_core, fusion_name):
            fusion_module = getattr(model_core, fusion_name)
            # 找到LWN3D小波模块
            for name, module in fusion_module.named_modules():
                if 'LWN3D' in str(type(module)) and hasattr(module, 'get_filters'):
                    lo, hi = module.get_filters()
                    w_loss = wavelet_criterion(lo, hi)
                    # 加权
                    weighted_loss = fusion_weights[idx] * w_loss
                    total_wave_loss += weighted_loss
                    # 记录单个损失
                    if idx == 0:
                        loss1 = w_loss.item()
                    elif idx == 1:
                        loss2 = w_loss.item()
                    else:
                        loss3 = w_loss.item()
                    break

    return total_wave_loss, loss1, loss2, loss3


# ===================== 总损失计算 =====================
def compute_total_loss(model, output, target, ce_dice_criterion, wavelet_criterion, wavelet_weight, device):
    # 分割主损失
    ce_dice_loss = ce_dice_criterion(output, target)

    # 小波约束损失
    wave_loss, w1, w2, w3 = compute_wavelet_loss(model, wavelet_criterion, device)
    total_wave_loss = wavelet_weight * wave_loss

    # 总损失
    total_loss = ce_dice_loss + total_wave_loss

    return total_loss, ce_dice_loss, total_wave_loss, [w1, w2, w3]


if __name__ == "__main__":
    date = (datetime.now()).strftime("%Y-%m-%d")
    frame = pd.read_csv("BraTS2020_Training_5folds.csv")
    for fold in range(5):
        print("Training fold: ", fold)
        parser = argparse.ArgumentParser()

        parser.add_argument('--epochs', type=int, default=400, help='epoch number')
        parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
        parser.add_argument('--train_batch_size', type=int, default=4, help='training batch size')
        parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size')
        parser.add_argument('--is_thop', type=bool, default=True, help='whether calculate FLOPs/Params')
        parser.add_argument('--path_image', type=str,
                            default="/mnt/data1/zhangjh/datasets/multimodal/BraTS2020/MICCAI_BraTS2020_TrainingData",
                            help='')
        parser.add_argument('--pretrained', type=str, default=None, help='')
        parser.add_argument('--modelname', type=str, default="WaveCo_Constraint", help='model name')
        parser.add_argument('--dataname', type=str, default="BraTS2020", help='dataset name')
        # 小波损失权重
        parser.add_argument('--wavelet_loss_weight', type=float, default=1.0, help='小波损失全局权重')

        arg = parser.parse_args()

        np.random.seed(100)
        torch.manual_seed(100)

        # 设备配置
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_ids = [0, 1]
            assert len(gpu_ids) <= torch.cuda.device_count()
            torch.cuda.set_device(gpu_ids[0])
        else:
            device = torch.device('cpu')
            gpu_ids = []

        # 模型初始化
        if arg.modelname == "WaveCo_Constraint":
            import models.WaveCo_Constraint_BraTS as net
            model = net.WaveCo_Constraint(inChannel=2, outChannel=4, baseChannel=16)

        # 数据变换
        train_transforms = transforms.Compose([
            transforms.NormalizeIntensity(),
            transforms.RandomCrop(margin=(16, 16, 16), target_size=(128, 128, 128), original_size=(155, 240, 240)),
            transforms.Mirroring(p=0.5),
            transforms.ToTensor()])
        val_transforms = transforms.Compose([
            transforms.NormalizeIntensity(),
            transforms.RandomCrop(margin=(0, 0, 0), target_size=(128, 128, 128), original_size=(155, 240, 240)),
            transforms.ToTensor()])

        # 数据集
        listTrainPatients = list(frame["ID"][frame["Fold_{}".format(fold)] == 1])
        listValPatients = list(frame["ID"][frame["Fold_{}".format(fold)] == 0])
        train_set = dataset.MedDataset(arg.path_image, listTrainPatients, transforms=train_transforms, mode="train")
        val_set = dataset.MedDataset(arg.path_image, listValPatients, transforms=val_transforms, mode="val")

        # 数据加载器
        train_loader = DataLoader(train_set, batch_size=arg.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=arg.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        dataloaders = {'train': train_loader, 'val': val_loader}

        # 模型部署
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        # 预训练权重
        if arg.pretrained is not None:
            print("Loading pretrained model")
            pretrain = torch.load(arg.pretrained, map_location=device)
            model.load_state_dict(pretrain, strict=False)
        else:
            print("Training from Scratch")
            if isinstance(model, nn.DataParallel):
                model.module.apply(init_weights)
            else:
                model.apply(init_weights)

        param_network(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lrate, betas=(0.9, 0.99))

        # 损失函数
        criterion = losses.CE_GeneralizedSoftDiceLoss().to(device)
        wavelet_criterion = WaveletLoss(device=device).to(device)
        dice_metric = metrics.DiceMetrics()

        min_loss = np.inf
        log_data = []

        for epoch in range(arg.epochs):
            # 训练指标初始化
            loss_train = 0.0
            ce_dice_train = 0.0
            wave_total_train = 0.0
            wave1_train = 0.0
            wave2_train = 0.0
            wave3_train = 0.0
            dice_train = [0]*4

            model.train()
            for sample in tqdm(dataloaders['train']):
                input, target = sample["input"].to(device), sample["target"].long().to(device)
                output = model(input)  # 模型单输出，无报错！

                # 计算损失
                loss, ce_loss, wave_loss, wave_list = compute_total_loss(
                    model, output, target, criterion, wavelet_criterion, arg.wavelet_loss_weight, device
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累计损失
                loss_train += loss.item()
                ce_dice_train += ce_loss.item()
                wave_total_train += wave_loss.item()
                wave1_train += wave_list[0]
                wave2_train += wave_list[1]
                wave3_train += wave_list[2]

                # Dice
                dice = dice_metric(output, target)
                for i in range(4):
                    dice_train[i] += dice[i].item()

            # 训练集平均
            n_train = len(dataloaders['train'])
            loss_train /= n_train
            ce_dice_train /= n_train
            wave_total_train /= n_train
            wave1_train /= n_train
            wave2_train /= n_train
            wave3_train /= n_train
            for i in range(4):
                dice_train[i] /= n_train

            # 验证
            loss_val = 0.0
            ce_dice_val = 0.0
            wave_total_val = 0.0
            wave1_val = 0.0
            wave2_val = 0.0
            wave3_val = 0.0
            dice_val = [0]*4

            model.eval()
            with torch.no_grad():
                for sample in tqdm(dataloaders['val']):
                    input, target = sample["input"].to(device), sample["target"].long().to(device)
                    output = model(input)

                    loss, ce_loss, wave_loss, wave_list = compute_total_loss(
                        model, output, target, criterion, wavelet_criterion, arg.wavelet_loss_weight, device
                    )

                    loss_val += loss.item()
                    ce_dice_val += ce_loss.item()
                    wave_total_val += wave_loss.item()
                    wave1_val += wave_list[0]
                    wave2_val += wave_list[1]
                    wave3_val += wave_list[2]

                    dice = dice_metric(output, target)
                    for i in range(4):
                        dice_val[i] += dice[i].item()

            # 验证集平均
            n_val = len(dataloaders['val'])
            loss_val /= n_val
            ce_dice_val /= n_val
            wave_total_val /= n_val
            wave1_val /= n_val
            wave2_val /= n_val
            wave3_val /= n_val
            for i in range(4):
                dice_val[i] /= n_val

            # 保存最优模型
            if min_loss > loss_val:
                min_loss = loss_val
                filename = f"/mnt/data1/zhangjh/AttCo/checkpoint/{arg.dataname}/{arg.modelname}/Fold_{fold}_bs_{arg.train_batch_size}_TC_{dice_val[0]:.4f}_ED_{dice_val[1]:.4f}_ET_{dice_val[2]:.4f}_WT_{dice_val[3]:.4f}.pt"
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), filename)
                print("Saving model: ", filename)

            # 打印日志
            print(f"[Epoch {epoch}]")
            print(f"Train | CE-Dice: {ce_dice_train:.4f} | Wave1: {wave1_train:.4f} | Wave2: {wave2_train:.4f} | Wave3: {wave3_train:.4f} | Total: {loss_train:.4f}")
            print(f"Val   | CE-Dice: {ce_dice_val:.4f} | Wave1: {wave1_val:.4f} | Wave2: {wave2_val:.4f} | Wave3: {wave3_val:.4f} | Total: {loss_val:.4f}")
            print(f"Dice TC: {dice_train[0]:.4f} | ED: {dice_train[1]:.4f} | ET: {dice_train[2]:.4f} | WT: {dice_train[3]:.4f}")

            # 保存日志
            log_data.append([
                epoch, loss_train, ce_dice_train, wave_total_train, wave1_train, wave2_train, wave3_train,
                dice_train[0], dice_train[1], dice_train[2], dice_train[3],
                loss_val, ce_dice_val, wave_total_val, wave1_val, wave2_val, wave3_val,
                dice_val[0], dice_val[1], dice_val[2], dice_val[3]
            ])

            log_df = pd.DataFrame(log_data, columns=[
                'Epoch','Loss_train','CE_Dice_train','Wave_Total_train','Wave1_train','Wave2_train','Wave3_train',
                'Dice_TC_train','Dice_ED_train','Dice_ET_train','Dice_WT_train',
                'Loss_val','CE_Dice_val','Wave_Total_val','Wave1_val','Wave2_val','Wave3_val',
                'Dice_TC_val','Dice_ED_val','Dice_ET_val','Dice_WT_val'
            ])
            log_df.to_csv(f"/mnt/data1/zhangjh/AttCo/checkpoint/{arg.dataname}/{arg.modelname}/log_fold_{fold}.csv", index=False)