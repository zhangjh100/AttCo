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


if __name__ == "__main__":
    date = (datetime.now()).strftime("%Y-%m-%d")
    frame = pd.read_csv("BraTS2020_Training_5folds.csv")
    for fold in range(5):
        print("Training fold: ", fold)
        parser = argparse.ArgumentParser()
        # -------------------------- 1. 调整默认batch size（两卡建议翻倍） --------------------------
        parser.add_argument('--epochs', type=int, default=400, help='epoch number')
        parser.add_argument('--lrate', type=float, default=0.002, help='learning rate（batch翻倍，学习率可选翻倍）')
        parser.add_argument('--train_batch_size', type=int, default=4, help='training batch size（原2→两卡改4）')
        parser.add_argument('--val_batch_size', type=int, default=2, help='validation batch size')
        parser.add_argument('--is_thop', type=bool, default=True, help='whether calculate FLOPs/Params (Thop)')
        parser.add_argument('--path_image', type=str,
                            default="/mnt/data1/zhangjh/datasets/multimodal/BraTS2020/MICCAI_BraTS2020_TrainingData",
                            help='')
        parser.add_argument('--pretrained', type=str, default=None, help='')
        parser.add_argument('--modelname', type=str, default="WaveCo", help='Choose one of models: ')
        parser.add_argument('--dataname', type=str, default="BraTS2020", help='Choose one of models: ')

        arg = parser.parse_args()

        np.random.seed(100)
        torch.manual_seed(100)

        # -------------------------- 2. 多卡设备配置 --------------------------
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # 指定要使用的两张GPU（例如0和1号卡）
            gpu_ids = [0, 1]
            # 校验GPU数量是否足够
            assert len(
                gpu_ids) <= torch.cuda.device_count(), f"仅检测到{torch.cuda.device_count()}张GPU，无法使用{len(gpu_ids)}张"
            torch.cuda.set_device(gpu_ids[0])  # 设置主卡
        else:
            device = torch.device('cpu')
            gpu_ids = []

        # 模型初始化
        if arg.modelname == "WaveCo":
            import models.WaveCo_BraTS as net

            model = net.WaveCo(inChannel=2, outChannel=4, baseChannel=16)

        # 数据变换（不变）
        train_transforms = transforms.Compose([
            transforms.NormalizeIntensity(),
            transforms.RandomCrop(margin=(16, 16, 16), target_size=(128, 128, 128), original_size=(155, 240, 240)),
            transforms.Mirroring(p=0.5),
            transforms.ToTensor()])
        val_transforms = transforms.Compose([
            transforms.NormalizeIntensity(),
            transforms.RandomCrop(margin=(0, 0, 0), target_size=(128, 128, 128), original_size=(155, 240, 240)),
            transforms.ToTensor()])

        # 数据集划分（不变）
        listTrainPatients = list(frame["ID"][frame["Fold_{}".format(fold)] == 1])
        listValPatients = list(frame["ID"][frame["Fold_{}".format(fold)] == 0])
        train_set = dataset.MedDataset(arg.path_image, listTrainPatients, transforms=train_transforms, mode="train")
        val_set = dataset.MedDataset(arg.path_image, listValPatients, transforms=val_transforms, mode="val")

        # -------------------------- 3. 数据加载器（多卡建议增大num_workers） --------------------------
        train_loader = DataLoader(train_set, batch_size=arg.train_batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=arg.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        dataloaders = {'train': train_loader, 'val': val_loader}

        # -------------------------- 4. 封装多卡模型 --------------------------
        print(arg.modelname)
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        # -------------------------- 5. 预训练权重加载（适配多卡module前缀） --------------------------
        if arg.pretrained is not None:
            print("Loading pretrained model")
            pretrain = torch.load(arg.pretrained, map_location=device)
            if isinstance(pretrain, (torch.nn.DataParallel, torch.nn.parallel.DataParallel)):
                pretrained_state_dict = pretrain.module.state_dict()
            else:
                pretrained_state_dict = pretrain.state_dict() if hasattr(pretrain, 'state_dict') else pretrain

            model_dict = model.state_dict()
            # 匹配参数key（处理module.前缀）
            new_pretrained_dict = {}
            for k, v in pretrained_state_dict.items():
                if k in model_dict:
                    new_pretrained_dict[k] = v
                elif 'module.' + k in model_dict:
                    new_pretrained_dict['module.' + k] = v
                else:
                    print(f"警告：参数{k}未在当前模型中找到，跳过")
            model_dict.update(new_pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("Training from Scratch")
            if isinstance(model, torch.nn.DataParallel):
                model.module.apply(init_weights)
            else:
                model.apply(init_weights)

        param_network(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lrate, betas=(0.9, 0.99))
        criterion = losses.CE_GeneralizedSoftDiceLoss()
        dice_metric = metrics.DiceMetrics()

        min_loss = np.inf
        log_data = []
        for epoch in range(arg.epochs):
            loss_train, loss_val = 0, 0
            dice_train, dice_val = [0] * 4, [0] * 4
            model.train()
            for sample in tqdm(dataloaders['train']):
                input, target = sample["input"].to(device), sample["target"].type(torch.LongTensor).to(device)
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                dice = dice_metric(output, target)
                loss_train += loss.item()
                for idx_dice in range(4):
                    dice_train[idx_dice] += dice[idx_dice].item()

            loss_train /= len(dataloaders['train'])
            for idx_dice in range(4):
                dice_train[idx_dice] /= len(dataloaders['train'])

            model.eval()
            with torch.no_grad():
                for sample in tqdm(dataloaders['val']):
                    input, target = sample["input"].to(device), sample["target"].type(torch.LongTensor).to(device)
                    output = model(input)
                    loss = criterion(output, target)
                    dice = dice_metric(output, target)

                    loss_val += loss.item()
                    for idx_dice in range(4):
                        dice_val[idx_dice] += dice[idx_dice].item()

            loss_val /= len(dataloaders['val'])
            for idx_dice in range(4):
                dice_val[idx_dice] /= len(dataloaders['val'])

            if min_loss > loss_val:
                min_loss = loss_val
                filename = f"/mnt/data1/zhangjh/AttCo/checkpoint/{arg.dataname}/{arg.modelname}/Fold_{fold}_bs_{arg.train_batch_size}_TC_{dice_val[0]}_ED_{dice_val[1]}_ET_{dice_val[2]}_WT_{dice_val[3]}.pt"
                if os.path.exists(filename):
                    os.remove(filename)
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), filename)
                else:
                    torch.save(model.state_dict(), filename)
                print("Saving model: ", filename)

            print(
                f"Epoch: {epoch} | Loss_train: {loss_train:.04f} | Dice_TC: {dice_train[0]:.04f} | Dice_ED: {dice_train[1]:.04f} | Dice_ET {dice_train[2]:.04f} | Dice_WT: {dice_train[3]:.04f}")
            print(
                f"Epoch: {epoch} | Loss_val: {loss_val:.04f} | Dice_TC: {dice_val[0]:.04f} | Dice_ED: {dice_val[1]:.04f} | Dice_ET {dice_val[2]:.04f} | Dice_WT: {dice_val[3]:.04f}")
            log_data.append(
                [epoch, loss_train, dice_train[0], dice_train[1], dice_train[2], dice_train[3], loss_val, dice_val[0],
                 dice_val[1], dice_val[2], dice_val[3]])

            log_data_frame = np.asarray(log_data)
            log_data_frame = pd.DataFrame(log_data_frame, columns=[
                ['Epoch', 'Loss_train', 'Dice_TC_train', 'Dice_ED_train', 'Dice_ET_train', 'Dice_WT_train', 'Loss_val',
                 'Dice_TC_val', 'Dice_ED_val', 'Dice_ET_val', 'Dice_WT_val']])
            log_data_frame.to_csv(
                f"/mnt/data1/zhangjh/AttCo/checkpoint/{arg.dataname}/{arg.modelname}/log_train_BraTS2020_fold_{fold}_bs_{arg.train_batch_size}.csv",
                index=False)