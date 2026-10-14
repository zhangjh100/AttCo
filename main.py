import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import losses, transforms, dataset, metrics
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


def parse_gpu_ids(value):
    try:
        ids = [int(part.strip()) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU ids must be comma-separated integers") from exc
    if not ids or any(i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("GPU ids must be distinct non-negative integers")
    return ids


def select_device(mode, gpu_ids):
    if mode == 'cpu':
        return torch.device('cpu'), []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --train_mode cpu to run on CPU")
    if mode == 'single' and len(gpu_ids) != 1:
        raise ValueError("single mode requires exactly one GPU id")
    if mode == 'multi' and len(gpu_ids) < 2:
        raise ValueError("multi mode requires at least two GPU ids")
    count = torch.cuda.device_count()
    if any(i >= count for i in gpu_ids):
        raise ValueError(f"Requested {gpu_ids}, but only {count} CUDA devices are visible")
    torch.cuda.set_device(gpu_ids[0])
    return torch.device(f'cuda:{gpu_ids[0]}'), gpu_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=400, help='epoch number')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size')
    parser.add_argument('--is_thop', type=bool, default=True, help='whether calculate FLOPs/Params')
    parser.add_argument('--path_image', type=str,
                        default="/datapool/home/ph_teacher1/zhangjh/datasets/multimodal/BraTS2020/MICCAI_BraTS2020_TrainingData",
                        help='')
    parser.add_argument('--pretrained', type=str, default=None, help='')
    parser.add_argument('--modelname', type=str, default="Wave", help='model name')
    parser.add_argument('--dataname', type=str, default="BraTS2020", help='dataset name')
    parser.add_argument('--train_mode', choices=('single', 'multi', 'cpu'), default='single')
    parser.add_argument('--gpu_ids', type=parse_gpu_ids, default=[0],
                        help='Visible GPU ids, e.g. 0 or 0,1; batch size is total across GPUs')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--csv_path', default='BraTS2020_Training_5folds.csv')
    parser.add_argument('--checkpoint_dir', default='./checkpoint')

    arg = parser.parse_args()

    if any(fold not in range(5) for fold in arg.folds) or len(set(arg.folds)) != len(arg.folds):
        parser.error("--folds must contain distinct values from 0 to 4")
    frame = pd.read_csv(arg.csv_path)
    for fold in arg.folds:
        print("Training fold:", fold)
        np.random.seed(100)
        torch.manual_seed(100)

        device, gpu_ids = select_device(arg.train_mode, arg.gpu_ids)
        if arg.train_mode == 'multi' and arg.train_batch_size < len(gpu_ids):
            raise ValueError("Total training batch size must be at least the GPU count")
        print(f"Mode: {arg.train_mode}; device: {device}; GPU ids: {gpu_ids}")
        save_dir = os.path.join(arg.checkpoint_dir, arg.dataname, arg.modelname)
        os.makedirs(save_dir, exist_ok=True)

        # 模型初始化
        if arg.modelname == "WaveCo_Constraint":
            import models.WaveCo_Constraint_BraTS as net
            model = net.WaveCo_Constraint(inChannel=2, outChannel=4, baseChannel=16)

        else:
            raise ValueError(f"Unsupported model: {arg.modelname}")

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
        train_loader = DataLoader(train_set, batch_size=arg.train_batch_size, shuffle=True, num_workers=8, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_set, batch_size=arg.val_batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
        dataloaders = {'train': train_loader, 'val': val_loader}

        # Load on the unwrapped model: both single/multi checkpoints are supported.
        if arg.pretrained is not None:
            pretrain = torch.load(arg.pretrained, map_location='cpu')
            if isinstance(pretrain, dict) and 'state_dict' in pretrain:
                pretrain = pretrain['state_dict']
            pretrain = {k.removeprefix('module.'): v for k, v in pretrain.items()}
            model.load_state_dict(pretrain, strict=True)
            print("Loaded pretrained model")
        else:
            print("Training from Scratch")
            init_weights(model)  # Exactly once, rather than nested apply calls.
        model = model.to(device)
        if arg.train_mode == 'multi':
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

        param_network(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lrate, betas=(0.9, 0.99))

        # 损失函数
        criterion = losses.CE_GeneralizedSoftDiceLoss().to(device)
        dice_metric = metrics.DiceMetrics()

        min_loss = np.inf
        log_data = []

        for epoch in range(arg.epochs):
            loss_train = 0.0
            dice_train = [0]*4

            model.train()
            for sample in tqdm(dataloaders['train']):
                input, target = sample["input"].to(device), sample["target"].long().to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(input)

                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                loss_train += loss.item()

                # Dice
                dice = dice_metric(output.detach(), target)
                for i in range(4):
                    dice_train[i] += dice[i].item()

            n_train = len(dataloaders['train'])
            if n_train == 0:
                raise ValueError("Training loader is empty")
            loss_train /= n_train
            for i in range(4):
                dice_train[i] /= n_train

            loss_val = 0.0
            dice_val = [0]*4

            model.eval()
            with torch.no_grad():
                for sample in tqdm(dataloaders['val']):
                    input, target = sample["input"].to(device), sample["target"].long().to(device)
                    output = model(input)

                    loss = criterion(output, target)

                    loss_val += loss.item()

                    dice = dice_metric(output.detach(), target)
                    for i in range(4):
                        dice_val[i] += dice[i].item()

            n_val = len(dataloaders['val'])
            if n_val == 0:
                raise ValueError("Validation loader is empty")
            loss_val /= n_val
            for i in range(4):
                dice_val[i] /= n_val

            # 保存最优模型
            if min_loss > loss_val:
                min_loss = loss_val
                filename = os.path.join(save_dir, f'best_fold_{fold}.pt')
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), filename)
                print("Saving model: ", filename)

            print(f"[Epoch {epoch + 1}/{arg.epochs}] Train loss: {loss_train:.4f} | Val loss: {loss_val:.4f}")
            print("Train Dice:", dict(zip(('TC', 'ED', 'ET', 'WT'), dice_train)))
            print("Val Dice:", dict(zip(('TC', 'ED', 'ET', 'WT'), dice_val)))
            log_data.append([epoch + 1, loss_train, *dice_train, loss_val, *dice_val])
            columns = ['Epoch', 'Loss_train', 'Dice_TC_train', 'Dice_ED_train',
                       'Dice_ET_train', 'Dice_WT_train', 'Loss_val', 'Dice_TC_val',
                       'Dice_ED_val', 'Dice_ET_val', 'Dice_WT_val']
            pd.DataFrame(log_data, columns=columns).to_csv(
                os.path.join(save_dir, f'log_fold_{fold}.csv'), index=False)
