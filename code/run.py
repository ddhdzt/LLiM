import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import get_args, get_different_task_params, get_dataset, get_model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from basic_model import DLinear, TimesNet, MLP, Pyraformer
import os

"""
task_num in [1, 3, 4]:
    1 -- capacity
    3 -- riding
    4 -- except
"""
task_num = 3
used_device = 3
date = f'0802_{task_num}_{used_device}_0'
task_name = ['', 'capacity', '', 'riding', 'except']

os.environ["CUDA_VISIBLE_DEVICES"] = f"{used_device}"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_continue(model, device, train_loader, test_loader, optimizer, epoch, scaler):
    model.train()
    criterion = nn.MSELoss()

    losses = 0.
    batch_idx = 1
    iter_loader_test = 6
    count = 0
    need_test_ietm = len(train_loader) // iter_loader_test
    mae_, mse_ = [], []
    for batch in tqdm(train_loader):
        if batch_idx == need_test_ietm:
            mae_temp, mse_temp = eval_continue(model, device, test_loader, (epoch - 1) * iter_loader_test + count)
            mae_.append(mae_temp.cpu().numpy()), mse_.append(mse_temp.cpu().numpy())
            batch_idx = 1
            count += 1
        data, target, _, _, _ = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():
            predict = model(data)
            loss = torch.sqrt(criterion(predict, target.float()))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += loss.item()
        batch_idx += 1

    print('[Down Task Train Epoch: {}], Average loss: {:.4f}'.format(epoch, losses / len(train_loader)))
    return mae_, mse_


def eval_continue(model, device, test_loader, epoch):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    losses = 0.
    diffs = 0.
    data_list = []
    predict_list = []
    circle_list = []
    temp_list = []
    data_29_list = []
    with torch.no_grad():
        for data, target, circle, temp, data_29 in test_loader:
            data, target = data.to(device), target.to(device)
            predict = model(data)
            diff = torch.abs(target - predict)
            diffs += torch.sum(diff)
            loss = criterion(predict, target)
            losses += loss

            circle_list.append(circle)
            temp_list.append(temp)
            data_29_list.append(data_29)
            data_list.append(target.cpu().numpy().reshape(-1))
            predict_list.append(predict.cpu().numpy().reshape(-1))

    path = f'./out/{date}/1B/'
    if not os.path.exists(path):
        os.makedirs(path)
    data_list = np.concatenate(data_list, axis=0)
    predict_list = np.concatenate(predict_list, axis=0)

    circle_list = pd.DataFrame(np.concatenate(circle_list, axis=0))
    temp_list = pd.DataFrame(np.concatenate(temp_list, axis=0))
    data_29_list = pd.DataFrame(np.concatenate(data_29_list, axis=0))
    recoder_data = pd.DataFrame({'target': data_list, 'predict': predict_list})
    recoder_data.to_csv(path + f'{epoch}.csv')
    circle_list.to_csv(path + f'circle_list{epoch}.csv')
    temp_list.to_csv(path + f'temp_list{epoch}.csv')
    data_29_list.to_csv(path + f'data_29_list{epoch}.csv')

    rmse = torch.sqrt(losses / len(test_loader.dataset))
    print('[Down Task Test Epoch: {}], Mean Difference: {:.4f}, MSE loss: {:.4f}'.format(epoch, diffs / len(
        test_loader.dataset), rmse ** 2))
    return diffs / len(test_loader.dataset), rmse ** 2


def train_classes(args, model, device, train_loader, test_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    batch_idx = 1
    iter_loader_test = 5
    count = 0
    need_test_ietm = len(train_loader) // iter_loader_test
    test_loss, acc, precision, recall, auc, f1 = [], [], [], [], [], []
    for batch in tqdm(train_loader):
        if batch_idx == need_test_ietm:
            test_loss_temp, acc_temp, precision_temp, recall_temp, auc_temp, f1_temp = (
                eval_classes(model, device, test_loader, task_name, (epoch - 1) * iter_loader_test + count))
            test_loss.append(test_loss_temp), acc.append(acc_temp), precision.append(precision_temp), recall.append(
                recall_temp), auc.append(auc_temp), f1.append(f1_temp)
            batch_idx = 1
            count += 1
        data, target = batch
        data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(
            output,
            target.long(),
            reduction="sum",
            weight=torch.tensor([0.5, 0.5], dtype=torch.float, device=device),
        )
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_idx += 1
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader)
        ))
    return test_loss, acc, precision, recall, auc, f1


def eval_classes(model, device, test_loader, task_name, epoch):
    print('begin to test................')
    model.eval()
    test_loss = 0.0
    correct = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    TP_index, TN_index, FP_index, FN_index = [], [], [], []
    predictions = []
    targets = []
    with torch.no_grad():
        for ids, (data, target) in enumerate(test_loader):
            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            output = model(data)
            test_loss += -F.cross_entropy(output, target.long()).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    max_length = max(len(FN_index), len(TN_index), len(FP_index), len(TP_index))
    FN_index = FN_index + [-1] * (max_length - len(FN_index))
    TN_index = TN_index + [-1] * (max_length - len(TN_index))
    TP_index = TP_index + [-1] * (max_length - len(TP_index))
    FP_index = FP_index + [-1] * (max_length - len(FP_index))
    temp_recoder = pd.DataFrame({'TPi': TP_index, 'FPi': FP_index, 'TNi': TN_index, 'FNi': FN_index})
    temp_recoder_file_root = f'./out/{date}/1B/'
    if not os.path.exists(temp_recoder_file_root):
        os.makedirs(temp_recoder_file_root)
    temp_recoder.to_csv(temp_recoder_file_root + f'{epoch}.csv')

    test_loss /= len(test_loader.dataset)
    precision, recall, _, _ = precision_recall_fscore_support(
        targets, predictions, average="binary"
    )
    f1 = precision * recall * 2 / (precision + recall)
    auc = roc_auc_score(targets, predictions)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Precision: {:.4f}, Recall: {:.4f}, AUC:{:.4f}, F1:{:.4f}\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            precision,
            recall,
            auc,
            f1
        )
    )
    if epoch % 10 == 0:
        torch.save(model.state_dict(),
                   f"{temp_recoder_file_root}/{task_name[task_num]}_down_task_fine_tune_%s.pt" % epoch)
        print('Save success!')
    return test_loss, correct / len(test_loader.dataset), precision, recall, auc, f1


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    print("***************************************************\n"
          f"task_num = {task_num} -----> {task_name[task_num]}\n"
          "***************************************************")
    batch_size, num_class = get_different_task_params(task_num)
    args.batch_size, args.num_class = batch_size, num_class

    # 
    fine_turn_path = 'iter-1000-0008000-ckpt.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = get_dataset(task_num)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    DownStreamNet = get_model(task_num)
    if task_num == 3:
        args.seq_len = 450
    # model = DownStreamNet(model_size='100M', n_classes=args.num_class, fine_turn_path=fine_turn_path,
    model = DownStreamNet(model_size='1B', n_classes=args.num_class, fine_turn_path=fine_turn_path,
                          n_dim=args.dim_output1, seq_length=args.seq_len)
    for name, param in model.named_parameters():
        print(f'Parameter name: {name} has shape {param.shape} and need gradient {param.requires_grad}')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataset))
    torch.set_float32_matmul_precision('high')

    if task_num in [4]:
        loss_, A, P, R, AUC, F1 = [], [], [], [], [], []
        for epoch in range(1, args.epochs + 1):
            test_loss, acc, precision, recall, auc, f1 = train_classes(args, model, device, train_loader, test_loader,
                                                                       optimizer, epoch)

            loss_, A, P, R, AUC, F1 = loss_ + test_loss, A + acc, P + precision, R + recall, AUC + auc, F1 + f1
            scheduler.step()
            if epoch % 20 == 0:
                torch.save(model.state_dict(),
                           f"./out/{date}/{task_num}{task_name[task_num]}_down_task_fine_tune_%s.pt" % epoch)
            data_recoder = pd.DataFrame(
                {'loss': loss_, 'Accuracy:': A, 'Precision:': P, 'Recall:': R, 'AUC': AUC, 'F1': F1})
            if not os.path.exists(f'./out/{date}/'):
                os.makedirs(f'./out/{date}')
            data_recoder.to_csv(f'./out/{date}/{task_name[task_num]}.csv')

    elif task_num in [1, 3]:
        mae_, mse_ = [], []
        for epoch in range(1, args.epochs + 1):
            mae_temp, mse_temp = train_continue(model, device, train_loader, test_loader, optimizer, epoch, scaler)
            mae_, mse_ = mae_ + mae_temp, mse_ + mse_temp
            data_recoder = pd.DataFrame({'mae': mae_, 'mse:': mse_})
            temp_recoder_file_root = f'./out/{date}/'
            if not os.path.exists(temp_recoder_file_root):
                os.makedirs(temp_recoder_file_root)
            data_recoder.to_csv(temp_recoder_file_root + f'{task_name[task_num]}.csv')
    else:
        print(f'Please check your task_num')
