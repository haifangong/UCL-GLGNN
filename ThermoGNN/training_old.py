import logging
import random

import numpy as np
import torch
from ThermoGNN.mcdrop import MCDrop
from torch.nn.functional import mse_loss
from torchmetrics.functional import pearson_corrcoef


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


momentum_list = []
loss_file = open('loss_analysis.csv', 'a+')

def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    total_train_loss = 0
    train_data_size = 0

    var_list = []
    # var = 0
    abs_correct_rate = [0, 0, 0]
    re_correct_rate = [0, 0, 0]
    curri1 = []
    curri = []
    encodings, labels = [], []

    for data in train_loader:
        data = data.to(device)
        if args.fds:
            out, feature = model(data, epoch)
            encodings.extend(feature.data.cpu().numpy())
            labels.extend(data.y.data.cpu().numpy())
        elif args.feature_level == 'local':
            out, out1 = model(data)
        else:
            out = model(data)

        if args.loss == "WeightedMSELoss()":
            loss = criterion(out, data.y, data.wy)
        elif args.loss == "curri":
            mse, logcosh = criterion
            loss = epoch / 100 * mse_loss(out, data.y) + (1 - epoch / 100) * logcosh(out, data.y)
        elif args.loss == "ce":
            new_gt = []
            for idx in range(out.shape[0]):
                if data.y[idx] > 5:
                    new_gt.append(torch.tensor(5))
                elif data.y[idx] > 4:
                    new_gt.append(torch.tensor(4))
                elif data.y[idx] > 3:
                    new_gt.append(torch.tensor(3))
                elif data.y[idx] > 2:
                    new_gt.append(torch.tensor(2))
                elif data.y[idx] > 1:
                    new_gt.append(torch.tensor(1))
                elif data.y[idx] > -1:
                    new_gt.append(torch.tensor(0))
                elif data.y[idx] > -2:
                    new_gt.append(torch.tensor(6))
                elif data.y[idx] > -3:
                    new_gt.append(torch.tensor(7))
                elif data.y[idx] > -4:
                    new_gt.append(torch.tensor(8))
            new_gt = torch.tensor(new_gt).cuda()
            loss = criterion(out, new_gt)
        elif "rmse" in args.loss:

            loss = 0
            loss_bin_label = [[] for i in range(11)]
            loss_bin_pred = [[] for i in range(11)]
            for idx in range(out.shape[0]):
                pred = out[idx].item()
                gt = data.y[idx].item()

                abs_error = abs(pred - gt)
                curri.append(abs_error)
                if data.y[idx].item() == 0:
                    continue
                relative_error = abs_error / (abs(gt))
                curri1.append(relative_error)
                if abs(gt) < 1:
                    abs_correct_rate[0] += abs_error
                    re_correct_rate[0] += relative_error
                elif abs(gt) < 3:
                    abs_correct_rate[1] += abs_error
                    re_correct_rate[1] += relative_error
                else:
                    abs_correct_rate[2] += abs_error
                    re_correct_rate[2] += relative_error
                temp_loss = criterion(out[idx], data.y[idx])
                idx = int(gt)
                if idx > 5:
                    idx = 5
                elif idx < -5:
                    idx = 10
                loss_bin_label[idx].append(temp_loss)
                # pred_list[idx].append(out[idx])

            if args.mcdrop:
                # mc_model = MCDrop(model)
                # var = mc_model.predict(data)
                alpha = 1 + var
            else:
                alpha = 2
            if args.momentum:
                if len(momentum_list) == 2:
                    past_past_mome, past_mome = momentum_list
                # print(len(momentum_list))
                # print(momentum_list)
                cur_list = []
                for idx in range(len(loss_bin_label)):
                    mean_cur = torch.mean(torch.tensor(loss_bin_label[idx]))
                    std_cur = torch.var(torch.tensor(loss_bin_label[idx]))
                    cur_list.append((mean_cur.item(), std_cur.item()))
                    if len(momentum_list) < 2:
                        mean = mean_cur
                        std = std_cur
                    else:
                        mean = mean_cur * args.momentum + past_mome[idx][0] * args.momentum * 0.1 + past_past_mome[idx][
                            0] * args.momentum * 0.01
                        std = std_cur * args.momentum + past_mome[idx][1] * args.momentum * 0.1 + past_past_mome[idx][
                            1] * args.momentum * 0.01

                    for j in range(len(loss_bin_label[idx])):
                        loss_idx = loss_bin_label[idx][j]
                        if loss_idx > mean + alpha * std:
                            loss += epoch / 50 * loss_idx
                        else:
                            loss += loss_idx

                if len(momentum_list) == 2:
                    momentum_list.pop(0)
                    momentum_list.append(cur_list)
                else:
                    momentum_list.append(cur_list)

            else:
                for idx in range(len(loss_bin_label)):
                    mean = torch.mean(torch.tensor(loss_bin_label[idx]))
                    # mean = torch.mean(torch.tensor(pred_list[idx]))
                    std = torch.std(torch.tensor(loss_bin_label[idx]))
                    # std = torch.std(torch.tensor(pred_list[idx]))

                    for j in range(len(loss_bin_label[idx])):
                        loss_idx = loss_bin_label[idx][j]
                        if loss_idx > mean + 2 * std:
                            loss += epoch / 100 * loss_idx
                        else:
                            loss += loss_idx
            del loss_bin_label
        else:
            if args.feature_level == 'local':
                loss = criterion(out, data.y) + criterion(out, -out1)
            else:
                loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.mcdrop:
            mc_model = MCDrop(model, args.mcdrop)
            var = mc_model.predict(data)
            var_list.append(var)
            # print(var)

        total_train_loss += loss * out.size(0)
        train_data_size += out.size(0)
        if True:
            loss_file.write(str(epoch) + ',' + str(np.mean(curri)) + ',' + str(np.mean(curri1)) + ',' + str(np.std(curri1))
                  + ',' + str(abs_correct_rate[0] / 1431) + ',' + str(abs_correct_rate[1] / 901) + ',' + str(
                abs_correct_rate[2] / 308)
                  + ',' + str(re_correct_rate[0] / 1431) + ',' + str(re_correct_rate[1] / 901) + ',' + str(
                re_correct_rate[2] / 308)+'\n'
                  )
    if args.fds:
        encodings, labels = torch.from_numpy(np.vstack(encodings)), torch.from_numpy(
            np.hstack(labels))
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(encodings, labels, epoch)
        del encodings, labels
    train_loss = total_train_loss / train_data_size
    # print(var_list)
    del var_list
    model.eval()
    total_valid_loss = 0
    valid_data_size = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            if args.fds:
                out, _ = model(data)
            elif args.feature_level == 'local':
                out, out1 = model(data)
                out = (out - out1)/2
            else:
                out = model(data)
            loss = mse_loss(out, data.y)
            total_valid_loss += loss * out.size(0)
            valid_data_size += out.size(0)

    valid_loss = total_valid_loss / valid_data_size

    return train_loss, valid_loss


def evaluate(args, model, loader, device, return_tensor=False):
    model.eval()
    pred = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if args.fds:
                out, _ = model(data)
            elif args.feature_level == 'local':
                out, out1 = model(data)
                out = (out - out1) / 2
            else:
                out = model(data)
            pred.append(out)
            y.append(data.y)

        pred_tensor = torch.cat(pred)
        y_tensor = torch.cat(y)
        corr = pearson_corrcoef(pred_tensor, y_tensor)
        rmse = torch.sqrt(mse_loss(pred_tensor, y_tensor))

    if return_tensor:
        return pred_tensor, y_tensor
    else:
        return corr, rmse


def metrics(pred_dir, pred_rev, y_dir, y_rev):
    corr_dir = pearson_corrcoef(pred_dir, y_dir)
    rmse_dir = torch.sqrt(mse_loss(pred_dir, y_dir))
    corr_rev = pearson_corrcoef(pred_rev, y_rev)
    rmse_rev = torch.sqrt(mse_loss(pred_rev, y_rev))
    corr_dir_rev = pearson_corrcoef(pred_dir, pred_rev)
    delta = torch.mean(pred_dir + pred_rev)

    return corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta


class EarlyStopping:

    def __init__(self, patience=10, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), self.path)
        self.best_score = score
