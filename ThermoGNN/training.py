import random
import numpy as np
import torch

from ThermoGNN.loss import unbiased_curriculum_loss
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


mome_bank = []
momentum_list2 = []


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    total_train_loss = 0
    train_data_size = 0

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
        elif args.contrast_curri:
            out, similarity = model(data, epoch)
        else:
            out = model(data)

        if args.loss == "WeightedMSELoss()":
            loss = criterion(out, data.y, data.wy)
        elif 'curri' in args.loss:
            if args.contrast_curri:
                loss_list = []
                diff_loss_list = []
                diff_simi_list = []
                for idx in range(out.shape[0]):
                    gt = abs(data.y[idx].item())
                    gt = 1 if gt < 1 else gt
                    temp_loss = criterion(out[idx], data.y[idx])
                    loss_list.append(temp_loss)
                    diff_loss_list.append(round(temp_loss.item() / gt, 3))
                    diff_simi_list.append(round(similarity[idx].item(), 3))
                mean_simi, std_simi = np.mean(diff_simi_list), np.std(diff_simi_list)
                mean_loss, std_loss = np.mean(diff_loss_list), np.std(diff_loss_list)
                loss = 0
                for loss_idx in range(len(loss_list)):
                    loss_value = loss_list[loss_idx]
                    if diff_simi_list[loss_idx] > mean_simi + args.std_coff * std_simi:
                        loss += linear(epoch, args.epochs) * loss_value
                    elif diff_loss_list[loss_idx] > mean_loss + args.std_coff * std_loss:
                        loss += linear(epoch, args.epochs) * loss_value
                    else:
                        loss += loss_value
            else:
                loss = unbiased_curriculum_loss(out, data, args, criterion, scheduler='linear')
        else:
            loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss * out.size(0)
        train_data_size += out.size(0)
        # if True:
        #     print(str(epoch) + ' ' + str(np.mean(curri)) + ' ' + str(np.mean(curri1)) + ' ' + str(np.std(curri1))
        #           + ' ' + str(abs_correct_rate[0] / 1431) + ' ' + str(abs_correct_rate[1] / 901) + ' ' + str(
        #         abs_correct_rate[2] / 308)
        #           + ' ' + str(re_correct_rate[0] / 1431) + ' ' + str(re_correct_rate[1] / 901) + ' ' + str(
        #         re_correct_rate[2] / 308)
        #           )
    if args.fds:
        encodings, labels = torch.from_numpy(np.vstack(encodings)), torch.from_numpy(
            np.hstack(labels))
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(encodings, labels, epoch)
        del encodings, labels
    train_loss = total_train_loss / train_data_size

    model.eval()
    total_valid_loss = 0
    valid_data_size = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            if args.fds or args.contrast_curri:
                out, _ = model(data)
            else:
                out = model(data)
            loss = mse_loss(out, data.y)
            total_valid_loss += loss * out.size(0)
            valid_data_size += out.size(0)

    valid_loss = total_valid_loss / valid_data_size

    return train_loss, valid_loss


def evaluate(args, model, loader, device, return_tensor=False):
    model.eval()
    auc_pred, auc_label = [], []
    pred = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if args.fds or args.contrast_curri:
                out, _ = model(data)
            else:
                out = model(data)
            pred.append(out)
            y.append(data.y)
            auc_pred.extend(out.cpu().numpy().reshape(-1).tolist())
            auc_label.extend(data.y.cpu().numpy().reshape(-1).tolist())


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
