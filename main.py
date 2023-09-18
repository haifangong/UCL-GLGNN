import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
# import wandb
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from ThermoGNN.dataset import load_dataset
from ThermoGNN.model import GraphGNN
from ThermoGNN.training import (EarlyStopping, evaluate, metrics, set_seed, train)
from ThermoGNN.loss import LogCoshLoss, WeightedMSELoss, SuperLoss


def run_case_study(args, model, task, graph_dir, weight_dir, fold=5, visualize=False):
    logging.info(f"Task: {task}")

    test_data_list = load_dataset(graph_dir, task)
    test_direct_dataset, test_reverse_dataset = test_data_list[::2], test_data_list[1::2]
    test_direct_loader = DataLoader(
        test_direct_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)
    test_reverse_loader = DataLoader(
        test_reverse_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)

    total_pred_dir = []
    total_pred_rev = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i in range(fold):
        model.load_state_dict(torch.load(f"{weight_dir}/model_{i + 1}.pkl"))
        pred_dir, y_dir = evaluate(args, model, test_direct_loader, device, return_tensor=True)
        pred_rev, y_rev = evaluate(args, model, test_reverse_loader, device, return_tensor=True)

        corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta = metrics(
            pred_dir, pred_rev, y_dir, y_rev)

        logging.info(f'Fold {i + 1}, Direct PCC: {corr_dir:.3f}, Reverse PCC: {corr_rev:.3f}, Direct RMSE: {rmse_dir:.3f},  Reverse RMSE: {rmse_rev:.3f}')

        total_pred_dir.append(pred_dir.tolist())
        total_pred_rev.append(pred_rev.tolist())

    avg_pred_dir = torch.Tensor(total_pred_dir).mean(dim=0).to(device)
    avg_pred_rev = torch.Tensor(total_pred_rev).mean(dim=0).to(device)
    avg_corr_dir, avg_rmse_dir, avg_corr_rev, avg_rmse_rev, avg_corr_dir_rev, avg_delta = metrics(
        avg_pred_dir, avg_pred_rev, y_dir, y_rev)

    logging.info(f'{avg_corr_dir:.3f} {avg_corr_rev:.3f} {avg_rmse_dir:.3f} {avg_rmse_rev:.3f}')

    if visualize:
        wandb.init(project="ThermoGNN", group=os.path.dirname(weight_dir),
                   name=f"{os.path.dirname(weight_dir)}-{task}")

        wandb.run.summary['Avg Direct PCC'] = avg_corr_dir
        wandb.run.summary['Avg Direct RMSE'] = avg_rmse_dir
        wandb.run.summary['Avg Reverse PCC'] = avg_corr_rev
        wandb.run.summary['Avg Reverse RMSE'] = avg_rmse_rev
        wandb.run.summary['Avg Dir-Rev PCC'] = avg_corr_dir_rev
        wandb.run.summary['Avg <Delta>'] = avg_delta

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300, figsize=(15, 5))

        ax1.scatter(y_dir.cpu().numpy(), pred_dir.cpu().numpy(), c=(
                y_dir - pred_dir).cpu().numpy(), cmap="bwr", alpha=0.5, edgecolors="grey", linewidth=0.1,
                    norm=colors.CenteredNorm())
        ax1.plot((-4.5, 6.5), (-4.5, 6.5), ls='--', c='k')
        ax1.set_xlabel(r'Experimental $\Delta \Delta G$ (kcal/mol)')
        ax1.set_ylabel(r'Predicted $\Delta \Delta G$ (kcal/mol)')
        ax1.set_xlim(-4.5, 6.5)
        ax1.set_ylim(-4.5, 6.5)
        ax1.text(0.25, 0.85, 'Direct mutations', horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        ax1.text(0.75, 0.2, r'$r = {:.2f}$'.format(avg_corr_dir), horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        ax1.text(0.75, 0.12, r'$\sigma = {:.2f}$'.format(avg_rmse_dir), horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        ax1.grid(ls='--', alpha=0.5, linewidth=0.5)

        ax2.scatter(y_rev.cpu().numpy(), pred_rev.cpu().numpy(), c=(
                y_rev - pred_rev).cpu().numpy(), cmap="bwr", alpha=0.5, edgecolors="grey", linewidth=0.1,
                    norm=colors.CenteredNorm())
        ax2.plot((-6.5, 4.5), (-6.5, 4.5), ls='--', c='k')
        ax2.set_xlabel(r'Experimental $\Delta \Delta G$ (kcal/mol)')
        ax2.set_ylabel(r'Predicted $\Delta \Delta G$ (kcal/mol)')
        ax2.set_xlim(-6.5, 4.5)
        ax2.set_ylim(-6.5, 4.5)
        ax2.text(0.25, 0.85, 'Reverse mutations', horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.text(0.75, 0.2, r'$r = {:.2f}$'.format(avg_corr_rev), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.text(0.75, 0.12, r'$\sigma = {:.2f}$'.format(avg_rmse_rev), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.grid(ls='--', alpha=0.5, linewidth=0.5)

        ax3.scatter(pred_dir.cpu().numpy(), pred_rev.cpu().numpy(),
                    c='#3944BC', alpha=0.2, edgecolors="grey", linewidth=0.1)
        ax3.plot((-5, 5), (5, -5), ls='--', c='k')
        ax3.set_xlabel('Prediction for direct mutation')
        ax3.set_ylabel('Prediction for reverse mutation')
        ax3.set_xlim(-5, 5)
        ax3.set_ylim(-5, 5)
        ax3.text(0.3, 0.2, r'$r = {:.2f}$'.format(avg_corr_dir_rev), horizontalalignment='center',
                 verticalalignment='center', transform=ax3.transAxes)
        ax3.text(0.3, 0.12, r'$\delta = {:.2f}$'.format(avg_delta), horizontalalignment='center',
                 verticalalignment='center', transform=ax3.transAxes)
        ax3.grid(ls='--', alpha=0.2, linewidth=0.5)

        plt.tight_layout()

        img = wandb.Image(fig)
        wandb.log({"chart": img})

        wandb.join()


def main():
    parser = argparse.ArgumentParser(description='ThermoGNN: predict thermodynamics stability')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=10,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=25,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--loss', type=str, default='curri',
                        help='loss function (mse, logcosh, wmse)')
    parser.add_argument('--num-layer', type=int, dest='num_layer', default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=200,
                        help='embedding dimensions (default: 200)')
    parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="mean",
                        help='graph level pooling (sum, mean, max, attention)')
    parser.add_argument('--graph-dir', type=str, dest='graph_dir', default='data/graphs',
                        help='directory storing graphs data')
    parser.add_argument('--logging-dir', type=str, dest='logging_dir', default='./',
                        help='logging directory (default: \'./\')')
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gat",
                        help='gnn type (gin, gcn, gat, graphsage)')
    parser.add_argument('--concat-type', type=str, dest='concat_type', default="concat",
                        help='concat type (lstm, bilstm, gru, concat)')
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--visualize', action='store_true', default=False,
                        help="Visualize training by wandb")

    parser.add_argument('--feature-level', type=str, dest='feature_level', default='global-local',
                        help='global-local, global, or local')

    # curricula setting
    parser.add_argument('--contrast-curri', dest='contrast_curri', action='store_true', default=False,
                        help='using node contrast curriculum learning or not')
    parser.add_argument('--bias-curri', dest='bias_curri', action='store_true', default=False,
                        help='directly use loss as the training data (biased) or not (unbiased)')
    parser.add_argument('--anti-curri', dest='anti_curri', action='store_true', default=False,
                        help='easy to hard (curri), hard to easy (anti)')
    parser.add_argument('--std-coff', dest='std_coff', type=float, default=1,
                        help='the hyper-parameter of std')

    parser.add_argument('--bins', type=int, dest='bins', default=6,
                        help='the number of the bins')
    parser.add_argument('--momentum', type=float, dest='momentum', default=0,
                        help='0.9 is good')
    parser.add_argument('--mcdrop', type=int, dest='mcdrop', default=0,
                        help='how many times performed mc drop')
    parser.add_argument('--fds', type=bool, dest='fds', default=False,
                        help='dir')
    parser.add_argument('--scheduler', type=str, dest='scheduler', default="linear",
                        help='linear')

    # noisy setting
    parser.add_argument('--noisy-rate', type=float, dest='noisy_rate', default=0,
                        help='the noisy rate of training data')

    args = parser.parse_args()
    set_seed(args.seed)

    feature = '=' + args.feature_level
    curricula = ''
    if args.loss == "logcosh":
        criterion = LogCoshLoss()
    elif args.loss == "wmse":
        criterion = WeightedMSELoss()
    elif args.loss == "mse" or "rmse":
        criterion = nn.MSELoss()
    elif args.loss == "curri":
        criterion = nn.MSELoss()
        curricula = '=std' + str(args.std_coff) + '-'
        curricula += 'simi-' if args.contrast_curri else 'loss-'
        curricula += 'bias-' if args.bias_curri else 'unbias-'
        curricula += 'anti-' if args.anti_curri else 'unanti-'
    elif args.loss == "super":
        criterion = SuperLoss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    weight_dir = os.path.join(args.logging_dir, "runs", args.gnn_type + "-" + args.loss + feature + curricula + str(args.seed))

    print('saving_dir: ', weight_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))
    torch.autograd.set_detect_anomaly(True)
    logging.info('Loading Training Dataset')
    data_list = load_dataset(args.graph_dir, "train")
    direct_dataset, reverse_dataset = data_list[::2], data_list[1::2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    five_fold_index = []
    for i in range(5):
        train_index, valid_index = [], []
        for j in range(len(direct_dataset)):
            if (j+i) % 5 != 0:
                train_index.append(j)
            else:
                valid_index.append(j)
        five_fold_index.append((train_index, valid_index))

    logging.info('Loading Test Dataset')
    test_data_list = load_dataset(args.graph_dir, "test")
    test_direct_dataset, test_reverse_dataset = test_data_list[::2], test_data_list[1::2]
    test_direct_loader = DataLoader(test_direct_dataset, batch_size=args.batch_size,
                                    follow_batch=['x_s', 'x_t'], shuffle=False)
    test_reverse_loader = DataLoader(test_reverse_dataset, batch_size=args.batch_size,
                                     follow_batch=['x_s', 'x_t'], shuffle=False)

    total_pred_dir = []
    total_pred_rev = []

    for i, (train_index, valid_index) in enumerate(five_fold_index):
        print(len(train_index))
        print(len(valid_index))
        model = GraphGNN(num_layer=args.num_layer, input_dim=60, emb_dim=args.emb_dim, out_dim=1, JK="last",
                         drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type,
                         concat_type=args.concat_type, fds=args.fds, feature_level=args.feature_level, contrast_curri=args.contrast_curri)
        model.to(device)

        train_direct_dataset, valid_direct_dataset = [direct_dataset[i] for i in train_index], \
                                                     [direct_dataset[j] for j in valid_index]
        train_reverse_dataset, valid_reverse_dataset = [reverse_dataset[i] for i in train_index], \
                                                       [reverse_dataset[j] for j in valid_index]

        print(len(train_direct_dataset)+len(valid_direct_dataset))
        train_loader = DataLoader(train_direct_dataset + train_reverse_dataset, batch_size=args.batch_size,
                                  follow_batch=['x_s', 'x_t'], shuffle=True)
        valid_loader = DataLoader(valid_direct_dataset + valid_reverse_dataset, batch_size=args.batch_size,
                                  follow_batch=['x_s', 'x_t'], shuffle=False)

        train_direct_loader = DataLoader(train_direct_dataset, batch_size=args.batch_size,
                                         follow_batch=['x_s', 'x_t'], shuffle=False)
        train_reverse_loader = DataLoader(train_reverse_dataset, batch_size=args.batch_size,
                                          follow_batch=['x_s', 'x_t'], shuffle=False)
        valid_direct_loader = DataLoader(valid_direct_dataset, batch_size=args.batch_size,
                                         follow_batch=['x_s', 'x_t'], shuffle=False)
        valid_reverse_loader = DataLoader(valid_reverse_dataset, batch_size=args.batch_size,
                                          follow_batch=['x_s', 'x_t'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        weights_path = f"{weight_dir}/model_{i + 1}.pkl"
        early_stopping = EarlyStopping(patience=args.patience, path=weights_path)
        logging.info(f'Running Cross Validation {i + 1}')

        for epoch in range(1, args.epochs + 1):
            train_loss, valid_loss = train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)

            train_dir_pcc, train_dir_rmse = evaluate(args, model, train_direct_loader, device)
            train_rev_pcc, train_rev_rmse = evaluate(args, model, train_reverse_loader, device)
            valid_dir_pcc, valid_dir_rmse = evaluate(args, model, valid_direct_loader, device)
            valid_rev_pcc, valid_rev_rmse = evaluate(args, model, valid_reverse_loader, device)
            test_dir_pcc, test_dir_rmse = evaluate(args, model, test_direct_loader, device)
            test_rev_pcc, test_rev_rmse = evaluate(args, model, test_reverse_loader, device)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}')
            print(f'Train Direct PCC: {train_dir_pcc:.3f}, Train Direct RMSE: {train_dir_rmse:.3f},'
                  f'Train Reverse PCC: {train_rev_pcc:.3f}, Train Reverse RMSE: {train_rev_rmse:.3f}')
            print(f'Valid Direct PCC: {valid_dir_pcc:.3f}, Valid Direct RMSE: {valid_dir_rmse:.3f},'
                  f'Valid Reverse PCC: {valid_rev_pcc:.3f}, Valid Reverse RMSE: {valid_rev_rmse:.3f}')
            print(f'Test Direct PCC: {test_dir_pcc:.3f}, Test Direct RMSE: {test_dir_rmse:.3f},'
                  f' Test Reverse PCC: {test_rev_pcc:.3f}, Test Reverse RMSE: {test_rev_rmse:.3f}')
            if args.visualize:
                wandb.init(project="ThermoGNN", group=args.logging_dir, name=f'{weight_dir}_fold_{i + 1}', config=args)
                wandb.log({'Train/Loss': train_loss, 'Valid/Loss': valid_loss}, step=epoch)
                wandb.log({'Train/Direct PCC': train_dir_pcc, 'Train/Direct RMSE': train_dir_rmse,
                           'Train/Reverse PCC': train_rev_pcc, 'Train/Reverse RMSE': train_rev_rmse}, step=epoch)
                wandb.log({'Valid/Direct PCC': valid_dir_pcc, 'Valid/Direct RMSE': valid_dir_rmse,
                           'Valid/Reverse PCC': valid_rev_pcc, 'Valid/Reverse RMSE': valid_rev_rmse}, step=epoch)

            # scheduler.step()
            # lr = scheduler.get_last_lr()
            # print('lr', lr)

            early_stopping(valid_loss, model, goal="minimize")

            if early_stopping.early_stop:
                logging.info(f"Early stopping at Epoch {epoch + 1}")
                break

        model.load_state_dict(torch.load(weights_path))
        pred_dir, y_dir = evaluate(args, model, test_direct_loader, device, return_tensor=True)
        pred_rev, y_rev = evaluate(args, model, test_reverse_loader, device, return_tensor=True)

        corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta = metrics(pred_dir, pred_rev, y_dir, y_rev)

        logging.info(f'Fold {i + 1}, Best Valid Loss: {-early_stopping.best_score:.3f}')
        logging.info(f'{corr_dir:.3f} {rmse_dir:.3f} {corr_rev:.3f} {rmse_rev:.3f} {corr_dir_rev:.3f} {delta:.3f}')

        if args.visualize:
            wandb.run.summary['Valid/Best Valid Loss'] = -early_stopping.best_score
            wandb.run.summary['Test/Direct PCC'] = corr_dir
            wandb.run.summary['Test/Direct RMSE'] = rmse_dir
            wandb.run.summary['Test/Reverse PCC'] = corr_rev
            wandb.run.summary['Test/Reverse RMSE'] = rmse_rev
            wandb.run.summary['Test/Dir-Rev PCC'] = corr_dir_rev
            wandb.run.summary['Test/<Delta>'] = delta

            wandb.join()

        total_pred_dir.append(pred_dir.tolist())
        total_pred_rev.append(pred_rev.tolist())

    avg_pred_dir = torch.Tensor(total_pred_dir).mean(dim=0).to(device)
    avg_pred_rev = torch.Tensor(total_pred_rev).mean(dim=0).to(device)
    avg_corr_dir, avg_rmse_dir, avg_corr_rev, avg_rmse_rev, avg_corr_dir_rev, avg_delta = metrics(avg_pred_dir,
                                                                                                  avg_pred_rev, y_dir,
                                                                                                  y_rev)

    logging.info(f'Cross Validation Finished!')
    logging.info(
        f'{avg_corr_dir:.3f} {avg_rmse_dir:.3f} {avg_corr_rev:.3f} {avg_rmse_rev:.3f} {avg_corr_dir_rev:.3f} {avg_delta:.3f}')

    # case studies
    model = GraphGNN(num_layer=args.num_layer, input_dim=60, emb_dim=args.emb_dim, out_dim=1, JK="last",
                     drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type,
                     concat_type=args.concat_type, fds=args.fds, feature_level=args.feature_level, contrast_curri=args.contrast_curri)
    run_case_study(args, model, "test", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)
    run_case_study(args, model, "myoglobin", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)
    run_case_study(args, model, "p53", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)


if __name__ == "__main__":
    main()
