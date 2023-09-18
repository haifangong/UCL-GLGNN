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
from torch_geometric.data import DataLoader

from ThermoGNN.dataset import load_dataset
from ThermoGNN.model import GraphGNN, LogCoshLoss, WeightedMSELoss
from ThermoGNN.training_old import (EarlyStopping, evaluate, metrics, set_seed, train)


def run_case_study(args, model, task, graph_dir, weight_dir, fold=5, visualize=False):
    logging.info(f"Task: {task}")

    test_data_list = load_dataset(graph_dir, task)
    test_direct_dataset, test_reverse_dataset = test_data_list[::2], test_data_list[1::2]
    test_direct_loader = DataLoader(
        test_direct_dataset, batch_size=128, follow_batch=['x_s', 'x_t'], shuffle=False)
    test_reverse_loader = DataLoader(
        test_reverse_dataset, batch_size=128, follow_batch=['x_s', 'x_t'], shuffle=False)

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

        logging.info(f'Fold {i + 1}, Direct PCC: {corr_dir:.3f}, Direct RMSE: {rmse_dir:.3f},'
                     f' Reverse PCC: {corr_rev:.3f}, Reverse RMSE: {rmse_rev:.3f},'
                     f' Dir-Rev PCC {corr_dir_rev:.3f}, <Delta>: {delta:.3f}')

        total_pred_dir.append(pred_dir.tolist())
        total_pred_rev.append(pred_rev.tolist())

    avg_pred_dir = torch.Tensor(total_pred_dir).mean(dim=0).to(device)
    avg_pred_rev = torch.Tensor(total_pred_rev).mean(dim=0).to(device)
    avg_corr_dir, avg_rmse_dir, avg_corr_rev, avg_rmse_rev, avg_corr_dir_rev, avg_delta = metrics(
        avg_pred_dir, avg_pred_rev, y_dir, y_rev)

    logging.info(f'Avg Direct PCC: {avg_corr_dir:.3f}, Avg Direct RMSE: {avg_rmse_dir:.3f},'
                 f' Avg Reverse PCC: {avg_corr_rev:.3f}, Avg Reverse RMSE: {avg_rmse_rev:.3f},'
                 f' Avg Dir-Rev PCC {avg_corr_dir_rev:.3f}, Avg <Delta>: {avg_delta:.3f}')

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
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=10,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--loss', type=str, default='mse',
                        help='loss function (mse, logcosh, wmse)')
    parser.add_argument('--num-layer', type=int, dest='num_layer', default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=200,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                        help='dropout ratio (default: 0.3)')
    parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="mean",
                        help='graph level pooling (sum, mean, max, attention)')
    parser.add_argument('--graph-dir', type=str, dest='graph_dir', default='data/graphs',
                        help='directory storing graphs data')
    parser.add_argument('--logging-dir', type=str, dest='logging_dir', default='./',
                        help='logging directory (default: \'./\')')
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gin",
                        help='gnn type (gin, gcn, gat, graphsage)')
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--visualize', action='store_true', default=True,
                        help="Visualize training by wandb")
    args = parser.parse_args()

    set_seed(args.seed)

    weight_dir = 'runs/gat-lstm-rmse-mome0.9-mc0-4'
    # weight_dir = os.path.join(args.logging_dir, "runs-9.2", args.gnn_type + "-" + args.loss + "-" + str(args.seed))
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    # case studies
    model = GraphGNN(num_layer=args.num_layer, input_dim=60, emb_dim=args.emb_dim, out_dim=1, JK="last",
                     drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    run_case_study(args, model, "test", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)
    # run_case_study(args, model, "p53", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)
    run_case_study(args, model, "myoglobin", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)


if __name__ == "__main__":
    main()
