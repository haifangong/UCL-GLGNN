import os
import argparse
import json

import torch
import numpy as np
from torch_geometric.data import DataLoader

from ThermoGNN.dataset import load_dataset
from ThermoGNN.model import GraphGNN


def predict_ddG(model, dataset, weight_dir, device, reverse=False):
    dataset = dataset[1::2] if reverse else dataset[::2]
    dataloader = DataLoader(dataset, batch_size=len(dataset), follow_batch=['x_s', 'x_t'], shuffle=False)

    model.to(device)
    total_pred = []

    for i in range(len(os.listdir(weight_dir))):
        model.load_state_dict(torch.load(f"{weight_dir}/model_{i + 1}.pkl"))
        pred = []

        model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                out = model(data)
                pred.append(out)

            pred_tensor = torch.cat(pred)

        total_pred.append(pred_tensor.tolist())
    avg_pred = np.mean(total_pred, axis=0)
    return avg_pred


def main():
    parser = argparse.ArgumentParser(description="Use ThermoGNN to predict ddG from ")
    parser.add_argument('-l', '--mutant-list', type=str, dest='mutant_list', required=True,
                        help='The file storing the names of the structures.')
    parser.add_argument('--model', type=str, required=True,
                        help='The directory of ThermoGNN model')
    parser.add_argument('--split', type=str, required=True,
                        help='pdb and hhm files are stored in data/pdbs/$spilt/ and data/hhm/$split/')
    parser.add_argument('-o', '--out-file', type=str, dest="out_file", default="prediction.csv",
                        help='The file to store the predictions.')
    parser.add_argument('--reverse', action="store_true",
                        help='predict ddGs for reverse mutations')

    args = parser.parse_args()

    gen_graph_cmd = ' '.join(['python gen_graph.py', '--feature_path data/features.txt', '--out_dir data/graphs',
                              '--data_path', args.mutant_list, '--split', args.split,
                              '--contact_threshold 5 --local_radius 12'])
    os.system(gen_graph_cmd)

    records = [line.strip() for line in open(args.mutant_list, "r")]

    with open(f"data/{args.split}_names.txt", "w") as f:
        for record in records:
            pdb_name, pos, wt, mut = record.split()
            f.write(f"{pdb_name}_{wt}{pos}{mut}\n")

    dataset = load_dataset(graph_dir="data/graphs", split=args.split, labeled=False)

    with open(os.path.join(args.model, "config.json"), "r") as f:
        config = json.load(f)
    model = GraphGNN(num_layer=config['num_layer'], input_dim=60, emb_dim=config['emb_dim'], out_dim=1, JK="last",
                     drop_ratio=config['dropout_ratio'], graph_pooling=config['graph_pooling'],
                     gnn_type=config['gnn_type'], concat_type=config['concat_type'])
    weight_dir = os.path.join(args.model, "weights")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prediction = predict_ddG(model, dataset, weight_dir, device, args.reverse)

    with open(args.out_file, "w") as f:
        f.write('PDB,POS,WT,MUT,DDG\n')
        for record, pred in zip(records, prediction):
            pdb_name, pos, wt, mut = record.split()
            if args.reverse:
                f.write(f'{pdb_name},{pos},{mut},{wt},{pred:.2f}\n')
            else:
                f.write(f'{pdb_name},{pos},{wt},{mut},{pred:.2f}\n')


if __name__ == "__main__":
    main()
