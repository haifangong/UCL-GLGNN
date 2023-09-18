import argparse
import os
import warnings

from ThermoGNN.utils.features import load_aa_features
from ThermoGNN.utils.graph import make_graph


def main():

    parser = argparse.ArgumentParser(description='Generate graphs for GNN model')
    parser.add_argument('--feature_path', type=str, default='data/features.txt',
                        help='path to file saving sequence encoding features')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to file recording mutations and ddGs')
    parser.add_argument('--out_dir', type=str, default='data/graphs',
                        help='directory to save the output graphs')
    parser.add_argument('--split', type=str, default="train",
                        help='split for different dataset (train, test, p53, myoglobin)')
    parser.add_argument('--contact_threshold', type=float, default=5,
                        help='threshold for contact edge between residues (defalut: 5)')
    parser.add_argument('--local_radius', type=float, default=12,
                        help='maximum distance from the mutation postion (default: 12)')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    if not os.path.exists(os.path.join(args.out_dir, args.split)):
        os.makedirs(os.path.join(args.out_dir, args.split))

    aa_features = load_aa_features(args.feature_path)

    for record in open(args.data_path):
        make_graph(record, aa_features, args.out_dir, is_wt=True, split=args.split,
                   contact_threshold=args.contact_threshold, local_radius=args.local_radius)
        make_graph(record, aa_features, args.out_dir, is_wt=False, split=args.split,
                   contact_threshold=args.contact_threshold, local_radius=args.local_radius)


if __name__ == "__main__":
    main()
