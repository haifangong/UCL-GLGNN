# python main.py --loss super --gnn-type gat --seed 1
# python main.py --loss super --gnn-type gat --seed 2
# python main.py --loss super --gnn-type gat --seed 3
# python main.py --loss super --gnn-type gat --seed 4
# python main.py --loss super --gnn-type gat --seed 5
CUDA_VISIBLE_DEVICES=4 python main.py --loss wmse --gnn-type gat --seed 1 --fds True
CUDA_VISIBLE_DEVICES=4 python main.py --loss wmse --gnn-type gat --seed 2 --fds True
CUDA_VISIBLE_DEVICES=4 python main.py --loss wmse --gnn-type gat --seed 3 --fds True
CUDA_VISIBLE_DEVICES=4 python main.py --loss wmse --gnn-type gat --seed 4 --fds True
CUDA_VISIBLE_DEVICES=4 python main.py --loss wmse --gnn-type gat --seed 5 --fds True