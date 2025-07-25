from argparse import ArgumentParser
import pickle
from datasets import capacity_Dataset,riding_Dataset, except_Dataset


def get_args():
    parser = ArgumentParser(description='Battery Transformer')
    parser.add_argument('--task', type=int, default=0,
                        help='Task in [capacity, except, riding], mean [1, 4, 3]')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the number of total epochs to run.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size. (default: 64), in different task is [16, 256, 128, 512]')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. (default: 1e-5)')
    parser.add_argument('--wd', type=float, default=1e-6,
                        help='weight decay. (default: 1e-5)')
    parser.add_argument('--seq_len', type=int, default=500,
                        help='Input sequence length. (default: 500)')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Internal dimension of transformer embeddings.')
    parser.add_argument('--num_class', type=int, default=1,
                        help='Number of class.')
    parser.add_argument('--dim_output1', type=int, default=128,
                        help='Dimension of fine tune of liner1.')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status.')
    parser.add_argument('--gpu', type=int, default=8,
                        help='gpu id to use. (default: 0)')
    parser.add_argument('--seed', type=int, default=609,
                        help='random seed. (default: 1)')
    parser.add_argument('--gpu_id', type=str, default='0, 1, 2, 3, 4， 5, 6, 7')
    parser.add_argument('--nproc_per_node', type=int, default=8)
    args = parser.parse_args()
    return args


def get_different_task_params(num):
    super_params = {
        1: [100, 1],
        3: [100, 1],
        4: [40, 2]
    }
    batch_size = super_params[num][0]
    num_class = super_params[num][1]
    return batch_size, num_class


def get_dataset(num):
    if  num == 1:
        return capacity_Dataset(train=True), capacity_Dataset(train=False)
    elif num == 3:
        return riding_Dataset(train=True), riding_Dataset(train=False)
    elif num == 4:
        return except_Dataset(train=True), except_Dataset(train=False)


def get_model(num):
    if num == 4:
        from except.model_lora import DownStreamNet
        return DownStreamNet
    elif num == 1:
        from capacity.model_lora import DownStreamNet
        return DownStreamNet
    elif num == 3:
        from riding.model_lora import DownStreamNet
        return DownStreamNet


