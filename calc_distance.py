import os
import argparse
from typing import List
import numpy as np
from dtw import dtw

MAX_PAIRS_PER_A = 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=r'exp')
    return parser.parse_args()


def euclidean(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def calculate_distance(x: np.ndarray, y: np.ndarray):
    dist, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean)
    # print(dist)
    # import matplotlib.pyplot as plt

    # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.show()
    return dist


def main():
    args = get_args()

    PaT_file = open(os.path.join(args.data_dir, 'PaT.txt'))
    PaC_file = open(os.path.join(args.data_dir, 'PaC.txt'))
    TaP_file = open(os.path.join(args.data_dir, 'TaP.txt'))

    for line in PaT_file:
        a_f, b_f, x_f = line.strip('\n').split()

        a = np.load(os.path.join(args.data_dir, f'{a_f}.npy'))
        b = np.load(os.path.join(args.data_dir, f'{b_f}.npy'))
        x = np.load(os.path.join(args.data_dir, f'{x_f}.npy'))

        dist_xa = calculate_distance(x, a)
        dist_xb = calculate_distance(x, b)
        print(dist_xa, dist_xb)


if __name__ == '__main__':
    main()
