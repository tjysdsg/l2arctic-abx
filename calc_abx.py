import os
import argparse
import numpy as np
from dtw import dtw

MAX_PAIRS_PER_A = 20
EXPECTED_SIGN = {
    'PaT': -1,  # A
    'PaC': 1,  # B
    'TaP': -1,  # A
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
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


def spk_from_utt(utt: str):
    return utt.split('-')[0]


def main():
    args = get_args()

    spk2error = {}
    for triplet_type in ['PaT', 'PaC', 'TaP']:
        with open(os.path.join(args.data_dir, f'{triplet_type}.txt')) as f:
            for line in f:
                a_f, b_f, x_f = line.strip('\n').split()

                a = np.load(os.path.join(args.data_dir, f'{a_f}.npy'))
                b = np.load(os.path.join(args.data_dir, f'{b_f}.npy'))
                x = np.load(os.path.join(args.data_dir, f'{x_f}.npy'))

                # get speaker pair
                spk = frozenset([
                    spk_from_utt(a_f),
                    spk_from_utt(b_f),
                    spk_from_utt(x_f),
                ])

                # calculate dtw distances
                dist_xa = calculate_distance(x, a)
                dist_xb = calculate_distance(x, b)
                spk2error.setdefault(spk, []).append(
                    (-1 if dist_xa - dist_xb < 0 else 1) != EXPECTED_SIGN[triplet_type]
                )

    mean_err_spk = [np.mean(err) for _, err in spk2error.items()]
    abx_score = np.mean(mean_err_spk)
    print(abx_score * 100)


if __name__ == '__main__':
    main()
