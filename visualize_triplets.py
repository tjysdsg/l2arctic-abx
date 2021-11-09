import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rc("savefig", dpi=300)

MAX_PAIRS_PER_A = 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=r'exp')
    parser.add_argument('--out-dir', type=str, default=r'exp')
    return parser.parse_args()


def main():
    args = get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    PaT_file = open(os.path.join(data_dir, 'PaT.txt'))
    PaC_file = open(os.path.join(data_dir, 'PaC.txt'))
    TaP_file = open(os.path.join(data_dir, 'TaP.txt'))

    for i, line in enumerate(PaT_file):
        a_f, b_f, x_f = line.strip('\n').split()

        a = np.load(os.path.join(data_dir, f'{a_f}.npy'))
        b = np.load(os.path.join(data_dir, f'{b_f}.npy'))
        x = np.load(os.path.join(data_dir, f'{x_f}.npy'))

        fig, axs = plt.subplots(3, 1)
        fig.set_figwidth(20)
        fig.set_figheight(20)
        axs = axs.flat

        axs[0].imshow(a.T)
        axs[0].set_title('A')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('code')

        axs[1].imshow(b.T)
        axs[1].set_title('B')
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('code')

        axs[2].imshow(x.T)
        axs[2].set_title('X')
        axs[2].set_xlabel('time')
        axs[2].set_ylabel('code')

        fig.savefig(os.path.join(out_dir, f'PaT{i}.png'))
        plt.close('all')


if __name__ == '__main__':
    main()
