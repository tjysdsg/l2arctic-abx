import dataclasses
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rc("savefig", dpi=300)

MAX_PAIRS_PER_A = 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--phone-alignment', type=str)
    parser.add_argument('--freq', type=int)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


@dataclasses.dataclass
class Phone:
    utt: str
    phone: str
    start_frame: int
    end_frame: int


def main():
    args = get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    utt2phones = {}
    utt2code = {}
    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()

            # skip non-existing utterances
            if not os.path.exists(os.path.join(data_dir, f'{utt}.npy')):
                continue

            start, end = float(start), float(end)
            start_frame, end_frame = int(np.round(start * args.freq)), int(np.round(end * args.freq))
            code = np.load(os.path.join(data_dir, f'{utt}.npy'))

            p = Phone(utt, phone, start_frame, end_frame)
            utt2phones.setdefault(utt, []).append(p)
            utt2code[utt] = code

    for utt, phones in utt2phones.items():
        code = utt2code[utt].T
        ymax = code.shape[0]

        plt.figure(figsize=(20, 20))
        plt.imshow(code)
        plt.title('A')
        plt.xlabel('time')
        plt.ylabel('code')

        boundaries = []
        for p in phones:  # type: Phone
            boundaries.append(p.start_frame)
            boundaries.append(p.end_frame)

        plt.vlines(boundaries, 0, ymax)

        plt.savefig(os.path.join(out_dir, f'{utt}.png'))
        plt.close('all')


if __name__ == '__main__':
    main()
