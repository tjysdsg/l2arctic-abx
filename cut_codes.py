import argparse
import os
import numpy as np

SR = 16000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--phone-alignment', type=str)
    parser.add_argument('--freq', type=int)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def cut_code(path: str, start: float, end: float, freq: int):
    code = np.load(path)
    start_frame, end_frame = int(np.round(start * freq)), int(np.round(end * freq))
    return code[start_frame: end_frame]


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()
            start, end = float(start), float(end)

            path = os.path.join(args.data_dir, f'{utt}.npy')
            try:
                code = cut_code(path, start, end, args.freq)
            except FileNotFoundError as e:
                print(e)
                continue

            np.save(os.path.join(args.out_dir, f'{utt}_{start}_{phone}.npy'), code)


if __name__ == '__main__':
    main()
