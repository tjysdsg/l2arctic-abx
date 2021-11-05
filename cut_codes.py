import argparse
import os
import librosa
import numpy as np

SR = 44100
HOP_LENGTH = 160
SUBSAMPLE_RATIO = 2  # VQ-VAE half the frequency


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=r'')
    parser.add_argument('--phone-alignment', type=str, default='phone_alignment.txt')
    parser.add_argument('--out-dir', type=str, default='exp')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()
            start, end = float(start), float(end)

            path = os.path.join(args.data_dir, f'{utt}.npy')
            code = np.load(path)

            start_frame, end_frame = (librosa.time_to_frames(
                [start, end], sr=SR, hop_length=HOP_LENGTH
            ) / SUBSAMPLE_RATIO).astype('int')
            code = code[start_frame: end_frame]

            np.save(os.path.join(args.out_dir, f'{utt}_{start}_{phone}.npy'), code)


if __name__ == '__main__':
    main()
