import dataclasses
import json
import math
import os
import argparse
import numpy as np
from utils import convert_time_to_frame_idx
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rc("savefig", dpi=300)

MAX_FEATURE_DIM = 128


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--phone-alignment', type=str)
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--sample-rate', type=int, default=None)
    parser.add_argument('--hop-length', type=int, default=None)
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
            model_config = json.load(open(args.model_config))
            start_frame, end_frame = (
                convert_time_to_frame_idx(start, model_config, args.sample_rate, args.hop_length),
                convert_time_to_frame_idx(end, model_config, args.sample_rate, args.hop_length)
            )

            code = np.load(os.path.join(data_dir, f'{utt}.npy'))

            p = Phone(utt, phone, start_frame, end_frame)
            utt2phones.setdefault(utt, []).append(p)
            utt2code[utt] = code

    for utt, phones in utt2phones.items():
        code = utt2code[utt].T
        code = code[:MAX_FEATURE_DIM]

        # clip values
        vmin = np.quantile(code, 0.1)
        vmax = np.quantile(code, 0.9)
        code = np.clip(code, vmin, vmax)

        plt.figure(figsize=(20, 15))
        plt.imshow(code, cmap='viridis')
        plt.title(f'Note that only the first {MAX_FEATURE_DIM} values along the feature dimension are shown')
        plt.xlabel('time')
        plt.ylabel('code')
        plt.colorbar(orientation='horizontal')

        prev_end_frame = -100
        for p in phones:  # type: Phone
            if abs(p.start_frame - prev_end_frame) > 1:
                plt.gca().axvline(p.start_frame, ymin=0, ymax=1, lw=1, color='white')
            plt.gca().axvline(p.end_frame, ymin=0, ymax=1, lw=1, color='white')

        plt.savefig(os.path.join(out_dir, f'{utt}.png'))
        plt.close('all')


if __name__ == '__main__':
    main()
