import os
import argparse
import json
import numpy as np
from cut_codes import cut_code


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encode-dir', type=str,
        default=r'/home/storage15/tangjiyang/DAU-MD/exp/zerospeech_vae/encode_i'
    )
    parser.add_argument('--phone-alignment', type=str, default='phone_alignment.txt')
    parser.add_argument('--out-dir', type=str, default='exp')
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    encode_dir = args.encode_dir
    os.makedirs(out_dir, exist_ok=True)

    code2phones = {}
    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()
            start, end = float(start), float(end)

            spk, _ = utt.split('-')

            file = os.path.join(encode_dir, f'{utt}.npy')
            try:
                code_idx = cut_code(file, start, end)
            except FileNotFoundError as e:
                print(e)
                continue

            for ci in code_idx:
                code2phones.setdefault(ci, []).append(phone)

    # get top phone counts of each code
    code2counts = {}
    for ci, phones in code2phones.items():
        vals, counts = np.unique(phones, return_counts=True)
        argidx = np.argsort(counts)[::-1]
        n = len(vals)
        code2counts[ci] = {vals[argidx[i]]: counts[argidx[i]] for i in range(n)}

    json.dump(code2counts, open(os.path.join(out_dir, 'code2counts.json'), 'w'), indent='  ')


if __name__ == '__main__':
    main()
