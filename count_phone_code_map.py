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


def get_top_n(mapping: dict):
    ret = {}
    for k, seq in mapping.items():
        vals, counts = np.unique(seq, return_counts=True)
        vals = vals.tolist()
        argidx = np.argsort(counts)[::-1]
        n = len(vals)
        ret[k] = {vals[argidx[i]]: int(counts[argidx[i]]) for i in range(n)}
    return ret


def top_n_to_percent(mapping: dict):
    ret = {}
    for k, counts in mapping.items():
        total = 0
        for _, n in counts.items():
            total += n

        percent = {}
        for p, n in counts.items():
            percent[p] = n / total

        ret[k] = percent

    return ret


def main():
    args = get_args()
    out_dir = args.out_dir
    encode_dir = args.encode_dir
    os.makedirs(out_dir, exist_ok=True)

    code2phones = {}
    phone2codes = {}
    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()
            start, end = float(start), float(end)

            # skip non-existing utterances
            if not os.path.exists(os.path.join(args.encode_dir, f'{utt}.npy')):
                continue

            spk, _ = utt.split('-')

            file = os.path.join(encode_dir, f'{utt}.npy')
            try:
                code_idx = cut_code(file, start, end)
            except FileNotFoundError as e:
                print(e)
                continue

            for ci in code_idx.tolist():
                code2phones.setdefault(ci, []).append(phone)
                phone2codes.setdefault(phone, []).append(ci)

    # get top phone counts of each code
    code2counts = get_top_n(code2phones)

    # get top code counts of each phone
    phone2counts = get_top_n(phone2codes)

    json.dump(code2counts, open(os.path.join(out_dir, 'code2counts.json'), 'w'), indent='  ')
    json.dump(phone2counts, open(os.path.join(out_dir, 'phone2counts.json'), 'w'), indent='  ')

    # get percentage
    json.dump(top_n_to_percent(code2counts), open(os.path.join(out_dir, 'code2percent.json'), 'w'), indent='  ')
    json.dump(top_n_to_percent(phone2counts), open(os.path.join(out_dir, 'phone2percent.json'), 'w'), indent='  ')


if __name__ == '__main__':
    main()
