import json

import numpy as np
import os
import argparse
import dataclasses
from typing import List


@dataclasses.dataclass
class Phone:
    utt: str
    phone: str
    spk: str
    start: float
    end: float


@dataclasses.dataclass
class Stimuli:
    p1: Phone
    p2: Phone


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode-dir', type=str, default=r'E:\repos\vqvae_encode_test')
    parser.add_argument('--phone-alignment', type=str, default='phone_alignment.txt')
    parser.add_argument('--out-dir', type=str, default='exp')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prev_phone: Phone = None
    stimuli: List[Stimuli] = []
    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()
            start, end = float(start), float(end)

            spk, _ = utt.split('-')

            p = Phone(utt=utt, phone=phone, spk=spk, start=start, end=end)
            if prev_phone is not None \
                    and prev_phone.utt == p.utt:  # 2-gram should be from the same utterance
                s = Stimuli(p1=prev_phone, p2=p)
                stimuli.append(s)

            prev_phone = p

    # json.dump([dataclasses.asdict(s) for s in stimuli], open(os.path.join(args.out_dir, 'stimuli.json'), 'w'))

    N = len(stimuli)
    # generate PaT triplets
    PaT_triplets = []
    for i in range(1):
        A = stimuli[i]
        Bs = []
        Xs = []
        for j in range(N):
            if i == j:
                continue

            s = stimuli[j]
            n_diff_phones = (A.p1.phone != s.p1.phone) + (A.p2.phone != s.p2.phone)
            if n_diff_phones == 1 and A.p1.spk == s.p1.spk:
                Bs.append(s)
            elif n_diff_phones == 0 and A.p1.spk != s.p1.spk:
                Xs.append(s)

        for b in Bs:
            for x in Xs:
                PaT_triplets.append([A, b, x])

    json.dump(
        [dataclasses.asdict(s) for pat in PaT_triplets for s in pat],
        open(os.path.join(args.out_dir, 'PaT.json'), 'w')
    )


if __name__ == '__main__':
    main()
