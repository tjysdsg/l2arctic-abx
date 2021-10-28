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

    json.dump([dataclasses.asdict(s) for s in stimuli], open(os.path.join(args.out_dir, 'stimuli.json'), 'w'))

    # generate PaT triplets


if __name__ == '__main__':
    main()
