import json
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

    out_file = open(os.path.join(args.out_dir, 'abx.txt'), 'w')

    N = len(stimuli)
    for i in range(N):
        A = stimuli[i]
        PaT_Bs = []
        PaT_Xs = []
        PaC_Bs = []
        PaC_Xs = []
        TaP_Bs = []
        TaP_Xs = []
        for j in range(N):
            if i == j:
                continue

            # 1. find B and X stimuli
            s = stimuli[j]
            n_diff_phones = (A.p1.phone != s.p1.phone) + (A.p2.phone != s.p2.phone)

            # PaT
            if n_diff_phones == 1 and A.p1.spk == s.p1.spk:
                PaT_Bs.append(s)
            elif n_diff_phones == 0 and A.p1.spk != s.p1.spk:
                PaT_Xs.append(s)

            # PaC
            if n_diff_phones == 1 and A.p1.spk == s.p1.spk:
                PaC_Bs.append(s)
            elif n_diff_phones == 2 and A.p1.spk == s.p1.spk:
                PaC_Xs.append(s)

            # TaP
            if n_diff_phones == 0 and A.p1.spk != s.p1.spk:
                TaP_Bs.append(s)
            elif n_diff_phones == 1 and A.p1.spk == s.p1.spk:
                TaP_Xs.append(s)

        # 2. generate pairs to list
        for b in PaT_Bs:
            for x in PaT_Xs:
                data = [dataclasses.asdict(e) for e in [A, b, x]]
                out_file.write(f'{json.dumps(data)}\n')
        for b in PaC_Bs:
            for x in PaC_Xs:
                data = [dataclasses.asdict(e) for e in [A, b, x]]
                out_file.write(f'{json.dumps(data)}\n')
        for b in TaP_Bs:
            for x in TaP_Xs:
                data = [dataclasses.asdict(e) for e in [A, b, x]]
                out_file.write(f'{json.dumps(data)}\n')


if __name__ == '__main__':
    main()
