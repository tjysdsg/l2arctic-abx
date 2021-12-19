import os
import argparse
import dataclasses
from typing import List
import numpy as np
from cut_codes import cut_code

# about 8000 samples for PaT, PaC, TaP each
MAX_NUM_OF_A = 400
MAX_PAIRS_PER_A = 20
N_JOBS = 16


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

    def get_start_end(self):
        assert self.p1.start < self.p2.end
        return self.p1.start, self.p2.end

    def get_utt(self):
        assert self.p1.utt == self.p2.utt
        return self.p1.utt

    def get_phone_pair(self):
        return [self.p1.phone, self.p2.phone]

    def get_phone_pair_str(self):
        return '_'.join(self.get_phone_pair())

    def to_stimuli_id(self):
        """
        utt_phone1_phone2_start_end
        """
        start, end = self.get_start_end()
        return f'{self.get_utt()}_{self.get_phone_pair_str()}_{start}_{end}'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode-dir', type=str)
    parser.add_argument('--phone-alignment', type=str)
    parser.add_argument('--utt2spk', type=str, default=None)
    parser.add_argument('--freq', type=int)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def save_abx_to_files(abx: List[Stimuli], encode_dir: str, out_dir: str, freq: int):
    for e in abx:
        start, end = e.get_start_end()
        utt = e.get_utt()
        file = os.path.join(encode_dir, f'{utt}.npy')
        code = cut_code(file, start, end, freq)
        if code.shape[0] <= 1:
            raise RuntimeError('Code too short')
        np.save(os.path.join(out_dir, f'{e.to_stimuli_id()}.npy'), code)


def generate_from_bx_list(args, a: Stimuli, out_file, bs: List[Stimuli], xs: List[Stimuli]):
    np.random.shuffle(bs)
    np.random.shuffle(xs)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # parallel execution of `save_abx_to_files([a, b, x], args.encode_dir, args.out_dir)` with exceptions
    def worker(*_args):
        ret = True
        try:
            save_abx_to_files(*_args)
        except FileNotFoundError as e:
            print(e)
            ret = False
        except RuntimeError:
            ret = False

        return ret

    executor = ThreadPoolExecutor(max_workers=N_JOBS)
    thread2out_line = {}

    n = 0
    for b in bs:
        for x in xs:
            if n < MAX_PAIRS_PER_A:
                thread2out_line[
                    executor.submit(worker, [a, b, x], args.encode_dir, args.out_dir, args.freq)
                ] = f'{a.to_stimuli_id()} {b.to_stimuli_id()} {x.to_stimuli_id()}\n'
                n += 1
            else:
                break

    for future in as_completed(thread2out_line):
        if future.result():
            out_file.write(thread2out_line[future])


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    utt2spk = {}
    if args.utt2spk is not None:
        with open(args.utt2spk) as f:
            for line in f:
                utt, spk = line.strip('\n').split()
                utt2spk[utt] = spk

    prev_phone: Phone = None
    stimuli: List[Stimuli] = []
    with open(args.phone_alignment) as f:
        for line in f:
            utt, start, end, phone = line.strip('\n').split()

            if phone in ['SIL', 'sil']:
                continue

            # skip non-existing utterances
            if not os.path.exists(os.path.join(args.encode_dir, f'{utt}.npy')):
                continue

            start, end = float(start), float(end)

            if args.utt2spk is not None:
                spk = utt2spk[utt]
            else:
                spk, _ = utt.split('-')

            p = Phone(utt=utt, phone=phone, spk=spk, start=start, end=end)
            # 2-gram should be from the same utterance
            # we also avoid gap in two phones
            if prev_phone is not None \
                    and prev_phone.utt == p.utt \
                    and prev_phone.end == p.start:
                s = Stimuli(p1=prev_phone, p2=p)
                stimuli.append(s)

            prev_phone = p

    # json.dump([dataclasses.asdict(s) for s in stimuli], open(os.path.join(args.out_dir, 'stimuli.json'), 'w'))

    PaT_file = open(os.path.join(args.out_dir, 'PaT.txt'), 'w')
    PaC_file = open(os.path.join(args.out_dir, 'PaC.txt'), 'w')
    TaP_file = open(os.path.join(args.out_dir, 'TaP.txt'), 'w')

    N = len(stimuli)
    np.random.shuffle(stimuli)
    for i in range(MAX_NUM_OF_A):
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

        # 2. generate pairs and cut codes
        generate_from_bx_list(args, A, PaT_file, PaT_Bs, PaT_Xs)
        generate_from_bx_list(args, A, PaC_file, PaC_Bs, PaC_Xs)
        generate_from_bx_list(args, A, TaP_file, TaP_Bs, TaP_Xs)


if __name__ == '__main__':
    main()
