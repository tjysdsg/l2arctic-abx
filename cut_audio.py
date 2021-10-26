import argparse
import os
import librosa
import soundfile

SR = 44100


def utt2wav_path(s: str):
    """
    Convert utt ID to the subpath of the wav file

    Specific to L2-ARCTIC folder structure
    """
    spk, filename = s.split('-', maxsplit=1)
    return os.path.join(spk, 'wav', f'{filename}.wav')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/e/datasets/l2arctic')
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

            wav_path = os.path.join(args.data_dir, utt2wav_path(utt))
            wav, _ = librosa.load(wav_path, sr=SR)
            start_frame, end_frame = librosa.time_to_samples([start, end], sr=SR)
            wav = wav[start_frame: end_frame]
            soundfile.write(os.path.join(args.out_dir, f'{utt}_{start}_{phone}.wav'), wav, samplerate=SR)


if __name__ == '__main__':
    main()
