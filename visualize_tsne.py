import json
import os
import argparse
import numpy as np
from utils import convert_time_to_frame_idx
from openTSNE import TSNE
import plotly.express as px


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode-dir', type=str)
    parser.add_argument('--phone-alignment', type=str)
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--sample-rate', type=int, default=None)
    parser.add_argument('--hop-length', type=int, default=None)
    parser.add_argument('--max_n', type=int, default=100)
    parser.add_argument('--out-dir', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    encode_dir = args.encode_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    n_utts = 0
    encodings: list = []
    labels = []
    prev_utt = None
    with open(args.phone_alignment) as f:
        for line in f:
            if n_utts >= args.max_n:
                break

            utt, start, end, phone = line.strip('\n').split()

            if prev_utt != utt and prev_utt is not None:
                encodings: np.ndarray = np.vstack(encodings)
                print(f'{utt}: size of training data:', encodings.shape)
                model = TSNE(n_components=2, random_state=42, n_jobs=32)
                model = model.fit(encodings)
                tsne = model.transform(encodings)

                assert len(tsne) == len(labels)

                # scatter
                fig = px.scatter(x=tsne[:, 0], y=tsne[:, 1], color=labels, text=labels)
                fig.write_html(os.path.join(out_dir, f"tsne_{prev_utt}.html"))

                # reset
                n_utts += 1
                encodings = []
                labels = []
                prev_utt = None

            # skip non-existing utterances
            if not os.path.exists(os.path.join(encode_dir, f'{utt}.npy')):
                continue

            start, end = float(start), float(end)
            model_config = json.load(open(args.model_config))
            start_frame, end_frame = (
                convert_time_to_frame_idx(start, model_config, args.sample_rate, args.hop_length),
                convert_time_to_frame_idx(end, model_config, args.sample_rate, args.hop_length)
            )

            if end_frame - start_frame < 2:
                continue

            # plot tsne for each utterance
            code = np.load(os.path.join(encode_dir, f'{utt}.npy'))[start_frame:end_frame]
            labels += [phone for _ in range(end_frame - start_frame)]
            encodings.append(code)

            prev_utt = utt


if __name__ == '__main__':
    main()
