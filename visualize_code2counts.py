import argparse
import json
import plotly.graph_objects as go


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code2counts', type=str, default=r'exp/code2counts.json')
    parser.add_argument('--out-path', type=str, default=r'exp/code2counts_viz.html')
    return parser.parse_args()


def main():
    args = get_args()

    codes = []
    phones = []
    freq = []
    code2counts = json.load(open(args.code2counts))

    for c, counts in code2counts.items():
        for p, n in counts.items():
            codes.append(c)
            phones.append(p)
            freq.append(n)

    # draw parallel sets
    fig = go.Figure(go.Parcats(
        dimensions=[
            {
                'label': 'Code',
                'values': codes,
            },
            {
                'label': 'Phones',
                'values': phones,
            },
        ],
        counts=freq,
    ))
    fig.write_html(args.out_path)


if __name__ == '__main__':
    main()
