import pandas as pd
import numpy as np
import sys
import os
import argparse


ARG_PARSER = argparse.ArgumentParser(
    description=(
        'convert'
    )
)

ARG_PARSER.add_argument(
    '--n_start_frame', type=int, required=True,
    help=(
        'The start frame #. Should be larger than 0.'
    )
)

ARG_PARSER.add_argument(
    '--fps', type=float, required=True,
    help=(
        'FPS of the video.'
    )
)

ARG_PARSER.add_argument(
    '--minutes', type=float, required=True,
    help=(
        'Length of time. Frames from `n_start_frame` to `n_start_frame + 60 * minutes * fps` are extracted'
    )
)

ARG_PARSER.add_argument(
    '--len_y', type=float, required=True,
    help=(
        'The actual y-length from the origin to the coordinate. Should be larger than 0.'
    )
)

ARG_PARSER.add_argument(
    '--filepath', type=str, required=True,
    help=(
        'CSV file'
    )
)

ARG_PARSER.add_argument(
    '--debug', type=bool, required=False, default=False,
    help=(
        'Debug flag.'
    )
)

def main(args):
    df = pd.read_csv(args.filepath, index_col='position')

    df = df[
        (args.n_start_frame + 60 * args.minutes * args.fps >= df.index)
        & (df.index >= args.n_start_frame)
    ]

    l = list(df.columns)
    l = [l[x:x+2] for x in range(0, len(l), 2)]

    v_orig = l[-2]
    v_y_coord = l[-1]
    l = l[:-2]

    for v in l:
        df[v[0]] -= df[v_orig[0]]
        df[v[1]] = df[v_orig[1]] - df[v[1]]

    df[v_y_coord[0]] -= df[v_orig[0]]
    df[v_y_coord[1]] = df[v_orig[1]] - df[v_y_coord[1]]

    df_coord = df[v_y_coord]

    def func(row):
        v_coord = df_coord.loc[row.name]
        r = np.linalg.norm(v_coord)
        sin = v_coord[0]
        cos = v_coord[1]

        result = args.len_y * row.dot(np.array([[cos, -sin], [sin, cos]]).T) / (r * r)

        return result

    for v in l:
        df[v] = df[v].apply(func, axis=1, result_type='broadcast')

    path, ext = os.path.splitext(args.filepath)

    df[sum(l, [])].to_csv('{}-converted{}'.format(path, ext))

    if not args.debug:
        return

    import matplotlib.pylab as plt
    fig, ax = plt.subplots()

    for v in l:
        df[v].plot(x=v[0], y=v[1], ax=ax)
    
    plt.show()

if __name__ == '__main__':
    args = ARG_PARSER.parse_args()
    main(args)
