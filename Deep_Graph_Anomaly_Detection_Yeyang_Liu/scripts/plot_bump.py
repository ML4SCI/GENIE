"""Plot Top 1%, 0.1%, 0.01% mass selections from saved event scores."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d


FRACTIONS = (0.01, 0.001, 0.0001)
BIN_EDGES_GEV = np.arange(1500., 7100., 100.)
SMOOTHING = dict(sigma=1.5, order=0, mode='reflect', cval=0., truncate=4.)
COLORS = dict(raw='#E9A39B', total='#D94B3D', background='#526477',
              signal='#1769D2', text='#193852', grid='#E3EAF0')


def read_scores(path, score_key):
    """Keep masses, truth and scores in the inference file's (source, row) order."""
    with np.load(path) as data:
        events = {name: data[name] for name in ('source', 'row', 'mjj', 'truth')}
        events['scores'] = data[score_key]
        metadata = {name: data[name].tolist() for name in (
            'mass_unit', 'source_files', 'checkpoint', 'epoch', 'calibration_file',
            'baseline_epoch', 'reg_epoch', 'split_file', 'split', 'score', 'edge_weight',
        ) if name in data}
    size = len(events['row'])
    if size == 0 or any(values.shape != (size,) for values in events.values()):
        raise ValueError('Expected nonempty, aligned one-dimensional event arrays')
    if not np.isfinite(events['mjj']).all() or not np.isfinite(events['scores']).all():
        raise ValueError('Masses and scores must be finite; no events are silently removed')
    if not np.isin(events['truth'], (0, 1)).all():
        raise ValueError('Expected truth=0 for background and truth=1 for signal')
    identifiers = np.column_stack([events['source'], events['row']])
    if len(np.unique(identifiers, axis=0)) != size:
        raise ValueError('Duplicate (source, row) event identifiers')
    events['mjj'] = events['mjj'].astype(np.float64) * {'GeV': 1., 'TeV': 1000.}[metadata['mass_unit']]
    return events, metadata


def selection_histograms(events):
    """Select on the full input sample before applying the histogram mass range."""
    mass, scores, truth = (events[name] for name in ('mjj', 'scores', 'truth'))
    arrays = dict(bin_edges_gev=BIN_EDGES_GEV, bin_edges_tev=BIN_EDGES_GEV / 1000,
                  fractions=np.array(FRACTIONS))
    selections, raw_histograms, smooth_histograms = [], [], []
    previous = np.ones(len(scores), dtype=bool)
    for index, fraction in enumerate(FRACTIONS):
        threshold = float(np.quantile(scores, 1 - fraction, method='linear'))
        selected = scores >= threshold  # Include every tie; do not force a top-k count.
        assert not (selected & ~previous).any()
        previous = selected
        masks = (selected, selected & (truth == 0), selected & (truth == 1))
        counts = [int(mask.sum()) for mask in masks]
        raw = np.stack([np.histogram(mass[mask], bins=BIN_EDGES_GEV)[0] for mask in masks])
        smooth = gaussian_filter1d(raw.astype(np.float64), axis=-1, **SMOOTHING)
        np.testing.assert_array_equal(raw[0], raw[1] + raw[2])
        np.testing.assert_allclose(smooth[0], smooth[1] + smooth[2], rtol=1e-13, atol=1e-12)
        selections.append(dict(
            title=f'Top {100 * fraction:g}%', fraction=fraction, threshold=threshold,
            n_selected=counts[0], n_background=counts[1], n_signal=counts[2],
            signal_fraction=counts[2] / counts[0], effective_fraction=counts[0] / len(scores),
            n_equal_threshold=int((scores == threshold).sum()),
            in_plot_range=dict(zip(('total', 'background', 'signal'), raw.sum(axis=1).tolist())),
            outside_plot_range=counts[0] - int(raw[0].sum()),
        ))
        for name in ('source', 'row'):
            arrays[f'selected_{name}_{index}'] = events[name][selected]
        raw_histograms.append(raw)
        smooth_histograms.append(smooth)
    for index, name in enumerate(('total', 'background', 'signal')):
        arrays[f'raw_{name}'] = np.stack([raw[index] for raw in raw_histograms])
        arrays[f'smoothed_{name}'] = np.stack([smooth[index] for smooth in smooth_histograms])
    arrays['thresholds'] = np.array([selection['threshold'] for selection in selections])
    return arrays, selections


class StepLegend(HandlerBase):
    """Show a step, rather than a filled patch, for the raw histogram legend."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x = np.array([0, .3, .3, .7, .7, 1]) * width - xdescent
        y = np.array([.25, .25, .8, .8, .25, .25]) * height - ydescent
        return [Line2D(x, y, color=COLORS['raw'], lw=.9, alpha=.85, transform=trans)]


def plot_histograms(arrays, selections):
    """Same counting scale for all curves; each panel has its own linear y range."""
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans'],
        'text.usetex': False,
        'text.color': COLORS['text'], 'axes.labelcolor': COLORS['text'],
        'axes.edgecolor': COLORS['background'], 'xtick.color': COLORS['background'],
        'ytick.color': COLORS['background'],
    })
    edges = arrays['bin_edges_tev']
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.), facecolor='white')
    fig.subplots_adjust(left=.075, right=.986, top=.88, bottom=.28, wspace=.24)
    for index, (ax, selection) in enumerate(zip(axes, selections)):
        raw = arrays['raw_total'][index]
        ax.stairs(raw, edges, color=COLORS['raw'], lw=.9, alpha=.85, fill=False, zorder=2)
        maximum = float(raw.max())
        for name, width, linestyle, zorder in (
            ('total', 2.4, '-', 3), ('background', 1.8, (0, (5, 3)), 4), ('signal', 2.5, '-', 5),
        ):
            curve = arrays[f'smoothed_{name}'][index]
            ax.plot(centers, curve, color=COLORS[name], lw=width, ls=linestyle, zorder=zorder)
            maximum = max(maximum, float(curve.max()))
        ax.set_title(selection['title'], fontsize=20, fontweight='semibold', pad=15)
        ax.set_xlim(1.5, 7.)
        ax.set_xticks([2, 3, 4, 5, 6, 7])
        ax.set_ylim(0, 1.10 * maximum if maximum else 1.)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=4))
        ax.tick_params(labelsize=14, width=.8, length=4, pad=5)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_linewidth(.8)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=COLORS['grid'], linewidth=.65)
    fig.supxlabel('Dijet mass [TeV]', fontsize=16, y=.145)
    fig.supylabel('Events / 100 GeV', fontsize=16, x=.012)
    step = Line2D([], [], color=COLORS['raw'], lw=.9, alpha=.85)
    fig.legend([step, *axes[0].lines], [
        'Selected events (raw)', 'Selected events (smoothed)',
        'Selected background (GT)', 'Selected signal (GT)',
    ], ncol=4, loc='lower center', bbox_to_anchor=(.5, .018), frameon=False,
        fontsize=13, handlelength=2.5, columnspacing=1.5, handletextpad=.6,
        handler_map={step: StepLegend()})
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True, help='event NPZ from scripts.inference')
    parser.add_argument('--score', default='total', help='score array in the NPZ; larger is more anomalous')
    parser.add_argument('--output', type=Path, required=True, help='output filename stem, without extension')
    args = parser.parse_args()
    if args.output.with_suffix('.npz').resolve() == args.scores.resolve():
        parser.error('--output must differ from the input score filename')
    events, provenance = read_scores(args.scores, args.score)
    arrays, selections = selection_histograms(events)
    signal = int((events['truth'] == 1).sum())
    count = len(events['truth'])
    full_sample = dict(total=count, background=count - signal, signal=signal, signal_fraction=signal / count)
    print(f'Full sample: {count - signal:,} background + {signal:,} signal ({100 * signal / count:.6f}%)')
    print('selection | N_selected | N_background | N_signal | signal fraction')
    for selected in selections:
        print(f"{selected['title']} | {selected['n_selected']} | {selected['n_background']} | "
              f"{selected['n_signal']} | {100 * selected['signal_fraction']:.4f}%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_histograms(arrays, selections)
    fig.savefig(args.output.with_suffix('.png'), dpi=300, facecolor='white')
    plt.close(fig)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    command = shlex.join([sys.executable, '-m', 'scripts.plot_bump', *sys.argv[1:]])
    metadata = dict(
        input_file=str(args.scores.resolve()), inference=provenance, score_key=args.score,
        score_definition='Stored event score, used without recomputation, reweighting or rescaling; higher is more anomalous',
        labels={'0': 'background', '1': 'signal'}, full_sample=full_sample, selections=selections,
        selection_rule='Full input sample: numpy.quantile(scores, 1-fraction, method=linear); score >= threshold; all ties included; mass range applied afterwards',
        binning=dict(width_gev=100, edges_gev=BIN_EDGES_GEV.tolist(), normalized=False, signal_scale=1),
        smoothing=dict(function='scipy.ndimage.gaussian_filter1d', **SMOOTHING,
                       sigma_unit='bins', sigma_gev=150., radius_bins=6,
                       input='float64 histogram counts within 1500–7000 GeV'),
        style=dict(colors=COLORS, figsize_inches=[13.2, 5.], dpi=300),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        command=command, working_directory=str(Path.cwd()),
    )
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    args.output.with_name(args.output.name + '_command.txt').write_text(command + '\n')
    print(f'Saved {args.output}.png / .npz / .json')


if __name__ == '__main__':
    main()
