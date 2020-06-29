#!/usr/bin/env python

"""Produce plots that illustrate the effect of DNN calibration."""

import argparse
import itertools
import os

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import uproot


def read_data(sources, predictions):
    """Read examples into a pandas Dataframe.

    Args:
        sources:  List of paths to source ROOT files.
        predictions:  Path to .npy file with DNN predictions.  The length and
            order of entries in this array must match the examples in
            the sources.

    Return:
        Dataframe with basic properties of jets.
    """

    dfs = []
    for df in uproot.pandas.iterate(
        sources, 'Jets', branches=[
            'pt_gen', 'eta_gen', 'hadron_flavor', 'parton_flavor',
            'pt', 'eta', 'pt_full_corr'
        ]
    ):
        df['comb_flavor'] = np.where(
            df.hadron_flavor != 0, df.hadron_flavor, np.abs(df.parton_flavor))
        df.drop(columns=['hadron_flavor', 'parton_flavor'], inplace=True)
        dfs.append(df)
    dataframe = pd.concat(dfs)

    predictions = np.load(predictions)
    if len(dataframe) != len(predictions):
        raise RuntimeError(
            'Mismatched numbers of entries in ROOT files and DNN predictions.'
        )
    dataframe['pt_nn_corr'] = dataframe.pt * np.exp(predictions)

    dataframe['response'] = dataframe.pt_full_corr / dataframe.pt_gen
    dataframe['nn_response'] = dataframe.pt_nn_corr / dataframe.pt_gen
    return dataframe


def plot_distrs(dataframe, fig_dir):
    """Plot distributions of response in a few representative bins."""

    binning = np.linspace(0.5, 1.5, num=101)
    pt_bins = [(30., 40.), (100., 110.), (1000., 1100.)]
    eta_bins = [(0., 2.5), (2.5, 5.)]

    ref_histograms, nn_histograms = {}, {}
    for (ipt, pt_bin), (ieta, eta_bin) in itertools.product(
        enumerate(pt_bins), enumerate(eta_bins)
    ):
        df_bin = dataframe[
            (dataframe.pt_gen >= pt_bin[0]) & (dataframe.pt_gen < pt_bin[1]) \
                & (np.abs(dataframe.eta_gen) >= eta_bin[0]) \
                & (np.abs(dataframe.eta_gen) < eta_bin[1])
        ]
        for label, selection in [
            ('uds', (df_bin.comb_flavor <=3) & (df_bin.comb_flavor != 0)),
            ('b', df_bin.comb_flavor == 5),
            ('g', df_bin.comb_flavor == 21)
        ]:
            h, _ = np.histogram(df_bin.response[selection], bins=binning)
            ref_histograms[ipt, ieta, label] = h
            h, _ = np.histogram(df_bin.nn_response[selection], bins=binning)
            nn_histograms[ipt, ieta, label] = h

    for ipt, ieta, flavor in itertools.product(
        range(len(pt_bins)), range(len(eta_bins)), ['uds', 'b', 'g']
    ):
        fig = plt.figure()
        fig.patch.set_alpha(0)
        axes = fig.add_subplot()
        axes.hist(
            binning[:-1], weights=ref_histograms[ipt, ieta, flavor],
            bins=binning, histtype='step', label='Standard')
        axes.hist(
            binning[:-1], weights=nn_histograms[ipt, ieta, flavor],
            bins=binning, histtype='step', label='DNN')
        axes.axvline(1., ls='dashed', lw=0.8, c='gray')
        axes.margins(x=0)
        axes.set_xlabel(
            r'$p_\mathrm{T}^\mathrm{corr}\//\/p_\mathrm{T}^\mathrm{ptcl}$')
        axes.set_ylabel('Jets')
        axes.legend()
        axes.text(
            1., 1.002,
            r'${}$, ${:g} < p_\mathrm{{T}}^\mathrm{{ptcl}} < {:g}$ GeV, '
            r'${:g} < |\eta^\mathrm{{ptcl}}| < {:g}$'.format(
                flavor, pt_bins[ipt][0], pt_bins[ipt][1],
                eta_bins[ieta][0], eta_bins[ieta][1]
            ),
            ha='right', va='bottom', transform=axes.transAxes
        )
        fig.savefig(os.path.join(
            fig_dir, f'{flavor}_pt{ipt + 1}_eta{ieta + 1}.pdf'
        ))
        plt.close(fig)


def bootstrap(x, num=30):
    """Compute errors on median and IQR with bootstrapping."""

    if len(x) == 0:
        return np.nan, np.nan

    medians, iqrs = [], []
    for _ in range(num):
        x_resampled = np.random.choice(x, len(x))
        medians.append(np.median(x_resampled))
        quantiles = np.percentile(x_resampled, [25, 75])
        iqrs.append(quantiles[1] - quantiles[0])
    return np.std(medians), np.std(iqrs)


def compute_iqr(groups):
    """Compute IQR from series GroupBy.

    There is a bug in pandas when computing quantiles from a
    GroupBy [1].
    [1] https://github.com/pandas-dev/pandas/issues/33200
    """

    iqr = []
    for _, series in groups:
        values = series.to_numpy()
        if len(values) <= 1:
            iqr.append(np.nan)
        else:
            q = np.quantile(values, [0.25, 0.75])
            iqr.append(q[1] - q[0])
    return np.asarray(iqr)


def plot_summary(dataframe, fig_dir):
    """Plot median response and its IQR as a function of pt."""

    pt_binning = np.geomspace(20., 3e3, num=20)
    pt_centres = np.sqrt(pt_binning[:-1] * pt_binning[1:])
    for (ieta, eta_bin), (flavor_label, flavors) in itertools.product(
        enumerate([(0., 2.5), (2.5, 5.)]),
        [
            ('uds', {1, 2, 3}), ('b', {5}), ('g', {21}),
            ('all', {0, 1, 2, 3, 4, 5, 21})
        ]
    ):
        df_bin = dataframe[
            (np.abs(dataframe.eta_gen) >= eta_bin[0])
            & (np.abs(dataframe.eta_gen) < eta_bin[1])
            & dataframe.comb_flavor.isin(flavors)
        ]
        bins = df_bin.groupby(pd.cut(df_bin.pt_gen, pt_binning))

        ref_median = bins.response.median().to_numpy()
        ref_iqr = compute_iqr(bins.response)
        ref_median_error = np.empty_like(ref_median)
        ref_iqr_error = np.empty_like(ref_median)
        for i, (_, df) in enumerate(bins):
            ref_median_error[i], ref_iqr_error[i] = bootstrap(
                df.response.to_numpy())

        nn_median = bins.nn_response.median().to_numpy()
        nn_iqr = compute_iqr(bins.nn_response)
        nn_median_error = np.empty_like(ref_median)
        nn_iqr_error = np.empty_like(ref_median)
        for i, (_, df) in enumerate(bins):
            nn_median_error[i], nn_iqr_error[i] = bootstrap(
                df.nn_response.to_numpy())

        fig = plt.figure()
        fig.patch.set_alpha(0)
        axes = fig.add_subplot()
        axes.errorbar(
            pt_centres, ref_median, yerr=ref_median_error,
            marker='o', lw=0, elinewidth=0.8, label='Standard')
        axes.errorbar(
            pt_centres, nn_median, yerr=nn_median_error,
            marker='o', lw=0, elinewidth=0.8, label='DNN')
        axes.axhline(1., ls='dashed', lw=0.8, c='gray')
        axes.set_xlim(pt_binning[0], pt_binning[-1])
        axes.set_xscale('log')
        axes.set_xlabel(r'$p_\mathrm{T}^\mathrm{ptcl}$')
        axes.set_ylabel('Median response')
        axes.legend()
        axes.text(
            1., 1.002,
            '{}${:g} < |\\eta^\\mathrm{{ptcl}}| < {:g}$'.format(
                f'${flavor_label}$, ' if flavor_label != 'all' else '',
                eta_bin[0], eta_bin[1]
            ),
            ha='right', va='bottom', transform=axes.transAxes
        )
        fig.savefig(os.path.join(
            fig_dir, f'{flavor_label}_eta{ieta + 1}_median.pdf'))
        plt.close(fig)

        fig = plt.figure()
        fig.patch.set_alpha(0)
        gs = mpl.gridspec.GridSpec(2, 1, hspace=0.02, height_ratios=[4, 1])
        axes_upper = fig.add_subplot(gs[0, 0])
        axes_lower = fig.add_subplot(gs[1, 0])

        axes_upper.errorbar(
            pt_centres, ref_iqr / ref_median, yerr=ref_iqr_error / ref_median,
            marker='o', lw=0, elinewidth=0.8, label='Standard')
        axes_upper.errorbar(
            pt_centres, nn_iqr / nn_median, yerr=nn_iqr_error / nn_median,
            marker='o', lw=0, elinewidth=0.8, label='DNN')
        axes_lower.plot(
            pt_centres, (nn_iqr / nn_median) / (ref_iqr / ref_median),
            marker='o', lw=0, color='black')

        axes_upper.set_ylim(0., None)
        axes_lower.set_ylim(0.85, 1.02)
        for axes in [axes_upper, axes_lower]:
            axes.set_xscale('log')
            axes.set_xlim(pt_binning[0], pt_binning[-1])
        axes_upper.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        axes_upper.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        axes_upper.legend()
        axes_upper.text(
            1., 1.002,
            '{}${:g} < |\\eta^\\mathrm{{ptcl}}| < {:g}$'.format(
                f'${flavor_label}$, ' if flavor_label != 'all' else '',
                eta_bin[0], eta_bin[1]
            ),
            ha='right', va='bottom', transform=axes_upper.transAxes
        )
        axes_upper.set_ylabel('IQR / median for response')
        axes_lower.set_ylabel('Ratio')
        axes_lower.set_xlabel(r'$p_\mathrm{T}^\mathrm{ptcl}$')
        fig.align_ylabels()

        fig.savefig(os.path.join(
            fig_dir, f'{flavor_label}_eta{ieta + 1}_iqr.pdf'))
        plt.close(fig)


def compare_flavors(dataframe, fig_dir):
    """Plot median response as a function of jet flavour."""

    pt_cut = 30.
    for ieta, eta_bin in enumerate([(0., 2.5), (2.5, 5.)]):
        df_pteta = dataframe[
            (np.abs(dataframe.eta_gen) >= eta_bin[0])
            & (np.abs(dataframe.eta_gen) < eta_bin[1])
            & (dataframe.pt_gen > pt_cut)
        ]
        ref_median, ref_median_error = [], []
        nn_median, nn_median_error = [], []
        flavors = [('g', {21}), ('uds', {1, 2, 3}), ('c', {4}), ('b', {5})]
        for _, pdg_ids in flavors:
            df = df_pteta[df_pteta.comb_flavor.isin(pdg_ids)]
            ref_median.append(df.response.median())
            ref_median_error.append(bootstrap(df.response)[0])
            nn_median.append(df.nn_response.median())
            nn_median_error.append(bootstrap(df.nn_response)[0])

        fig = plt.figure()
        fig.patch.set_alpha(0)
        axes = fig.add_subplot()
        axes.errorbar(
            np.arange(len(flavors)) - 0.02, ref_median, yerr=ref_median_error,
            marker='o', ms=2, lw=0, elinewidth=0.8, label='Standard'
        )
        axes.errorbar(
            np.arange(len(flavors)) + 0.02, nn_median, yerr=nn_median_error,
            marker='o', ms=2, lw=0, elinewidth=0.8, label='DNN'
        )
        axes.set_xlim(-0.5, len(flavors) - 0.5)
        axes.axhline(1, ls='dashed', lw=0.8, c='gray')
        axes.set_xticks(np.arange(len(flavors)))
        axes.set_xticklabels([f[0] for f in flavors])
        axes.legend()
        axes.set_ylabel('Median response')
        axes.text(
            1., 1.002,
            r'$p_\mathrm{{T}}^\mathrm{{ptcl}} > {:g}$ GeV, '
            r'${:g} < |\eta^\mathrm{{ptcl}}| < {:g}$'.format(
                pt_cut, eta_bin[0], eta_bin[1]
            ),
            ha='right', va='bottom', transform=axes.transAxes
        )
        fig.savefig(os.path.join(fig_dir, f'eta{ieta + 1:02d}.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('source_dir', help='Directory with ROOT files.')
    arg_parser.add_argument(
        'start_index', type=int,
        help='Zero-based index of the first source file.'
    )
    arg_parser.add_argument(
        'end_index', type=int,
        help='Zero-based index of the next-to-last source file. If -1, will '
        'take all files till the end.'
    )
    arg_parser.add_argument('nn', help='NumPy file with NN predictions.')
    arg_parser.add_argument(
        '-o', '--fig-dir', default='fig',
        help='Directory where to store produced figures.'
    )
    arg_parser.add_argument(
        '--style', help='Matplotlib style.'
    )
    args = arg_parser.parse_args()

    if args.style:
        plt.style.use(args.style)
    for subdir in ['distrs', 'summary', 'flavors']:
        try:
            os.makedirs(os.path.join(args.fig_dir, subdir))
        except FileExistsError:
            pass

    source_files = [
        os.path.join(args.source_dir, filename)
        for filename in sorted(os.listdir(args.source_dir))
    ]
    if args.end_index == -1:
        args.end_index = len(source_files)
    dataframe = read_data(
        source_files[args.start_index:args.end_index],
        args.nn
    )

    plot_distrs(dataframe, os.path.join(args.fig_dir, 'distrs'))
    plot_summary(dataframe, os.path.join(args.fig_dir, 'summary'))
    compare_flavors(dataframe, os.path.join(args.fig_dir, 'flavors'))
