#!/usr/bin/env python

"""Build transformation for input features."""

import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import uproot
import yaml


def build_transform(sources, save_path):
    """Construct preprocessing transformation for input features.

    Apply manually chosen non-linear transformation to certain features.
    After that, rescale all features to range [0., 1.].  Save parameters
    of the transformations to a YAML file.

    Args:
        sources:  List of paths to source ROOT files.
        save_path:  Save path for the YAML file with parameters of the
            transformations.

    Return:
        pandas dataframe after the transformation.
    """

    features_to_drop = [
        'pt_gen', 'eta_gen', 'phi_gen', 'mass_gen',
        'hadron_flavor', 'parton_flavor',
        'pt_full_corr'
    ]
    dataframe = pd.concat(
        df.drop(columns=features_to_drop)
        for df in uproot.pandas.iterate(sources, 'Jets')
    )

    # Use arcsinh transformation to features with heavy tails.  Choose
    # arcsinh instead of log because these features often approach or
    # reach zero.  Describe transformations with a mapping from feature
    # name to the scale to be used for arcsinh.
    arcsinh_scales = {
        'pt': 10., 'pt_l1corr': 10., 'mass': 10.,
        'mult_mu': 1., 'mult_el': 1., 'mult_ph': 1.,
        'mult_ch_had': 1., 'mult_ne_had': 1.,
        'mult_hf_em': 1., 'mult_hf_had': 1.,
        'mean_dr2': 1e-3, 'pull_magnitude': 1.,
        'sv_mass': 1., 'sv_pt_frac': 0.01, 'sv_distance': 0.1,
        'sv_significance': 1., 'sv_num_tracks': 1.
    }

    # Apply the non-linear transformations and scale the resulting
    # features to range [0, 1]
    transforms = []
    for feature in dataframe.columns:
        transform = {'feature': feature}
        if feature in arcsinh_scales:
            scale = arcsinh_scales[feature]
            transform['arcsinh'] = {'scale': scale}
            dataframe[feature] = np.arcsinh(dataframe[feature] / scale)
        transform['loc'] = float(dataframe[feature].min())
        transform['scale'] = float(dataframe[feature].max() - transform['loc'])
        dataframe[feature] -= transform['loc']
        dataframe[feature] /= transform['scale']
        transforms.append(transform)

    with open(save_path, 'w') as f:
        yaml.safe_dump(transforms, f)
    return dataframe


def plot_features(dataframe, save_dir):
    """Plot transformed features.

    Args:
        dataframe:  pandas dataframe with transformed features.
        save_dir:  Directory where produced plots should be saved.
    """

    for feature, series in dataframe.iteritems():
        fig = plt.figure()
        fig.patch.set_alpha(0)
        axes = fig.add_subplot()
        axes.hist(series, range=(0., 1.), bins=50)
        axes.set_xlim(0., 1.)
        axes.set_xlabel(feature)
        fig.savefig(os.path.join(save_dir, feature + '.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        'sources', nargs='+',
        help='ROOT files on which to build transformation.'
    )
    arg_parser.add_argument(
        '-o', '--output', default='transform.yaml',
        help='Path for output file describing transformations.'
    )
    arg_parser.add_argument(
        '--plots', help='If given, specifies directory for distributions '
        'of transformed features.'
    )
    args = arg_parser.parse_args()
    transformed_dataframe = build_transform(args.sources, args.output)
    if args.plots:
        try:
            os.makedirs(args.plots)
        except FileExistsError:
            pass
        plot_features(transformed_dataframe, args.plots)
