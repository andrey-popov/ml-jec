#!/usr/bin/env python

"""Build transformation for input features."""

import argparse
import math
import os
from typing import Dict, Iterable, Union

import awkward
from matplotlib import pyplot as plt
import numpy as np
import uproot
import yaml


# Predefined non-linear transformations for each feature.  Linear
# scaling will be added automatically.
CUSTOM_TRANSFORMS = {
    'pt': [
        {
            'type': 'arcsinh',
            'params': {'scale': 10.}
        }
    ],
    'eta': [],
    'phi': [],
    'mass': [
        {
            'type': 'arcsinh',
            'params': {'scale': 10.}
        }
    ],
    'area': [],
    'num_pv': [],
    'rho': []
}


def build_transform(
    sources: Iterable[str], save_path: str
) -> Dict[str, Union[np.ndarray, awkward.JaggedArray]]:
    """Construct preprocessing transformation for input features.

    Apply predefined non-linear transformations.  Then rescale all
    specified features to range [0, 1].  Save parameters of the
    transformations to a YAML file.

    Args:
        sources:  Paths to source ROOT files.
        save_path:  Save path for the YAML file with parameters of the
            transformations.

    Return:
        Transformed features.
    """

    branches = list(CUSTOM_TRANSFORMS.keys())
    shards = list(uproot.iterate(
        sources, 'Jets', branches=branches, entrysteps=math.inf
    ))

    # Merge all shards together
    data = {}
    for branch in branches:
        data[branch] = awkward.concatenate(
            [shard[branch.encode()] for shard in shards]
        )
    del shards

    transforms_all_features = {}
    for feature, values in data.items():
        transforms = []
        values = values.astype(np.float32)

        # Apply all custom transformations
        for cfg in CUSTOM_TRANSFORMS[feature]:
            transforms.append(cfg)
            if cfg['type'] == 'abs':
                values = np.abs(values)
            elif cfg['type'] == 'arcsinh':
                scale = cfg['params']['scale']
                values = np.arcsinh(values / scale)

        # Scale the transformed feature to the range [0, 1]
        loc = float(values.min())
        scale = float(values.max() - loc)
        transforms.append({
            'type': 'linear',
            'params': {'loc': loc, 'scale': scale}
        })
        values -= loc
        values /= scale

        data[feature] = values
        transforms_all_features[feature] = transforms

    with open(save_path, 'w') as f:
        yaml.safe_dump(transforms_all_features, f)
    return data


def plot_features(
    arrays: Dict[str, Union[np.ndarray, awkward.JaggedArray]],
    save_dir: str
) -> None:
    """Plot transformed features.

    Args:
        arrays:  Transformed features.
        save_dir:  Directory where produced plots should be saved.
    """

    for feature, values in arrays.items():
        fig = plt.figure()
        fig.patch.set_alpha(0)
        axes = fig.add_subplot()
        axes.hist(values, range=(0., 1.), bins=50)
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
    data = build_transform(args.sources, args.output)
    if args.plots:
        try:
            os.makedirs(args.plots)
        except FileExistsError:
            pass
        plot_features(data, args.plots)
