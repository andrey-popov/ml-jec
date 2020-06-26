#!/usr/bin/env python

"""Convert files from ROOT to protobuf messages."""

import argparse
import math
import os

import numpy as np
import tensorflow as tf
import uproot
import yaml


def serialize_example(series):
    """Serialize a single example into protobuf massage.

    Args:
        series:  pandas series representing the example.

    Return:
        String with serialized protobuf message.
    """

    features = {}
    for name, value in series.items():
        features[name] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def convert_file(path, save_path):
    """Convert a ROOT file to protobuf messages.

    Define the regression target and drop features that cannot be used
    by the model.

    Args:
        path:  Path to the input file.
        save_path:  Path for the output file.

    Return:
        Number of examples written to the converted file.
    """

    dataframe = next(iter(uproot.pandas.iterate(
        path, 'Jets', entrysteps=math.inf
    )))

    # Target for regression
    dataframe['target'] = np.log(dataframe.pt_gen / dataframe.pt)

    # Drop features that won't be used in deployment
    dataframe.drop(
        [
            'pt_gen', 'eta_gen', 'phi_gen', 'mass_gen',
            'hadron_flavor', 'parton_flavor',
            'pt_full_corr'
        ],
        axis='columns', inplace=True
    )

    with tf.io.TFRecordWriter(
        save_path,
        tf.io.TFRecordOptions(compression_type='GZIP')
    ) as writer:
        for _, row in dataframe.iterrows():
            example = serialize_example(row)
            writer.write(example)
    return len(dataframe)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        'source_dir', help='Directory with files to be converted.'
    )
    arg_parser.add_argument(
        '-o', '--output', default='protobuf',
        help='Name for root directory for converted files.'
    )
    args = arg_parser.parse_args()

    sources = [
        os.path.join(args.source_dir, filename)
        for filename in os.listdir(args.source_dir)
        if filename.endswith('.root')
    ]
    sources.sort()
    os.makedirs(args.output)
    counts = {}

    for source in sources:
        output_filename = os.path.splitext(os.path.basename(source))[0] \
            + '.tfrecord.gz'
        n = convert_file(source, os.path.join(args.output, output_filename))
        counts[output_filename] = n

    with open(os.path.join(args.output, 'counts.yaml'), 'w') as f:
        yaml.safe_dump(counts, f)

