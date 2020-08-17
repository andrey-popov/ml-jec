import collections
import os

import numpy as np
import tensorflow as tf

from mljec import data


def test_create_mask():
    lengths = [0, 3, 5]
    max_length = 5
    mask = data._create_mask(np.asarray(lengths), max_length)
    true_mask = np.asarray([
        [True] * n + [False] * (max_length - n)
        for n in lengths
    ])
    assert np.array_equal(mask, true_mask)


def test_read_root():
    branches_global = np.array([b'pt', b'eta'])
    branches_ch = np.array([b'ch_pt', b'ch_eta'])
    max_length = 32
    arrays = data._read_root_file(
        b'data/shards/1.root', branches_global, max_length, branches_ch
    )
    assert len(arrays) == (len(branches_global) + len(branches_ch) + 1)
    for a in arrays:
        assert isinstance(a, np.ndarray)
    for a in arrays[:len(branches_global)]:
        assert len(a.shape) == 2 and a.shape[1] == 1
    for a in arrays[len(branches_global):]:
        assert len(a.shape) == 2 and a.shape[1] == max_length


def test_read_root_wrapper():
    path = tf.constant('data/shards/1.root')
    branches_global = ['pt', 'eta', 'mass', 'rho']
    branches_ch_numeric = ['ch_pt', 'ch_eta', 'ch_lost_hits']
    features = {
        'global': {'numeric': branches_global},
        'ch': {
            'max_size': 64,
            'numeric': branches_ch_numeric
        }
    }
    result = tf.function(
        lambda path: data._read_root_file_wrapper(path, features),
        input_signature=[tf.TensorSpec((), dtype=tf.string)]
    )(path)
    assert isinstance(result, collections.abc.Mapping)
    columns = ['pt_gen'] + branches_global + ['ch_mask'] + branches_ch_numeric
    for column in columns:
        assert column in result


def test_dataset():
    input_files = [
        os.path.join('data', 'shards', f'{i + 1}.root')
        for i in range(5)
    ]
    branches_global = ['pt', 'eta', 'mass', 'rho']
    branches_ch_numeric = ['ch_pt', 'ch_eta', 'ch_lost_hits']
    max_length = 64
    features = {
        'global': {'numeric': branches_global},
        'ch': {
            'max_size': max_length,
            'numeric': branches_ch_numeric
        }
    }
    batch_size = 5
    dataset = data._build_dataset(
        input_files, features, {}, batch_size=batch_size
    )
    batch = next(iter(dataset))

    assert len(batch) == 2
    inputs = batch[0]
    target = batch[1]
    assert isinstance(inputs, collections.abc.Mapping)

    assert 'global_numeric' in inputs
    shape = inputs['global_numeric'].shape
    assert (
        len(shape) == 2 and shape[0] == batch_size
        and shape[1] == len(branches_global)
    )
    assert inputs['global_numeric'].dtype == tf.float32

    assert 'ch_mask' in inputs
    shape = inputs['ch_mask'].shape
    assert (
        len(shape) == 2 and shape[0] == batch_size and shape[1] == max_length
    )
    assert inputs['ch_mask'].dtype == tf.bool

    assert 'ch_numeric' in inputs
    shape = inputs['ch_numeric'].shape
    assert (
        len(shape) == 3 and shape[0] == batch_size and shape[1] == max_length
        and shape[2] == len(branches_ch_numeric)
    )
    assert inputs['ch_numeric'].dtype == tf.float32

    shape = target.shape
    assert len(shape) == 2 and shape[0] == batch_size and shape[1] == 1
    assert target.dtype == tf.float32


def test_datasets_full():
    config = {
        'data': {
            'location': 'data',
            'split': [3, 1, -1],
            'batch_size': 5
        },
        'features': {
            'global': {
                'numeric': ['pt', 'eta', 'mass', 'rho']
            },
            'ch': {
                'max_size': 64,
                'numeric': ['ch_pt', 'ch_eta', 'ch_lost_hits']
            }
        }
    }
    metadata, train_ds, val_ds, test_ds = data.build_datasets(config)
    assert 'counts' in metadata and 'features' in metadata
    for ds in [train_ds, val_ds, test_ds]:
        assert isinstance(ds, tf.data.Dataset)
        assert len(ds.element_spec) == 2
