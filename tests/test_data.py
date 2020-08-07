import collections
import os

import numpy as np
import tensorflow as tf

from mljec import data


def test_read_root():
    branches = np.array([b'pt', b'eta'])
    arrays = data._read_root_file(b'data/shards/1.root', branches)
    assert len(arrays) == len(branches)
    for a in arrays:
        assert isinstance(a, np.ndarray)
        assert len(a.shape) == 2 and a.shape[1] == 1


def test_read_root_wrapper():
    path = tf.constant('data/shards/1.root')
    result = data._read_root_file_wrapper(path)
    assert isinstance(result, collections.abc.Mapping)


def test_dataset():
    input_files = [
        os.path.join('data', 'shards', f'{i + 1}.root')
        for i in range(5)
    ]
    batch_size = 5
    dataset = data._build_dataset(input_files, None, batch_size=batch_size)
    batch = next(iter(dataset))
    assert len(batch) == 2
    shape = batch[0].shape
    assert len(shape) == 2 and shape[0] == batch_size and shape[1] == 7
    shape = batch[1].shape
    assert len(shape) == 2 and shape[0] == batch_size and shape[1] == 1


def test_datasets_full():
    config = {'data': {
        'location': 'data',
        'split': [3, 1, -1],
        'batch_size': 5
    }}
    metadata, train_ds, val_ds, test_ds = data.build_datasets(config)
    assert 'counts' in metadata and 'features' in metadata
    assert len(train_ds.element_spec) == 2
