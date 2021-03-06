import collections
import itertools
import os

import numpy as np
import tensorflow as tf

import mljec


JETS_IN_FILE = 10


def test_transform_op_categorical():
    allowed_values = [-42, 0, 7, 42]
    transform = mljec.data._create_transform_op_categorical(allowed_values)
    x = tf.constant([[7, -42, -42, 7, 42]], dtype=tf.int32)
    mapped_x = transform(x)
    expected_result = np.array([[2, 0, 0, 2, 3]], dtype=np.int32)
    assert np.array_equal(mapped_x.numpy(), expected_result)


def test_read_root():
    branches_global = np.array([b'pt', b'eta'])
    branches_ch_num = np.array([b'ch_pt', b'ch_eta'])
    branches_ch_cat = np.array([b'ch_id'])
    arrays = mljec.data._read_root_file(
        b'data/shards/1.root', branches_global, np.array([]),
        b'ch_size', branches_ch_num, branches_ch_cat
    )

    n = len(branches_global) + 1 + len(branches_ch_num) + len(branches_ch_cat)
    assert len(arrays) == n
    for a in arrays:
        assert isinstance(a, np.ndarray)
        assert len(a.shape) == 1
    start_ch = len(branches_global)
    length_ch = sum(arrays[start_ch])
    for a in arrays[:start_ch + 1]:
        assert len(a) == JETS_IN_FILE
    for a in arrays[start_ch + 1:]:
        assert len(a) == length_ch


def test_read_root_wrapper():
    path = tf.constant('data/shards/1.root')
    branches_global = ['pt', 'eta', 'mass', 'rho']
    branches_ch_num = ['ch_pt', 'ch_eta', 'ch_lost_hits']
    branches_ch_cat = ['ch_id']
    features = mljec.Features({
        'global': {
            'numerical': branches_global,
            'numerical_branches': branches_global + ['pt_gen']
        },
        'ch': {
            'numerical': branches_ch_num,
            'categorical': branches_ch_cat
        }
    })
    result = tf.function(
        lambda path: mljec.data._read_root_file_wrapper(path, features),
        input_signature=[tf.TensorSpec((), dtype=tf.string)]
    )(path)
    assert isinstance(result, collections.abc.Mapping)
    for branch in itertools.chain(
        ['pt_gen'], branches_global, branches_ch_num, branches_ch_cat
    ):
        assert branch in result
    for branch in itertools.chain(['pt_gen'], branches_global):
        shape = result[branch].shape
        assert len(shape) == 2 and shape[1] == 1
    for branch in branches_ch_num:
        column = result[branch]
        assert isinstance(column, tf.RaggedTensor)
        shape = column.shape
        assert len(shape) == 3 and shape[2] == 1
    for branch in branches_ch_cat:
        column = result[branch]
        assert isinstance(column, tf.RaggedTensor)
        shape = column.shape
        assert len(shape) == 2


def test_synthetic_features():
    features = mljec.Features({
        'global': {
            'numerical': ['pt'],
            'numerical_branches': ['pt', 'phi', 'pt_gen']
        },
        'ch': {
            'numerical': ['ch_dphi'],
            'numerical_branches': ['ch_phi']
        }
    })
    batch = {
        'pt': tf.constant([30., 100.]),
        'pt_gen': tf.constant([25., 110.]),
        'phi': tf.constant([0.1, 3.1]),
        'ch_phi': tf.ragged.constant([
            [0.2, -0.1, -3.0], [3.0, -3.1]
        ])
    }
    for name, values in batch.items():
        batch[name] = tf.expand_dims(values, axis=-1)
    processed_batch = mljec.data._create_synthetic_features(batch, features)
    assert set(processed_batch.keys()) == {'pt', 'ch_dphi', 'target'}
    assert processed_batch['target'].shape.as_list() == [2, 1]
    assert processed_batch['ch_dphi'].shape.as_list() == [2, None, 1]
    ref_dphi = np.expand_dims(
        [0.1, -0.2, -3.1, -0.1, 2 * np.pi - 6.2], axis=-1
    )
    assert np.all(np.isclose(
        processed_batch['ch_dphi'].flat_values.numpy(), ref_dphi
    ))


def test_dataset():
    input_files = [
        os.path.join('data', 'shards', f'{i + 1}.root')
        for i in range(5)
    ]
    branches_global = ['pt', 'eta', 'mass', 'rho']
    branches_ch_num = ['ch_pt', 'ch_eta', 'ch_lost_hits']
    branches_ch_cat = ['ch_id']
    features = mljec.Features({
        'global': {
            'numerical': branches_global,
            'numerical_branches': branches_global + ['pt_gen']
        },
        'ch': {
            'numerical': branches_ch_num + ['ch_deta'],
            'numerical_branches': branches_ch_num,
            'categorical': branches_ch_cat
        }
    })
    batch_size = 5
    dataset = mljec.data._build_dataset(
        input_files, features, {}, batch_size=batch_size
    )
    batch = next(iter(dataset))

    assert len(batch) == 2
    inputs = batch[0]
    target = batch[1]
    assert isinstance(inputs, collections.abc.Mapping)

    assert 'global_numerical' in inputs
    shape = inputs['global_numerical'].shape
    assert (
        len(shape) == 2 and shape[0] == batch_size
        and shape[1] == len(branches_global)
    )
    assert inputs['global_numerical'].dtype == tf.float32

    assert 'ch_numerical' in inputs
    shape = inputs['ch_numerical'].shape
    assert (
        len(shape) == 3 and shape[0] == batch_size and shape[1] == None
        and shape[2] == len(branches_ch_num) + 1
    )
    assert inputs['ch_numerical'].dtype == tf.float32

    for branch in branches_ch_cat:
        assert branch in inputs
        column = inputs[branch]
        shape = column.shape
        assert (
            len(shape) == 2 and shape[0] == batch_size and shape[1] == None
        )
        assert column.dtype == tf.int32

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
                'numerical': ['pt', 'eta', 'mass', 'rho'],
                'numerical_branches': ['pt', 'eta', 'mass', 'rho', 'pt_gen']
            },
            'ch': {
                'numerical': ['ch_pt', 'ch_eta', 'ch_lost_hits', 'ch_deta'],
                'numerical_branches': ['ch_pt', 'ch_eta', 'ch_lost_hits'],
                'categorical': ['ch_id']
            }
        }
    }
    metadata, train_ds, val_ds, test_ds = mljec.data.build_datasets(config)
    assert (
        'counts' in metadata and 'features' in metadata
        and 'cardinalities' in metadata
    )
    for ds in [train_ds, val_ds, test_ds]:
        assert isinstance(ds, tf.data.Dataset)
        assert len(ds.element_spec) == 2
