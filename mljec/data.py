import copy
import os

import numpy as np
import tensorflow as tf
import uproot
import yaml


def build_datasets(config):
    """Build training, validation, and test datasets.

    Args:
        config:  Dictionary representing configuration file.

    Return:
        Metadata and the three datasets.  The metadata are represented
        by a dictionary.  It contains numbers of examples in each set
        and a dictionary of input features.
    """

    data_config = config['data']

    with tf.io.gfile.GFile(
        os.path.join(data_config['location'], 'data.yaml')
    ) as f:
        data_file_infos = yaml.safe_load(f.read())
    data_files = [
        os.path.join(data_config['location'], c['path'])
        for c in data_file_infos
    ]

    features = config['features']
    with tf.io.gfile.GFile(
        os.path.join(data_config['location'], 'transform.yaml')
    ) as f:
        transforms = yaml.safe_load(f.read())
    transforms = {
        feature: _create_transform_op(transform_config)
        for feature, transform_config in transforms.items()
    }

    splits = _find_splits(data_config['split'], len(data_files))
    metadata = {}
    metadata['counts'] = {}
    metadata['features'] = features

    datasets = {}
    for set_label, file_range in splits.items():
        if set_label == 'train':
            repeat = True
            batch_size = data_config['batch_size']
            map_num_parallel = tf.data.experimental.AUTOTUNE
        else:
            repeat = False
            batch_size = 32768
            map_num_parallel = None

        metadata['counts'][set_label] = sum(
            c['count'] for c in data_file_infos[file_range[0]:file_range[1]]
        )
        datasets[set_label] = _build_dataset(
            data_files[file_range[0]:file_range[1]], features, transforms,
            repeat=repeat, batch_size=batch_size,
            map_num_parallel=map_num_parallel
        )
    return metadata, datasets['train'], datasets['val'], datasets['test']


def _build_dataset(
    paths, features, transforms, repeat=False, batch_size=128,
    map_num_parallel=None
):
    """Build a dataset.

    Args:
        paths:  Paths to files included in the dataset.
        features:  Dictionary with names of input features.
        transforms:  List of pairs of names of features and
            preprocessing operations to be applied to them.
        repeat:  Whether the dataset should be repeated.
        batch_size:  Batch size.
        map_num_parallel:  Number of parallel calls for Dataset.map.
            Directly forwarded to that method.

    Return:
        TensorFlow Dataset.
    """

    # Read input ROOT files one at a time (with prefetching)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(
        lambda path: _read_root_file_wrapper(path, features)
    )
    dataset = dataset.prefetch(1)

    dataset = dataset.map(
        lambda batch: _preprocess(batch, features, transforms),
        num_parallel_calls=map_num_parallel
    )

    dataset = dataset.unbatch()
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def _create_transform_op(transform_config):
    """Construct preprocessing operation for one feature."""

    loc = transform_config['loc']
    scale = transform_config['scale']
    if 'arcsinh' in transform_config:
        arcsinh_scale = transform_config['arcsinh']['scale']
        def op(x):
            return (tf.math.asinh(x / arcsinh_scale) - loc) / scale
    else:
        def op(x):
            return (x - loc) / scale
    return op


def _find_splits(nums, num_total):
    """Find ranges of indices of input files for the three sets.

    Args:
        nums:  Sequence with numbers of files in the training,
            validation, and test sets.  At maximum one element can be
            -1, which deduced from the total number of files.
        num_total:  Total number of files.

    Return:
        Mapping from labels of the tree sets to ranges of indices of
        files in each set.  Left boundary is included, right is not.
    """

    assert len(nums) == 3
    num_placeholders = nums.count(-1)
    assert num_placeholders <= 1
    if num_placeholders == 1:
        nums = copy.copy(nums)
        nums[nums.index(-1)] = num_total - sum(nums) - 1

    splits = {}
    i = 0
    for key, n in zip(['train', 'val', 'test'], nums):
        splits[key] = (i, i + n)
        i += n
    return splits


def _preprocess(batch, features, transforms={}):
    """Perform preprocessing for a batch.

    Args:
        batch:  Dictionary of tensors representing individual features
            in the current batch.
        features:  Dictionary with names of input features.
        transforms:  Dictionary of transformation operations.

    Return:
        Tuple of preprocessed input features and target.  The input
        features are represented with a dictionary of tensors.
    """

    # Target
    target = tf.math.log(batch['pt_gen'] / batch['pt'])
    batch.pop('pt_gen')

    # Apply preprocessing
    for feature_name, column in batch.items():
        if feature_name in transforms:
            transform = transforms[feature_name]
            batch[feature_name] = transform(column)

    # Concatenate global features in a single dense block
    global_features = [
        batch[name]
        for name in features['global']['numeric']
    ]
    global_features_block = tf.concat(global_features, axis=1)

    return {'global_numeric': global_features_block}, target


def _read_root_file(path, branches):
    """Read a single ROOT file.

    Args:
        path:  Path the file represented as bytes.
        branches:  NumPy array with branches to be read.  The branches
            are represented as bytes.

    Return:
        List of NumPy arrays for each specified branch.  The order
        matches the order of branches.
    """

    input_file = uproot.open(path.decode())
    tree = input_file['Jets']
    data = tree.arrays(branches=branches)
    return [
        np.expand_dims(data[name].astype(np.float32), axis=1)
        for name in branches
    ]


def _read_root_file_wrapper(path, features):
    """Wrapper around _read_root_file.

    Specify branches to read and convert the output to a dictionary.

    Args:
        path:  Tensor representing path to the file to read.
        features:  Dictionary with names of input features.

    Return:
        Dictionary that maps branch names to the corresponding tensors.
    """

    # Construct the list of branches to read.  In addition to input
    # features, include pt_gen, which is needed to compute the target.
    branches = ['pt_gen']
    branches += features['global']['numeric']

    data = tf.numpy_function(
        func=_read_root_file,
        inp=[path, branches], Tout=[tf.float32] * len(branches),
        name='read_root'
    )
    for column in data:
        column.set_shape((None, 1))
    return {k: v for k, v in zip(branches, data)}
