import copy
import os
from typing import (
    Callable, Dict, Mapping, Iterable, List, Sequence, Tuple, Union
)

import numpy as np
import tensorflow as tf
import uproot
import yaml


def build_datasets(
    config: Mapping
) -> Tuple[Dict, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build training, validation, and test datasets.

    Args:
        config:  Dictionary representing configuration file.

    Return:
        Metadata and the three datasets.  The metadata include  numbers
        of examples in each set and a dictionary of input features (as
        read from the configuration).
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
    paths: Iterable, features: Mapping,
    transforms: Mapping[str, Callable[[tf.Tensor], tf.Tensor]],
    repeat: bool = False, batch_size: int = 128,
    map_num_parallel: Union[int, None] = None
) -> tf.data.Dataset:
    """Build a dataset.

    Args:
        paths:  Paths to files included in the dataset.
        features:  Dictionary with names of input features.
        transforms:  Preprocessing operations to be applied to
            individual features.
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


def _create_mask(lengths: np.ndarray, max_length: int) -> np.ndarray:
    """Create 2D mask that selects given numbers of elements.

    The mask is a 2D array of boolean values.  Each row starts with the
    number of True elements as given by an input array; they are
    followed by False elements to pad the row.

    Args:
        lengths:  1D array the number of elements to be selected on
            each row.
        max_length:  The length of the rows.

    Return:
        Boolean mask for shape (len(lengths), max_length).
    """

    indices = np.repeat(
        np.arange(max_length)[np.newaxis, :],
        len(lengths), axis=0
    )
    return indices < lengths[:, np.newaxis]


def _create_transform_op(
    transform_config: Sequence[Mapping]
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Construct preprocessing operation for one feature.

    Args:
        transform_config:  Sequence of preprocessing steps.  Each step
            is described by a mapping with keys "type" and "params"
            (optional), which specify the type of the transformation and
            parameters of the transformation.

    Return:
        Callable that applies the transformation.
    """

    def op(x):
        for step in transform_config:
            transform_type = step['type']
            if transform_type == 'abs':
                x = tf.abs(x)
            elif transform_type == 'arcsinh':
                x = tf.math.asinh(x / step['params']['scale'])
            elif transform_type == 'log':
                x = tf.math.log(x)
            elif transform_type == 'linear':
                x = (x - step['params']['loc']) / step['params']['scale']
            else:
                raise RuntimeError(
                    f'Unknown transformation type "{transform_type}".'
                )
        return x

    return op


def _find_splits(
    nums: Sequence[int], num_total: int
) -> Dict[str, Tuple[int, int]]:
    """Find ranges of indices of input files for the three sets.

    Args:
        nums:  Numbers of files in the training, validation, and test
            sets.  At maximum one element can be -1, which deduced from
            the total number of files.
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


def _preprocess(
    batch: Mapping[str, tf.Tensor], features: Mapping,
    transforms: Mapping[str, Callable[[tf.Tensor], tf.Tensor]] = {}
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Perform preprocessing for a batch.

    Args:
        batch:  Tensors representing individual features in the
            current batch.
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

    # Concatenate numeric features into blocks
    global_numeric_block = tf.concat(
        [batch[name] for name in features['global']['numeric']],
        axis=1
    )
    ch_numeric_block = tf.concat(
        [
            tf.expand_dims(batch[name], axis=2)
            for name in features['ch']['numeric']
        ],
        axis=2
    )

    return (
        {
            'global_numeric': global_numeric_block,
            'ch_mask': batch['ch_mask'],
            'ch_numeric': ch_numeric_block
        },
        target
    )


def _read_root_file(
    path: bytes, branches_global_numeric: np.ndarray,
    max_size_ch: int, branches_ch_numeric: np.ndarray
) -> List[np.ndarray]:
    """Read a single ROOT file.

    Args:
        path:  Path the file.
        branches_global_numeric:  Names of branches with global numeric
            features, represented with bytes.
        max_size_ch:  Maximal number of charged constituents to
            consider.
        branches_ch_numeric:  Names of branches with numeric features of
            charged constituents, represented with bytes.

    Return:
        NumPy arrays for all specified branches.  The order matches
        the order of branches in the arguments.  The arrays for charged
        constituents are preceeded by a boolean mask that indicates the
        presence of the constituents.
    """

    input_file = uproot.open(path.decode())
    tree = input_file['Jets']
    branches_to_read = branches_global_numeric.tolist()
    branches_to_read.append(b'ch_size')
    branches_to_read += branches_ch_numeric.tolist()
    data = tree.arrays(branches=branches_to_read)

    results = []
    for branch in branches_global_numeric:
        results.append(np.expand_dims(data[branch].astype(np.float32), axis=1))
    results.append(_create_mask(data[b'ch_size'], max_size_ch))
    for branch in branches_ch_numeric:
        results.append(
            data[branch].pad(max_size_ch).regular().astype(np.float32)
        )
    return results


def _read_root_file_wrapper(
    path: str, features: Mapping
) -> Dict[str, tf.Tensor]:
    """Wrapper around _read_root_file.

    Specify what branches to read, call _read_root_file, and convert the
    output to a dictionary.

    Args:
        path:  Tensor representing path to the file to read.
        features:  Dictionary with names of input features.

    Return:
        Dictionary that maps branch names to the corresponding tensors.
        In addition to requested branches, this dictionary includes a
        mask indicating the presence of constituents.
    """

    inputs = [path]
    output_types = []
    output_names = []
    output_shapes = []

    # Global features.  In addition to those specified in the mapping
    # given to this function, include pt_gen, which is needed to
    # compute the regression target.
    branches = ['pt_gen']
    branches += features['global']['numeric']
    inputs.append(branches)
    output_types += [tf.float32] * len(branches)
    output_names += branches
    output_shapes += [(None, 1)] * len(branches)

    # Features related to charged constituents
    max_size = features['ch']['max_size']
    inputs.append(max_size)
    branches = features['ch']['numeric']
    inputs.append(branches)
    output_types += [tf.bool] + [tf.float32] * len(branches)
    output_names += ['ch_mask'] + branches
    output_shapes += [(None, max_size)] * (1 + len(branches))

    data = tf.numpy_function(
        func=_read_root_file, inp=inputs, Tout=output_types,
        name='read_root'
    )
    for column, shape in zip(data, output_shapes):
        column.set_shape(shape)
    return {k: v for k, v in zip(output_names, data)}
