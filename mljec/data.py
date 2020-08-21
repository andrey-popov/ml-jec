import copy
import itertools
import os
from typing import (
    Callable, Dict, Mapping, Iterable, List, Sequence, Tuple, Union
)

import numpy as np
import tensorflow as tf
import uproot
import yaml


MaybeRaggedTensor = Union[tf.Tensor, tf.RaggedTensor]


def build_datasets(
    config: Mapping
) -> Tuple[Dict, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build training, validation, and test datasets.

    Args:
        config:  Dictionary representing configuration file.

    Return:
        Metadata and the three datasets.  The metadata include  numbers
        of examples in each set, a dictionary with names of input
        features (as read from the configuration), and a dictionary with
        cardinalities of categorical features.
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
    transforms, cardinalities = _create_transforms(
        os.path.join(data_config['location'], 'transform.yaml')
    )

    splits = _find_splits(data_config['split'], len(data_files))
    metadata = {}
    metadata['counts'] = {}
    metadata['features'] = features
    metadata['cardinalities'] = cardinalities

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
    transforms: Mapping[str, Callable[[MaybeRaggedTensor], MaybeRaggedTensor]],
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


def _create_transform_op_numerical(
    transform_config: Sequence[Mapping]
) -> Callable[[MaybeRaggedTensor], MaybeRaggedTensor]:
    """Construct preprocessing operation for one numerical feature.

    Args:
        transform_config:  Sequence of preprocessing steps.  Each step
            is described by a mapping with keys "type" and "params"
            (optional), which specify the type of the transformation and
            parameters of the transformation.

    Return:
        Callable that applies the transformation.
    """

    step_ops = []
    for step in transform_config:
        transform_type = step['type']
        if transform_type == 'abs':
            step_ops.append(tf.abs)
        elif transform_type == 'arcsinh':
            step_ops.append(
                lambda x: tf.math.asinh(x / step['params']['scale'])
            )
        elif transform_type == 'log':
            step_ops.append(tf.math.log)
        elif transform_type == 'linear':
            step_ops.append(
                lambda x: (x - step['params']['loc']) / step['params']['scale']
            )
        else:
            raise RuntimeError(
                f'Unknown transformation type "{transform_type}".'
            )

    def op(x):
        for step_op in step_ops:
            x = step_op(x)
        return x

    return op


def _create_transform_op_categorical(
    allowed_values: Sequence[int]
) -> Callable[[MaybeRaggedTensor], MaybeRaggedTensor]:
    """Construct preprocessing operation for one categorical feature.

    The operation maps unique values of the feature to consecutive
    integers starting from 0.

    Args:
        allowed_values:  Allowed values for the feature.  Their order is
            respected in the mapping.

    Return:
        Callable that applies the transformation.
    """

    initializer = tf.lookup.KeyValueTensorInitializer(
        allowed_values, tf.range(len(allowed_values)), key_dtype=tf.int32
    )
    mapping = tf.lookup.StaticHashTable(initializer, -1)

    def op(x):
        # StaticHashTable doesn't accept RaggedTensor directly
        return tf.ragged.map_flat_values(mapping.lookup, x)

    return op


def _create_transforms(
    path: str
) -> Tuple[
    Dict[str, Callable[[MaybeRaggedTensor], MaybeRaggedTensor]],
    Dict[str, int]
]:
    """Create preprocessing operations.

    Args:
        path:  Path to YAML configuration file that defines the
            transformations.

    Return:
        Dictionary with preprocessing operations for individual
        features and a dictionary with cardinalities of categorical
        features.
    """

    with tf.io.gfile.GFile(path) as f:
        transform_defs = yaml.safe_load(f.read())
    transforms = {
        feature: _create_transform_op_numerical(transform_config)
        for feature, transform_config in transform_defs['numerical'].items()
    }
    cardinalities = {}
    for feature, allowed_values in transform_defs['categorical'].items():
        transforms[feature] = _create_transform_op_categorical(
            allowed_values
        )
        cardinalities[feature] = len(allowed_values)
    return transforms, cardinalities


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
    batch: Mapping[str, MaybeRaggedTensor], features: Mapping,
    transforms: Mapping[
        str, Callable[[MaybeRaggedTensor], MaybeRaggedTensor]] = {}
) -> Tuple[Dict[str, MaybeRaggedTensor], tf.Tensor]:
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

    inputs = {}
    constituent_types = list(features.keys())
    constituent_types.remove('global')

    # Concatenate numerical features into blocks
    block = tf.concat(
        [batch[name] for name in features['global']['numerical']],
        axis=1, name='global_numerical'
    )
    inputs['global_numerical'] = block
    for constituent_type in constituent_types:
        block = tf.concat(
            [batch[name] for name in features[constituent_type]['numerical']],
            axis=2, name=f'{constituent_type}_numerical'
        )
        inputs[f'{constituent_type}_numerical'] = block

    # Propagate categorical features directly
    for constituent_type in constituent_types:
        for name in features[constituent_type]['categorical']:
            inputs[name] = batch[name]

    return (inputs, target)


def _read_root_file(
    path: bytes, branches_global_numerical: np.ndarray,
    *constituents_args
) -> List[np.ndarray]:
    """Read a single ROOT file.

    Args:
        path:  Path the file.
        branches_global_numerical:  Names of branches with global
            numerical features, represented with bytes.
        constituents_args:  Names of branches for an arbitrary number of
            types of constituents.  For each type, three arguments must
            be provided, in this order:
            - name of the branch with the number of constituents of that
              type in each jet, represented with bytes,
            - NumPy array with names of branches with numerical
              features for that type of constituents,
            - NumPy array with names of branches with categorical
              features for that type of constituents.

    Return:
        Rank 1 NumPy arrays for all specified branches.  The order
        matches the order of branches in the arguments.  The arrays for
        constituents are flattened.
    """

    assert len(constituents_args) % 3 == 0
    constituents_blocks = [
        constituents_args[i:i + 3]
        for i in range(0, len(constituents_args), 3)
    ]

    input_file = uproot.open(path.decode())
    tree = input_file['Jets']
    branches_to_read = []
    branches_to_read += branches_global_numerical.tolist()
    for block in constituents_blocks:
        branches_to_read.append(block[0])
        branches_to_read.extend(itertools.chain(block[1], block[2]))
    data = tree.arrays(branches=branches_to_read)

    results = []
    for branch in branches_global_numerical:
        results.append(data[branch].astype(np.float32))

    for block in constituents_blocks:
        results.append(data[block[0]].astype(np.int32))
        for branch in block[1]:
            results.append(data[branch].astype(np.float32).flatten())
        for branch in block[2]:
            results.append(data[branch].astype(np.int32).flatten())
    return results


def _read_root_file_wrapper(
    path: str, features: Mapping
) -> Dict[str, MaybeRaggedTensor]:
    """Wrapper around _read_root_file.

    Specify what branches to read, call _read_root_file, and convert the
    output to a dictionary.

    Args:
        path:  Tensor representing path to the file to read.
        features:  Dictionary with names of input features.

    Return:
        Dictionary that maps branch names to the corresponding tensors.
        Shapes of output tensors (parentheses denote ragged dimensions):
        - global numeric: (BATCH, 1),
        - per-constituent numeric: (BATCH, (None), 1),
        - per-constituent categorical: (BATCH, (None)).
    """

    inputs = [path]
    output_types = []
    output_names = []
    all_numerical = []

    # Global features.  In addition to those specified in the mapping
    # given to this function, include pt_gen, which is needed to
    # compute the regression target.
    branches_num = ['pt_gen'] + features['global']['numerical']
    inputs.append(branches_num)
    output_types += [tf.float32] * len(branches_num)
    output_names += branches_num
    all_numerical += branches_num

    # Features of constituents
    block_names = list(features.keys())
    block_names.remove('global')
    for block in block_names:
        branches_num = features[block]['numerical']
        branches_cat = features[block]['categorical']
        inputs.extend([f'{block}_size', branches_num, branches_cat])
        output_types.extend(
            [tf.int32]
            + [tf.float32] * len(branches_num)
            + [tf.int32] * len(branches_cat)
        )
        output_names += [f'{block}_size'] + branches_num + branches_cat
        all_numerical += branches_num

    data = tf.numpy_function(
        func=_read_root_file, inp=inputs, Tout=output_types,
        name='read_root'
    )
    data = {k: v for k, v in zip(output_names, data)}
    for values in data.values():
        values.set_shape((None,))

    for block in block_names:
        size = data.pop(f'{block}_size')
        for branch in itertools.chain(
            features[block]['numerical'], features[block]['categorical']
        ):
            data[branch] = tf.RaggedTensor.from_row_lengths(
                values=data[branch], row_lengths=size
            )

    for branch in all_numerical:
        data[branch] = tf.expand_dims(data[branch], axis=-1)
    return data
