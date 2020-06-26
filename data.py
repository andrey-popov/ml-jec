import copy
import os

import tensorflow as tf
import yaml


def build_datasets(config):
    """Build training, validation, and test datasets.

    Args:
        config:  Dictionary representing configuration file.

    Return:
        Metadata and the three datasets.  The metadata are represented
        by a dictionary.  It contains numbers of examples in each set
        and the list of input features.
    """

    config = config['data']

    data_dir = os.path.join(config['location'], 'data')
    data_files = [
        os.path.join(data_dir, filename)
        for filename in sorted(tf.io.gfile.listdir(data_dir))
    ]

    with tf.io.gfile.GFile(
        os.path.join(config['location'], 'transform.yaml')
    ) as f:
        feature_infos = yaml.safe_load(f.read())
    example_schema = {
        info['feature']: tf.io.FixedLenFeature([1], tf.float32)
        for info in feature_infos
    }
    example_schema['target'] = tf.io.FixedLenFeature([1], tf.float32)
    transforms = [
        (info['feature'], _create_transform_op(info))
        for info in feature_infos
    ]

    splits = _find_splits(config['split'], len(data_files))
    metadata = {}
    metadata['counts'] = _count_examples(
        os.path.join(config['location'], 'counts.yaml'), data_files, splits
    )
    metadata['features'] = [info['feature'] for info in feature_infos]

    datasets = {}
    for set_label, file_range in splits.items():
        if set_label == 'train':
            repeat = True
            batch_size = config['batch_size']
            map_num_parallel = tf.data.experimental.AUTOTUNE
        else:
            repeat = False
            batch_size = 32768
            map_num_parallel = None

        datasets[set_label] = _build_dataset(
            data_files[file_range[0]:file_range[1]],
            example_schema, transforms,
            repeat=repeat, batch_size=batch_size,
            map_num_parallel=map_num_parallel
        )
    return metadata, datasets['train'], datasets['val'], datasets['test']


def _build_dataset(
    files, example_schema, transforms, repeat=False, batch_size=128,
    map_num_parallel=None
):
    """Build a dataset.

    Args:
        files:  Paths to files included in the dataset.
        example_schema:  Schema to parse profobuf messages.
        transforms:  List of pairs of names of features and
            preprocessing operations to be applied to them.
        repeat:  Whether the dataset should be repeated.
        batch_size:  Batch size.
        map_num_parallel:  Number of parallel calls for Dataset.map.
            Directly forwarded to that method.

    Return:
        TFRecordDataset.
    """

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda e: _deserialize_example(e, example_schema, transforms),
        num_parallel_calls=map_num_parallel
    )
    return dataset


def _count_examples(counts_path, data_files, splits):
    """Count examples in training, validation, and test sets.

    Args:
        counts_path:  Path to YAML file with numbers of examples in each
            data file.
        data_files:  Paths to all data files.
        splits:  Splitting of data files into the three sets.  Same
            format as the return value of _find_splits.

    Return:
        Mapping from labels of the three sets to numbers of examples.
    """

    with tf.io.gfile.GFile(counts_path) as f:
        per_file_counts = yaml.safe_load(f.read())
    counts = {}
    for set_label, file_range in splits.items():
        n = sum(
            per_file_counts[os.path.split(path)[1]]
            for path in data_files[file_range[0]:file_range[1]]
        )
        counts[set_label] = n
    return counts


def _create_transform_op(feature_info):
    """Construct preprocessing operation for one feature."""

    loc = feature_info['loc']
    scale = feature_info['scale']
    if 'arcsinh' in feature_info:
        arcsinh_scale = feature_info['arcsinh']['scale']
        def op(x):
            return (tf.math.asinh(x / arcsinh_scale) - loc) / scale
    else:
        def op(x):
            return (x - loc) / scale
    return op


def _deserialize_example(example, schema, transforms):
    """Deserialize protobuf messages and apply preprocessing."""

    decoded_batch = tf.io.parse_example(example, schema)
    features = []
    for column, transform in transforms:
        features.append(transform(decoded_batch[column]))
    return tf.concat(features, axis=1), decoded_batch['target']


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

