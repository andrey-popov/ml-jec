import copy
from typing import List, Mapping, Sequence, Set, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np


class Features:
    """Structured collection of names of features and branches.

    Collections of branches specify what should be read from the input
    data files, while features are the inputs for the neural network.
    Some features may be constructed on the fly, and some branches may
    only be used to construct other features but not as inputs to the
    neural network directly.  Because of this, the two collections are
    not necessarily identical.

    Attributes:
        constituent_types:  Set with names of types of jet constituents.
    """

    def __init__(self, config: Mapping[str, Mapping[str, List[str]]]):
        """Initialize from configuration.

        The configuration provided in the argument includes one or more
        blocks.  Block "global", which contains jet- and event-level
        features, is mandatory.  Other blocks describe different types
        of jet constituents.  Within each block, input features are
        listed in fields "numerical" and "categorical", depending on
        their types.  A missing field is interpreted as an empty list.
        Names of branches within each block are listed in fields
        "numerical_branches" and "categorical_branches".  If not given,
        assumed to be the same as the corresponding feature names.
        """

        if 'global' not in config:
            raise RuntimeError('Mandatory block "global" is missing.')
        self._features = {}
        self._branches = {}
        for block_name, block in config.items():
            features = {}
            branches = {}
            features['numerical'] = block.get('numerical', [])
            branches['numerical'] = block.get(
                'numerical_branches', features['numerical']
            )
            features['categorical'] = block.get('categorical', [])
            branches['categorical'] = block.get(
                'categorical_branches', features['categorical']
            )
            self._features[block_name] = features
            self._branches[block_name] = branches
            for key in block:
                if key not in {
                    'numerical', 'numerical_branches',
                    'categorical', 'categorical_branches'
                }:
                    raise RuntimeError(f'Unexpected key "{key}".')

        self.constituent_types = set(self._features.keys())
        self.constituent_types.remove('global')

    def all(self) -> Set[str]:
        """Return names of all features."""

        features = set()
        for block in self._features.values():
            for names in block.values():
                features.update(names)
        return features

    def categorical(self, block: str, branches: bool = False) -> List[str]:
        """Return names of categorical features or branches."""

        if branches:
            return self._branches[block]['categorical']
        else:
            return self._features[block]['categorical']

    def numerical(self, block: str, branches: bool = False) -> List[str]:
        """Return names of numerical features or branches."""

        if branches:
            return self._branches[block]['numerical']
        else:
            return self._features[block]['numerical']


def plot_history(
    config: Mapping, history: Mapping[str, Sequence[float]],
    save_path: str,
    ylim: Tuple[Union[float, None], Union[float, None]] = (None, None),
    zoom: bool = False
) -> None:
    """Plot training history.

    Plot the loss on the training and validation sets and mark points
    when the learning rate was adjusted.

    Args:
        config:  Master configuration.
        history:  Dictionary with training history.
        save_path:  Save path for produced figure.
        ylim:  Manual range for y axis.  If not given, an automatic
            range will be used.
        zoom:  If True, will skip the initial segment of training when
            computing the range for the y axis.  Cannot be used together
            with ylim.
    """

    freq = config['train']['steps_per_epoch']
    if ylim[0] is None and ylim[1] is None and zoom:
        # Find the range of values in the last 75% of epochs
        skip = round(0.25 * len(history['loss']))
        maximum = max(
            max(history['loss'][skip:]), max(history['val_loss'][skip:])
        )
        minimum = min(
            min(history['loss'][skip:]), min(history['val_loss'][skip:])
        )
        # Include a 5% margin on each side
        margin = 0.05 / (1 - 2 * 0.05) * (maximum - minimum)
        ylim = (minimum - margin, maximum + margin)

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_alpha(0)
    axes = fig.add_subplot(111)
    x = np.arange(1, len(history['loss']) + 1) * freq
    axes.plot(x, history['loss'], label='Training')
    axes.plot(x, history['val_loss'], label='Validation')
    lr_changes = [
        (i + 0.5) * freq for i in range(len(history['lr']) - 1)
        if history['lr'][i] != history['lr'][i + 1]
    ]
    if len(lr_changes) < 0.5 * len(x):
        for x in lr_changes:
            axes.axvline(x, ls='dashed', lw=0.8, c='gray')
    axes.set_ylim(ylim)
    axes.set_xlabel('Parameter updates')
    axes.set_ylabel('Loss')
    axes.legend()

    fig.savefig(save_path)
    plt.close(fig)
