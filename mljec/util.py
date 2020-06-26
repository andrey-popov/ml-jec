from matplotlib import pyplot as plt
import numpy as np


def plot_history(config, history, save_path, ylim=(None, None), zoom=False):
    """Plot training history.

    Plot the loss on the training and validation sets and mark points
    when the learning rate was adjusted.

    Args:
        config:  Directory representing configuration file.
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
    for x in lr_changes:
        axes.axvline(x, ls='dashed', lw=0.8, c='gray')
    axes.set_ylim(ylim)
    axes.set_xlabel('Parameter updates')
    axes.set_ylabel('Loss')
    axes.legend()

    fig.savefig(save_path)
    plt.close(fig)

