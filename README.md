# Jet calibration with DNN

This repository contains code for calibration of the transverse momentum p<sub>T</sub> of [jets](https://en.wikipedia.org/wiki/Jet_(particle_physics)) produced in proton collisions in the [CMS experiment](https://en.wikipedia.org/wiki/Compact_Muon_Solenoid) at the Large Hadron Collider (LHC). The calibration is performed with deep neural networks, which are trained to predict the true value of p<sub>T</sub> starting from properties of a reconstructed jet. The goal is to improve the p<sub>T</sub> resolution and the dependence on the jet flavour compared to the standard calibration procedure.

Technically, the DNN is trained to predict the logarithm of the ratio between the true p<sub>T</sub> known from simulation and the uncalibrated p<sub>T</sub> of the corresponding reconstructed jet. It uses as inputs event- and jet-level features as well as properties of individual jet constituents. The snapshot of an old version of the code that used a simple model that did not exploit jet constituents, is available in [this tag](https://github.com/andrey-popov/ml-jec/tree/tabular).


## Dataset

The input dataset, containing some 10<sup>7</sup> examples, is produced using code from a [dedicated repository](https://gitlab.cern.ch/aapopov/ml-jec-vars). This includes a non-uniform downsampling from an initial set of 10<sup>10</sup> examples, which prunes overpopulated regions of the parameter space while preserving less populated but physically more relevant ones.

The dataset consists of files in the native [ROOT format](https://root.cern.ch), which are read with the help of [`uproot`](https://github.com/scikit-hep/uproot). Several example files with just a handful of jets are provided for tests in directory `tests/data`. It is assumed that the data files are placed in a directory (local or in a Google Cloud Storage bucket) with the following structure:
```
data
├── data.yaml
├── transform.yaml
└── shards
    ├── 001.root
    ├── 002.root
    └── ...
```
The file `data.yaml` lists the ROOT files and also specifies the number of jets in each of them (typically, 10<sup>5</sup>). Consult the [example](tests/data/data.yaml) to see its structure.

The file `transform.yaml` above defines preprocessing transformations to be applied to individual features in the dataset before the start of the training. This file is created by the script `build_transform.py`. Non-linear transformations are specified manually. They are normally followed by a linear rescaling whose parameters are determined by the script.

Features are not only read from the ROOT files but also constructed on the fly. Examples of this are relative p<sub>T</sub>, &Delta;&eta;, and &Delta;&phi; of jet constituents.


## Model

The DNN consists of multiple blocks. A jet can contain a variable number of constituents, and their order is not relevant. To reflect this, each type of jet constituents is processed with a block based on [Deep Sets](http://arxiv.org/abs/1703.06114). This approach has already been used to classify jets in [Particle Flow Networks](http://arxiv.org/abs/1810.05165). Within each block, an MLP with shared weights is applied to every jet constituent of given type, and the output of the MLP is then summed over all constituents of that type in a jet. The resulting outputs for different types of constituents, as well as jet-level features, are concatenated and processed by another MLP. This jet-level MLP has a single output unit, whose value gives (the logarithm of) the correction factor for the jet p<sub>T</sub>.

The input features to use and the parameters of the DNN are described in the master configuration file ([example](./example_config.yaml)). In section `model`, it specifies the dimensionality of the embeddings for categorical features, the numbers of units in each layer of each MLP block, and the type of the MLP blocks. The supported types are vanilla MLP and ResNet.


## Environment and tests

To use this package, its location should be added to `PYTHONPATH`. This can be done by executing
```sh
. ./env.sh
```
The version of Python used is 3.7. Python dependencies are listed in [`requirements.txt`](requirements.txt). To read files from Google Cloud Platform buckets, the `gsutil` program should also be installed and configured.

Run tests with
```sh
cd tests
pytest
```
The full training pipeline can be tried with
```sh
train.py test_config.yaml -o test_output
```
(from the same directory). Note that since the test files are tiny, the results of this training are not meaningful and the runtime is completely dominated by various overheads.


## Training

The training is done with the script `train.py`, as in the example above. Its behaviour is controlled by master configuration file provided as an argument. Script `steer.py` will perform the training for multiple configuration files consecutively and copy logs and all outputs produced in each task to a given destination (normally, a Google Cloud Storage bucket).
