# Jet calibration with DNN

This repository contains code for calibration of the magnitude of the transverse momentum p<sub>T</sub> of [jets](https://en.wikipedia.org/wiki/Jet_(particle_physics)) produced in particle collisions in the [CMS experiment](https://en.wikipedia.org/wiki/Compact_Muon_Solenoid) at the Large Hadron Collider (LHC). The calibration is performed with deep neural networks, which are trained to predict the true value of p<sub>T</sub> starting from properties of a reconstructed jet. The goal is to improve the p<sub>T</sub> resolution and the dependence on the jet flavour compared to the standard calibration procedure.

Technically, the DNN is trained to predict the logarithm of the ratio between the true p<sub>T</sub> known from simulation and the uncalibrated p<sub>T</sub> of the corresponding reconstructed jet. In the current implementation a simple MLP architecture is used, and around 70 properties of a jet are fed as inputs.

**Instructions below refer to tag `tabular`.**

## Dataset

The input dataset, containing some 10<sup>7</sup> examples, is produced using code from a [dedicated repository](https://gitlab.cern.ch/aapopov/ml-jec-vars). This includes a non-uniform downsampling from an initial set of 10<sup>10</sup> examples, which prunes overpopulated regions of the parameter space while preserving less populated but physically more relevant ones.

Script `to_protobuf.py` converts files with examples from the [ROOT format](https://root.cern.ch) to protobuf messages used by TensorFlow. It also creates a YAML file containing the number of examples written to each protobuf file.

Script `build_transform.py` finds parameters of preprocessing transformations, which will be applied at the start of the training. Non-linear transformations chosen manually are applied to some features, followed by a linear rescaling applied to all features. Parameters of those transformations are saved to a YAML file, which also defines the reference ordering of features.

In the following it is assumed that files defining the dataset are placed in a directory (local or in a Google Cloud Storage bucket) with the following structure:
```
├── data
│   ├── 001.tfrecord.gz
│   ├── 002.tfrecord.gz
│   └── ...
├── counts.yaml
└── transform.yaml
```
The two YAML files are produced by the scripts above.


## Training

The training is done with script `train.py`. Its behaviour is controlled via a configuration file ([example](./config_example.yaml)). Script `steer.py` will perform the training for multiple configuration files consecutively and copy logs and all outputs produced in each task to a given destination (normally, a Google Cloud Storage bucket).
