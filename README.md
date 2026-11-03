# Code for TADRED: TAsk-DRiven Experimental Design in imaging

## This is a fork
This repository is a modified version of [TADRED](https://github.com/sbb-gh/tadred), originally developed by Stefano Blumberg and collaborators. Modifications are identified in the relevant source files. 

## Contact 
[slatorp@cardiff.ac.uk](mailto:slatorp@cardiff.ac.uk).

## Overview
TADRED identifies the most informative channel-subset whilst simultaneously training a network to execute the task given the subset.

TADRED is a novel method for TAsk-DRiven experimental design in imaging.  TADRED couples feature scoring and task execution in consecutive networks.  The scoring and subsampling procedure enables efficient identification of subsets of complementarily informative channels jointly with training a high-performing network for the task.  TADRED also gradually reduces the full set of samples stepwise to obtain the subsamples, which improves optimization.


## Citation

If you use TADRED in your work please cite the accompanying paper: 

1. Blumberg, S., Slator, P.J., & Alexander, D.C. (2024). **Experimental Design for Multi-Channel Imaging via Task-Driven Feature Selection.** *International Conference on Learning Representations (ICLR 2024).* [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aec2dfc4f5e19acf05c15587c889dbc4-Abstract-Conference.html), [code](https://github.com/sbb-gh/tadred)

<!--## Contact

stefano.blumberg.17@ucl.ac.uk-->

## Installation 

To improve robustness and avoid unnecessary dependencies, the relevant pyproject.toml files specify flexible dependency constraints rather than pinning exact versions. The code has been fully tested with PyTorch 2.15. We  use pyenv for python version management and recommend **Python 3.14**. 

###Installation Part 1: Directory set up

First create and move to the desired installation directory.

```bash 
mkdir <INSTALL_DIR>
cd <INSTALL_DIR>
```

###Installation Part 2: Environment

First create an environment and enter it.  We provide two examples either using Pyenv or Conda:

#### Option 1 - [Pyenv](https://github.com/pyenv)

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Option 2 - [Conda](https://docs.conda.io/en/latest/)

```bash
conda create -n tadred 
conda activate tadred
```

#### Installation Part 2: Packages and Code

Code requires: pytorch, numpy, pyyaml, hydra.

Code is tested using PyTorch v2.0.0, cuda 11.7 on the GPU.

We provide examples of installing packages, using pip,

### Python Package from Source

```bash
pip install git+https://github.com/sbb-gh/tadred.git@main
```

### Using pip

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 # Install PyTorch 2.0
pip install pyyaml hydra-core==1.3 # Install PyYAML and Hydra
```
Then clone this repository.

## Arguments/Options and Running from the Command Line

Please see config in [types.py](./tadred_code/types.py) for base arguments and descriptions.

To run from the command line:

```bash
python train_and_eval.py --cfg <YAML_CONFIG_PATH>
```

where <YAML_CONFIG_PATH> is a path to a config file.

## License

This project is licensed under the terms of the Apache 2.0 license. For more details, see the LICENSE file in the root of this repository.

## Acknowledgments

The original TADRED code is [here](https://github.com/sbb-gh/tadred)
