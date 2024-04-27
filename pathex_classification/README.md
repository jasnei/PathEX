# Pathex Classification

This repository contains the code for the Pathex Classification project.

## Getting Started

To get started, follow the instructions in the [README.md](README.md) file.

## Prerequisites

To run the code, you will need the following:

- Python 3.9
- PyTorch 1.11
- timm
- albumentations
- pandas
- numpy


## Installing

To install the required packages, run the following command in the terminal:

```bash
conda env create -f requirements.yml
```

This will create a conda environment with the required packages.

## Running the training and testing scripts

To train the model, run the following command in the terminal:

```
python train.py --data_dir=<path to data directory>
```

If your want to run testing, please run using the shell script `train.sh` in the terminal.

```
bash train.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
