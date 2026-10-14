# NeuralWave_EN3150_A03
EN3150 Assignment 03 - Simple convolutional neural network to perform classification.

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

Download the dataset from [here](https://archive.ics.uci.edu/dataset/908/realwaste),
save it as `data/realwaste.zip`, then unzip it:

```bash
cd data
unzip realwaste.zip
```

The images should end up in `data/realwaste-main/RealWaste/<ClassName>/*.jpg`.

## 3. Run

Open the notebooks in this order and click "Run All":

| # | Notebook | Assignment task |
|---|---|---|
| 1 | `data_preparation.ipynb` | Task 1: data preparation (64x64, 70/15/15 split) |
| 2 | `custom_models.ipynb` | Task 2: Model A / Model B architectures, parameter counts, activations |
| 3 | `optimizer_tuning.ipynb` | Task 3: SGD vs SGD + momentum vs Adam |
| 4 | `sota_transfer_learning.ipynb` | Task 5: MobileNetV2 / SqueezeNet transfer learning + fine-tuning |
| 5 | `main_report.ipynb` | Task 4 (Model A vs B) and Task 6 (Model B vs SOTA), all four models retrained |

```bash
jupyter notebook
```

(Each one takes a few minutes to train on a CPU; `main_report.ipynb` takes the
longest, roughly 10–20 minutes. The first notebook is slow the very first time
since it has to read and resize all the images — after that it's fast.)
