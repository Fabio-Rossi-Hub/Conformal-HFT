# CONFORMAL-HFT

This project implements conformal prediction techniques for high-frequency trading (HFT) using deep learning models, with a focus on improving the DeepLOB model.

## Description

We employ a conformal wrapper by customizing TorchCP package to calibrate DeepLOB, a popular deep learning architecture for mid-price
forecasting. The study explore different score functions such as Adaptive Prediction Sets (APS),
Regularized APS (RAPS), and Soft APS (SAPS), and uses Temperature Scaling. This study highlights the importance of machine
learning calibration to improve models reliability and informativeness for better risk management
and more informed algorithmic trading decisions.


## Usage
1. Clone this repository:
```
git clone https://github.com/Fabio-Rossi-Hub/CONFORMAL-HFT.git
cd CONFORMAL-HFT
```
2. Create a conda enviroments with the required dependencies:
```
conda env create --file=requirements.yaml
```
3. Replace TorchCP library with `src/custom_calibrators/custom_torchcp` or modify import structure.
   
## Project Structure
```
CONFORMAL-HFT/
├── src/
│   ├── custom_calibrators/
│   │   └── custom_torchcp/  # Modified TorchCP library
│   ├── venn_abers.py
├── data/
├── hyperparameters/
│   ├── optimal_briercsv
│   ├── optimal_logloss.csv
│   ├── optimal_minsetsize.csv
│   └── optimal_unilabel.csv
├── input/
│   └── data.zip
├── results/
│   ├── results_maxunilabel.json
│   ├── results_minbrier.json
│   ├── results_minlogloss.json
│   └── results_minsetsize.json
├── helpers/
│   ├── get_datahandler.py
│   └── model.py
├── best_val_model.pytorch
├── DeepLOB.py
├── old/
├── VA_calibration.ipynb
├── utils/
│   ├── pycache/
│   ├── constant.py
│   └── torch_dfs.py
├── evaluation.ipynb
└── torchcp_calibration.ipynb
```

## Key Components

- `src/custom_calibrators/custom_torchcp/`: Contains our modified version of the TorchCP library, customized for this project.
- `data/`: Directory for storing datasets (not included in the repository).
- `hyperparameters/`: Stores CSV files with optimal hyperparameters for different metrics.
- `input/`: Contains the input data (data.zip).
- `results/`: Stores JSON files with results for various metrics.
- `helpers/`: Utility functions for data handling and model operations.
- `utils/`: Additional utility functions and constants.
- `*.ipynb`: Jupyter notebooks for various analyses and calibrations.

## Notebooks

This project includes two Jupyter notebooks that demonstrate the core results:

### 1. torchcp_calibration.ipynb

- Performs hyperparameter tuning and calibration of DeepLOB
- Generates optimal hyperparameters (stored in `hyperparameters/`)

### 2. evaluation.ipynb

- Evaluates the best calibrated model against the original DeepLOB
- Analyzes prediction coverage, set sizes, and other metrics


## Results

Our empirical analysis demonstrates that the conformal DeepLOB model achieves:
- 90% prediction coverage with an average set size of 1.69, with 63% of sets having a single label using SAPS
- Out of sample forecasting accuracy increased from 75% to 87% thanks to uncertainty quantification and Log Loss halved to 0.77

These results highlight the potential of conformal prediction to enhance the reliability and interpretability of deep learning models in financial markets.
