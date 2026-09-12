# Medical Insurance Charges Prediction

Predicting individual medical insurance charges from demographic and lifestyle attributes, using a linear regression pipeline with custom smoker-interaction feature engineering.

## Overview

This project builds a regression model to estimate `Charges` (medical insurance cost) from a person's age, sex, BMI, number of children, smoking status, and region. The core insight driving the feature engineering is that smoking status doesn't just shift charges up or down on average — it changes the *relationship* between charges and both age and BMI. The pipeline is built to let a linear model capture that directly.

## Dataset

`input_files/insurance.csv` contains 1,337 records, each representing one individual, with no missing values.

| Column | Description |
|---|---|
| `Age` | Age of the individual |
| `Sex` | Categorical — male / female |
| `BMI` | Body mass index |
| `Children` | Number of dependents covered by insurance |
| `Smoker` | Categorical — yes / no |
| `Region` | Categorical — residential region in the US |
| `Charges` | **Target** — individual medical insurance cost (USD) |

## Project Structure

```
.
├── README.md
├── work_space.ipynb                 # Main analysis & modeling notebook
├── input_files/
│   └── insurance.csv                # Raw dataset
└── output_files/
    ├── linear_model.pkl             # Saved model (optional — see Notes)
    ├── model_evaluation.txt         # Training, cross-validation & test metrics
    └── insurance_predictions.csv    # Test-set predictions with residuals (optional)
```

## Methodology

**1. Exploratory Data Analysis**
- Column names standardized to consistent casing
- Pairplot and boxplots to examine relationships between each feature and `Charges`
- Findings: `Smoker` status is a clear, strong differentiator of `Charges`; `Sex` and `Region` show no clear relationship with `Charges`; `Age` and `BMI` each show a roughly linear relationship with `Charges`, but the slope of that relationship differs sharply between smokers and non-smokers
- `Charges` distribution is right-skewed

**2. Train/Test Split**
- `Age` binned into four life-stage groups (young adult, reproductive peak, mid-life, pre-Medicare) purely to support stratified sampling — this binned column is not used as a model input
- Stratified 80/20 split on age group combined with sex, verified to preserve the original sex ratio in both splits

**3. Feature Engineering**
- `Children`: standard-scaled
- `Sex`, `Region`: one-hot encoded
- `Age` and `BMI`: each split into two separate dimensions via custom `FunctionTransformer`s — one holding the value only for smokers (zeroed out for non-smokers), one holding it only for non-smokers (zeroed out for smokers), each then standard-scaled. This lets a single linear model learn a different effective slope for smokers vs. non-smokers, instead of forcing one shared slope across both groups.
- All steps combined into a single `ColumnTransformer` and `Pipeline`

**4. Model**
- Linear Regression

**5. Evaluation**
- Training RMSE and 10-fold cross-validated RMSE computed on the training set
- Final model evaluated once on the held-out test set

## Features that the Model is trained upon
- "Age" 
- "Children"
- "Sex" : "male", "female"
- "Smoker" : "yes", "no"
- "Region" : "southwest", "southeast", "northeast", "northwest"
- "BMI"

## Input and Output datatype
- input datatype : pandas dataframe
- output datatype : numpy ndarray

## Results

### Test Set Performance

| Metric | Value |
|---|---|
| RMSE | 5,159.89 |
| MAE | 3,109.34 |
| MAE / Median Charges | 32.75% |
| R² | 0.839 |

### Training Set Performance

| Metric | Value |
|---|---|
| RMSE | 5,027.75 |
| 10-fold CV RMSE (mean) | 5,054.06 |
| 10-fold CV RMSE (std) | 554.35 |
| 10-fold CV RMSE (min / max) | 3,874.95 / 5,957.64 |

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Installation

```bash
git clone https://github.com/QuantamSayan/insurance.git
cd insurance
pip install -r requirements.txt
```

Minimum dependencies:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## Usage

```bash
jupyter notebook work_space.ipynb
```

Run all cells in order. The notebook loads `input_files/insurance.csv`, runs the full preprocessing and modeling pipeline, and prints evaluation metrics at the end.

## Notes

- Model saving (`output_files/linear_model.pkl`) and prediction export (`output_files/insurance_predictions.csv`) are present in the notebook but commented out by default. Uncomment those cells to generate them locally.
- This project was developed and tested in a Kaggle notebook environment before being adapted for local/GitHub use.

## Acknowledgments

- Built on a widely used medical insurance cost dataset commonly distributed as `insurance.csv`.