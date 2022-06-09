# Predict Customer Churn

This repository contains the project **Predict Customer Churn** from the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

Overview of contents:

- Project Description
- Files Description
- Installation
- Running Files
- How to Improve This?


## Project Description

In the project, credit card customers that are most likely to churn are identified.  
A first version of the data analysis and modelling is provided in a notebook.  
The main goal of the project is to transform the notebook content into a production ready status, applying clean code principles:

- Readable, simple, concise code
- PEP8 conventions, checked with `pylint` and `autopep8`
- Refactoring: Modular and efficient code
- Documentation at different stages: code, `README` files, etc.
- Error handling
- Testing with `pytest`
- Logging

The used dataset is from Kaggle: [Credit Card customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code). It has the following properties:

- Original dataset size: 28 columns, 10127 data points.
- It consists of categorical and numerical features of clients which churn (cease to be clients) or not; that churn status is the binary classification target used for training, encoded as the feature `Attrition_Flag`.
- Number of features after feature engineering is carried out: 19.
- Modelling used on the data: logistic regression and random forests with grid search.

## Files Description

The notebooks present in the repository are:

- `Guide.ipynb`: the project guideline
- `churn_notebook.ipynb`: the provided data analysis and modelling, which covers:
	- Dataset loading 
	- Exploratory Data Analysis (EDA)
	- Feature Engineering (FE)
	- Training: Random Forest and Logistic Regression models are fit
	- Generation of classification report plots

In the project, the code in `churn_notebook.ipynb` is modified for a production environment and transferred to the following modules:

- `churn_library.py`: this file contains the refactored code from `churn_notebook.ipynb`
- `churn_script_logging_and_tests.py`: this file applies tests to the functions defined in `churn_library.py`

In addition, these files and folders are also present in the repository:

- `requirements_py*.txt`: dependencies to be installed (see section below)
- `README.md`: this documentation file
- `Project_1_Customer_Churn.md`: a summary of `Guide.ipynb` and the project requirements defined by Udacity
- `data/`: folder where the dataset is stored
- `images/`: folder for the EDA and classification report images
- `models/`: folder where the generated models are stored as pickle objects
- `logs/`: folder for the log files

Note that for completing the project, these files have been created/completed:

- `churn_library.py`
- `churn_script_logging_and_tests.py`
- `README.md`

## Installation

First, decide between python 3.6 and 3.8; then, create an environment (e.g., with Anaconda), and install the dependencies:

```bash
conda create -n my-env python=3.6.3
conda activate my-env
conda install pip
# Note: I had to modify the matplotlib version: 2.10 -> 2.2.0
~/opt/anaconda3/envs/my-env/bin/pip install -r requirements_py3.6.txt
conda install jupyter jupyterlab
~/opt/anaconda3/envs/my-env/bin/pip install -U pytest
```

The dependencies are:

```
Python 3.6:

scikit-learn==0.22       
shap==0.40.0     
joblib==0.11
pandas==0.23.3
numpy==1.19.5 
matplotlib==2.2.0      
seaborn==0.11.2
pylint==2.7.4
autopep8==1.5.6

Python 3.8:

scikit-learn==0.24.1
shap==0.40.0
joblib==1.0.1
pandas==1.2.4
numpy==1.20.1
matplotlib==3.3.4
seaborn==0.11.2
pylint==2.7.4
autopep8==1.5.6
```

## Running Files

To **run the data analysis and modelling library**:

```bash
python churn_library.py
```

This command should generate EDA and classification report images, as well as model pickles in their respective folders (see above).

To **test the functions in the data analysis and modelling library** using `pytest`:

```bash
python ccurn_library.py
```

This command tests the functions from `churn_library.py`: in addition to generating all the aforementioned artifacts, checks are perfomed to assure code reliability.

In contrast to standard `pytest` environments, here

- the testing file is *not* prefixed with `test_`
- the testing configuration happens in the testing file itself, not in `conftest.py`
- and the testing functions have logging.

Thus, it is not enough running barely `pytest` in the Terminal.

If you would like to check the PEP8 conformity of the files:

```bash
pylint churn_library.py # should yield 8.30/10
pylint churn_script_logging_and_tests.py # should yield 7.86/10
```

## How to Improve This?

- [x] Add dependencies and libraries to `README.md`.
- [ ] Re-organize each script to work as a class.
- [ ] Update functions to move constants to their own `constants.py` or `conftest.py` file.
- [ ] Work towards `pylint` score of 10/10. However, note that some variable names were chosen to be non-PEP8-conform due to their popular use in the field (e.g., `X_train`).
- [ ] Create Dockerfile.
