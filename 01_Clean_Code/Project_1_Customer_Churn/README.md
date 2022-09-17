# Predict Customer Churn Using Production-Level Code

This repository contains a project that analyzes and predicts **Customer Churn** using the [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code) dataset from [Kaggle](https://www.kaggle.com/). 

**IMPORTANT NOTE: This repository is old; a new more advanced version can be found here: []()**

Data analysis and modeling pipelines are implemented in the project to end up with an interpretable model that is also able to predict customer churn. However, the focus of the project are neither the business case nor the analysis and modeling techniques; instead, **the goal is to provide with a boilerplate that shows how to transform a research notebook into a development/production environment code**. Also, note that not all aspects necessary in a ML worklflow are covered either:

- The EDA and FE are very simple.
- The data artifacts are not tracked or versioned, neither are the models or the pipelines.
- No through inference is defined.
- No through deployment nor API are performed.
- etc.

If you are interested in some of the aforementioned topics, you can visit other of my boilerplate projects listed in the section [Interesting Links](#interesting-links).

The starter code comes from the [Udacity Machine Learning DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

Overview of contents:

- [Predict Customer Churn Using Production-Level Code](#predict-customer-churn-using-production-level-code)
	- [Project Description](#project-description)
	- [Files Description](#files-description)
	- [How to Use This](#how-to-use-this)
		- [Installation](#installation)
		- [Running the Scripts](#running-the-scripts)
	- [Possible Improvements](#possible-improvements)
	- [Interesting Links](#interesting-links)
	- [Authorship](#authorship)


## Project Description

In the project, credit card customers that are most likely to churn are identified.  
A first version of the data analysis and modeling is provided in a notebook.  
**The main goal of the project is to transform the notebook content into a production ready status, applying clean code principles**:

- Readable, simple, concise code
- PEP8 conventions, checked with `pylint` and `autopep8`
- Refactoring: Modular and efficient code
- Documentation at different stages: code, `README` files, etc.
- Error handling
- Testing with `pytest`
- Logging

The used [Credit Card Customers dataset from Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code) has the following properties:

- Original dataset size: 28 columns, 10127 data points.
- It consists of categorical and numerical features of clients which churn (cease to be clients) or not; that churn status is the binary classification target used for training, encoded as the feature `Attrition_Flag`.
- Number of features after feature engineering is carried out: 19.
- Modelling used on the data: logistic regression and random forests with grid search.

## Files Description

The notebooks present in the repository are:

- [`Guide.ipynb`](Guide.ipynb): the project guideline (from Udacity; uninteresting)
- [`churn_notebook.ipynb`](churn_notebook.ipynb): the provided data analysis and modeling, which covers:
	- Dataset loading 
	- Exploratory Data Analysis (EDA) and Data Cleaning
	- Feature Engineering (FE)
	- Training: Random Forest and Logistic Regression models are fit
	- Generation of classification report plots

In the project, the code in [`churn_notebook.ipynb`](churn_notebook.ipynb) is modified for a production environment and transferred to the following modules:

- [`churn_library.py`](churn_library.py): this file contains the refactored code from [`churn_notebook.ipynb`](churn_notebook.ipynb).
- [`churn_script_logging_and_tests.py`](churn_script_logging_and_tests.py): this file applies tests to the functions defined in [`churn_library.py`](churn_library.py).

The following sequence diagram shows the workflow in the main file [`churn_library.py`](churn_library.py):

![Sequence Diagram of churn_library.py](./images/../pics/sequencediagram.jpeg)

In addition, those files and folders are also present in the repository:

- `requirements_py*.txt`: dependencies to be installed (see section below)
- `README.md`: this documentation file
- `Customer_Churn_Guide.md`: a summary of `Guide.ipynb` and the project requirements defined by Udacity
- `data/`: folder where the dataset is stored
- `images/`: folder for the EDA and classification report images
- `models/`: folder where the generated models are stored as pickle objects
- `logs/`: folder for the log files

## How to Use This
### Installation

First, decide between python 3.6 and 3.8; then, create an environment (e.g., with [conda](https://docs.conda.io/en/latest/)), and install the dependencies:

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

### Running the Scripts

Basically, the processing from the notebook [`churn_notebook.ipynb`](churn_notebook.ipynb) has been transferred to the module [`churn_library.py`](churn_library.py), which also performs additional tasks. To **run that data analysis and modeling module**:

```bash
python churn_library.py
```

This command should generate EDA and classification report images, as well as model pickles in their respective folders (see above).

To **test the functions in the data analysis and modeling library** using `pytest`:

```bash
python churn_script_logging_and_tests.py
```

This command tests the functions from `churn_library.py`: in addition to generating all the aforementioned artifacts, checks are performed to assure code reliability.

In contrast to standard `pytest` environments, here

- the testing file is *not* prefixed with `test_`
- the testing configuration happens in the testing file itself, not in `conftest.py`
- and the testing functions have logging.

Thus, it is not enough running barely `pytest` in the Terminal. The main reason why this is so is to enable logging during testing.

If you would like to **check the PEP8 conformity** of the files:

```bash
pylint churn_library.py # should yield 8.30/10
pylint churn_script_logging_and_tests.py # should yield 7.86/10
```

Tip: If you'd like to automatically edit and improve the score of a file that already has a good score, you can try `autopep8`:

```bash
autopep8 --in-place --aggressive --aggressive churn_library.py
```

## Possible Improvements

- [x] Add dependencies and libraries to `README.md`.
- [ ] Re-organize each script to work as a class.
- [ ] Update functions to move constants to their own `constants.py` or `conftest.py` file.
- [ ] Work towards `pylint` score of 10/10. However, note that some variable names were chosen to be non-PEP8-conform due to their popular use in the field (e.g., `X_train`).
- [ ] Create Dockerfile.

## Interesting Links

- Guide on EDA, Data Cleaning and Feature Engineering: [eda_fe_summary](https://github.com/mxagar/eda_fe_summary).
- A boilerplate for reproducible ML pipelines with MLflow and Weights & Biases which uses a music genre classification dataset from Spotify: [music_genre_classification](https://github.com/mxagar/music_genre_classification).
- My notes on the [Udacity Machine Learning DevOps Engineering Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821), which features more MLOps content: [mlops_udacity](https://github.com/mxagar/mlops_udacity).
## Authorship

Mikel Sagardia, 2022.  
No guarantees.

The original content of this project was taken from the [Udacity Machine Learning DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821), but it has been modified and extended to the present state.