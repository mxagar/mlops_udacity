# Udacity Machine Learning DevOps Engineer: Project 1 - Predicting Customer Churn with Cleean Code

In this project, we identify credit card customers that are most likely to churn; instead of focusing on the modelling, the goal is to apply clean code principles:

- Readable, simple, concise code
- PEP8 conventions
- Refactoring: Modular and efficient code
- Documentation at different stages: code, README, etc.
- Error handling
- Testing
- Logging

The used dataset is from Kaggle: [Credit Card customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code).

## Tasks

Refactor the model in `churn_notebook.ipynb` to create the following files:

1. `churn_library.py`
2. `churn_script_logging_and_tests.py`
3. `README.md`

The notebook `Guide.ipynb` is a guide; this document you are reading contains a summary of its contents and more information.

The file `churn_library.py` should complete a typical data science process, which is displayed in the sequence diagram shown below, and which includes the following steps:

- EDA
- Feature Engineering (including encoding of categorical variables)
- Model Training
- Prediction
- Model Evaluation

![Sequence diagram](./pics/sequence_diagram.jpeg)


### `churn_library.py`

Follow the instructions in the function signatures; add `if __name__ == "__main__"` to check the functions:

```bash
python churn_library.py
```

Look at the sequence diagram to better understand the function calls.

### `churn_script_logging_and_tests.py`

The file should contain (1) unit tests and (2) logging of anything that occurs in the tests.

If the file is run, we should get logs in `logs/`:

```bash
python churn_script_logging_and_tests.py
```

### `README.md`

Suggested sections:

- Project description
- Files and data description
- Running the files

The use of `churn_library.py` and `churn_script_logging_and_tests.py` should be explained.

## Rubric / Requirements

Use `pylint` and `autopep8` to obtain at least 7/10:

```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

Criteria:

- PEP8 is followed: correct naming, syntax, import ordering, documentation, etc.
- The README file provides an overview of the project, the instructions to use the code
- All functions and files have document strings; functions: inputs, outputs, and purpose; files: purpose, author, date.
- Tests written for each function.
- Log for info and errors in each test function.
- Logs stored in a `.log` file.
- Easy to understand error and info messages.
- Test and logging can be completed on the command line; info on how to do it in README.
- Store EDA plots: univariate quatitative, categorical, bivariate.
- Store result plots: ROC, Feature importances.
- Store at least two model objects that can easily be loaded and used in a production environment: use `joblib` to store `pkl` files.
- Code in churn_library.py completes the process for solving the data science process: EDA, FE, Trainig, Prediction, Evaluation.
- Handle categorical columns: refactor one-hot encoding (by looping)

Suggestions to stand out:

- Re-organize each script to work as a class.
- Update functions to move constants to their own `constants.py` file, which can then be passed to the necessary functions, rather than being created as variables within functions.
- Work towards `pylint` score of 10/10.
- Add dependencies and libraries (or dockerfile) to README.md
- Add requirements.txt with needed dependencies and libraries for simple install.

## Environment installation

```bash
conda create -n dev-nd python=3.6.3
conda activate dev-nd
conda install pip
# I had to modify the matplotlib version: 2.10 -> 2.2.0
~/opt/anaconda3/envs/dev-nd/bin/pip install -r requirements_py3.6.txt
conda install jupyter jupyterlab
~/opt/anaconda3/envs/dev-nd/bin/pip install -U pytest
```

## Notes on the analysis done in `churn_notebook.ipynb`

- Binary classification: churn (0), no churn (1)
- The dataset is very clean: no missing values
- 14 numerical values, 5 qualitative
- Qualitative values are encoded as the churn ratio mean associated to their category; thus, they are converted into numerical ratios
- Some EDA plots are done: histograms, correlations heat maps, bar plots
- Two models are trained: a random forest with grid search and a logistic regression
- ROC curves for both models are obtained, as well as classification reports (accuracy, F1, etc. for each class); the random forest is unbeatable with 0.99 AOC, vs 0.89 AOC for the logistic regression.
- Models are stored as pickles.
- Feature importances are plotted.
- Classification reports are printed as images.

Libraries to look at: `shap`