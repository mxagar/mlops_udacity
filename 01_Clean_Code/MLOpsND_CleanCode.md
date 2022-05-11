# Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree

These are my notes of the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

The nanodegree is composed of four modules:

1. Clean Code Principles
2. Building a Reproducible Model Workflow
3. Deploying a Scalable ML Pipeline in Production
4. ML Model Scoring and Monitoring

Each module has a folder with its respective notes. This folder and file refer to the **first** module: **Clean Code Principles**.

Note that:
- `TODO`
- `TODO`

Mikel Sagardia, 2022.
No guarantees.

## Practical Installation Notes

We need to install:

- Python 3.X
- Anaconda
- Machine Learning libraries: Scikit-Learn, Matplotlib, Pandas, Seaborn, Numpy, etc.
- Git
- GitHub account

```bash
conda create -n ds python=3.7
conda install jupyter numpy pandas matplotlib scipy sympy cython numba pytables jupyterlab pip -y
conda install scikit-learn scikit-image -y
conda install -c pytorch pytorch -y
conda install statsmodels -y
conda install seaborn -y
conda install pandas-datareader -y
# ...
conda activate ds
```

## Overview of Contents

1. Lesson 1: Introduction
2. Lesson 2: Coding Best Practices
3. Lesson 3: Working with Others Using Version Control
4. Lesson 4: Production Ready Code
5. Project: Predict Customer Churn with Clean Code

## 1. Lesson 1: Introduction

Coding machine learning models in local notebooks is a mess. We need to follow some clean code principles in order to be able to work in teams, make the code and models reproducible, maintainable, and deployable.

Clean code principles take time to be implemented; if we perform a simple ad-hoc analysis with a notebook that will not be used any more, we don't need to apply them. In the rest of the cases: di apply them! E.g., when we or others use the code in the future, when the code needs to be deployed, etc.

Clean code principles go back to the 1950's when Fortran was developed; although social coding and code version didn't exist yet, they had the basic principles in mind already.

Typical actions for clean code:

- Write modular code in scripts, not in messy notebooks
- Make code shareable (git) and easily maintainable
- Refactoring
- Optimization
- Tests
- Logging
- Documentation
- Meet PEP8 standards

## 2. Lesson 2: Coding Best Practices


