# Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree

These are my notes of the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

The nanodegree is composed of four modules:

1. Clean Code Principles
2. Building a Reproducible Model Workflow
3. Deploying a Scalable ML Pipeline in Production
4. ML Model Scoring and Monitoring

Each module has a folder with its respective notes. This folder and file refer to the **third** module: **Deploying a Scalable ML Pipeline in Production**.

Mikel Sagardia, 2022.  
No guarantees.

## Overview of Contents

- [Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree](#machine-learning-devops-engineer-personal-notes-on-the-udacity-nanodegree)
  - [Overview of Contents](#overview-of-contents)
  - [1. Introduction to Deployment](#1-introduction-to-deployment)
    - [Deployment considerations](#deployment-considerations)
    - [Ways to deploy model](#ways-to-deploy-model)
    - [Final module project](#final-module-project)
    - [Stakeholders in Model Deployment](#stakeholders-in-model-deployment)
    - [When and How Should a Model Be Deployed?](#when-and-how-should-a-model-be-deployed)
  - [2. Performance Testing and Preparing a Model for Production](#2-performance-testing-and-preparing-a-model-for-production)
    - [2.1 K-Fold Coss-Validation](#21-k-fold-coss-validation)
    - [2.2 Data Slicing](#22-data-slicing)
    - [2.3 Data Slicing Use Cases](#23-data-slicing-use-cases)
    - [2.4 Unit Testing for Data Slicing](#24-unit-testing-for-data-slicing)
      - [Exercise: Data Slicing with the Iris Dataset](#exercise-data-slicing-with-the-iris-dataset)
    - [2.5 Model Bias](#25-model-bias)
    - [2.6 Investigating Model Bias: The Aequitas Package](#26-investigating-model-bias-the-aequitas-package)
      - [Exercise/Demo: Aequitas Workflow with COMPAS Dataset](#exercisedemo-aequitas-workflow-with-compas-dataset)
      - [Exercise: Aequitas Workflow with Car Evaluation Dataset](#exercise-aequitas-workflow-with-car-evaluation-dataset)
    - [2.7 Model Cards](#27-model-cards)
    - [2.8 Performance Testing: Final Exercise](#28-performance-testing-final-exercise)
  - [3. Data and Model Versioning](#3-data-and-model-versioning)
  - [4. CI/CD](#4-cicd)
  - [5. API Deployment with FastAPI](#5-api-deployment-with-fastapi)

## 1. Introduction to Deployment

The module is taught by [Justin Clifford Smith, PhD](https://www.justincliffordsmith.com/). Principal data scientist at [Optum](https://www.optum.com/).

Deployment consists in putting the model to work in production. But not all deployments are the same: sometimes we want to keep learning during the usage of the model!

The demos and exercises of this module are located in:

- [`./lab`](./lab)
- [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos)

### Deployment considerations

- Model performance and reliability: follow them.
- Model transparency and explainability.
- Legal & privacy aspects: used data is compliant?
- Model resilience and security: does the model run in a secure environment? is it a secure pipeline end-to-end?
- Model management in production: how frequently do we need to re-train and update?
- Management of good data flows.
- Automation: which steps, if not all, can be automated? If there are humans in the loop, is the model scalable?
- Scalability: scaling up or down the model usage dynamically can be essential; thus, always design for scalability.

### Ways to deploy model

- Run on ad-hoc batches.
- Run on a schedule.
- Run with an API.

### Final module project

It's a portfolio project that uses different technologies:

- DVC: Data/model/code version control.
- Github actions (CI).
- Unit tests.
- Continuous deployment (CD) using Heroku.
- An APi with FastAPI.
- Also, we're going to write a model card.

### Stakeholders in Model Deployment

- Users: UI/UX, value?
- Business Leaders: value?
- DevOps Engineers: they want a successful and smooth deployment.
- Colleagues.
- Your future self. Good decisions and efforts pay out.

### When and How Should a Model Be Deployed?

- Steps before deploying a model
  - Test performance
  - Check for bias
  - Write a model card
- Depending on the use case:
  - We might perform inferences on ad-hoc batches
  - We might need to develop an API for performing online inferences
- Always develop thinking on deployment for production
  - Unit tests
  - Continuous integration

## 2. Performance Testing and Preparing a Model for Production

Goals of the lesson:

- Analyze slices of data when training ans testing models:
  - Data slicing
  - K-fold cross validation
  - Testing
- Probe a model for bias using common frameworks such as Aequitas.
- Write model cards that explain teh purpose, provenance, and pitfalls of a model.

### 2.1 K-Fold Coss-Validation

In the beginning, we split our dataset in:

- training
- testing

The testing dataset will be evaluated only once at the end. The training split is further split in:

- train
- validation

The easiest way of computing the validation split is just taking a fraction of the dataset; typically, the **single validation set**

- is a fraction of 10-30%; but if the dataset is very large, it can be as low as 1%
- it is evaluated several times, during the training
- it should contain all classes; we can use *stratify* tools for that
- it should cover the entire dataset

However, a better way consists in using **K-Fold Cross-Validation (CV)**: we split the data in `k` subsets; we train in `k-1` subsets and validate in the last subset. Then, we repeat the process `k` times, and each time the validation subset is different. Then, the performance is averaged across all `k` subsets. Thus, it is more robust, because

- we use the entire dataset,
- an error average is computed.

However, this is expensive, because we need to train the model `k` times -- it is used for non Neural Network techniques. For Neural Networks, a unique validation split is used.

### 2.2 Data Slicing

Data slicing consists in stratifying or grouping the data according to

- class values: the output label itself
- or feature values: age groups, etc.

That is, we fix the value of a feature/label. Then, for each group with a fixed value the performance metric is computed.

Whereas cross-validation takes random samples or rows (horizontal slices) and computes the overall performance, data slicing takes groups in columns (vertical slices) and computes group performance.

An overall metric value with cross validation can be good, but the same metric in a data slice can be bad -- and it is undetected if we don't perform data slicing!

![Data Slicing](./pics/data_slicing.jpg)

### 2.3 Data Slicing Use Cases

Possible slicing groups:

- Features
- Label classes
- Length of audio/text
- Sources of data

We should select the slices which are *relevant* to the model.

We can perform slicing:

- Pre-deployment: during training
- Post-deployment: the same slices we used during training should be used in the monitoring!

Interesting links:

- [Slice-based Learning](https://www.snorkel.org/blog/slicing)
- [Slice-based Learning: A Programming Model for Residual Learning in Critical Data Slices](https://papers.nips.cc/paper/2019/file/351869bde8b9d6ad1e3090bd173f600d-Paper.pdf)

### 2.4 Unit Testing for Data Slicing

Data slicing is used together with unit tests: we write unit tests that check the performance of data slices.

Unit tests can be run pre- and post-deployment.

Write unit tests following the SOLID principles:

- Single responsibility: every class/function should do only one thing.
- Open-closed: software should be open to extension but closed to modification.
- Liskov substitution: functions that use pointers to base classes should be able to use derived classes.
- Interface segregation: clients should not be forced to depend upon interfaces that they don't use.
- Dependency inversion: depend upon abstractions, not concretions.

Example of unit test for data slicing:

```python

### --- foo.py

def foo():
    return "Hello world!"

### --- test_foo.py

from .foo import foo

def test_foo():
    foo_result = foo()

    expected_foo_result = "Hello world!"
    assert foo_result == expected_foo_result

### --- test_slice.py

import pandas as pd
import pytest


@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "numeric_feat": [3.14, 2.72, 1.62],
            "categorical_feat": ["dog", "dog", "cat"],
        }
    )
    return df


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_slice_averages(data):
    """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
    for cat_feat in data["categorical_feat"].unique():
        avg_value = data[data["categorical_feat"] == cat_feat]["numeric_feat"].mean()
        assert (
            2.5 > avg_value > 1.5
        ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."

### --- Run

pytest -v test_slice.py

```

#### Exercise: Data Slicing with the Iris Dataset

Used dataset: [Iris](https://archive.ics.uci.edu/ml/datasets/Iris).

> Load the data using Pandas and then write a function that outputs the descriptive stats for each numeric feature while the categorical variable is held fixed. Run this function for each of the four numeric variables in the Iris data set.

The code is in [`DataSlicing_UnitTest.ipynb`](./lab/DataSlicing_UnitTest.ipynb):

```python
def slice_iris(df, feature):
    """ Function for calculating descriptive stats on slices of the Iris dataset."""
    for cls in df["class"].unique():
        # Alternative:
        # df.groupby('class').mean()
        # df.groupby('class').std()
        df_temp = df[df["class"] == cls]
        mean = df_temp[feature].mean()
        stddev = df_temp[feature].std()
        print(f"Class: {cls}")
        print(f"{feature} mean: {mean:.4f}")
        print(f"{feature} stddev: {stddev:.4f}")
    print()

slice_iris(df, "sepal_length")
slice_iris(df, "sepal_width")
slice_iris(df, "petal_length")
slice_iris(df, "petal_width")
```

### 2.5 Model Bias

- Bias = performance in favor or against one group in an unfair manner.
- Do not confuse with the bias of the bias-variance tradeoff.
- Model bias is often related to data bias, but not always (?).
- Types of **data bias**:
  - Human error
    - Sampling error: the sample doesn't capture the real population.
    - Exclusion bias: a group is excluded from the data acquisition; that can be on purpose/unwanted.
    - Recall bias: the unreliability of trying people to remember past events; data that consists of past event descriptions is usually unreliable.
  - Society error:
    - Implicit bias / stereotypes.
    - Unjust policy.

Bias is everywhere! Examples of data bias:

- In a dataset with animals, the bias in favor of cats might be significant.
- In a dataset with humans, a bias against black people can be significant.
- Collection: images from web might have a higher quality than the ones fed by regular users.
- Annotation: labeler bias, due to many factors, e.g., challenging time with little time.
- Preprocessing: truncating long text so that it fits into memory/model input size; maybe longer texts are long because they have certain characteristic features!

### 2.6 Investigating Model Bias: The Aequitas Package

[Aequitas](http://aequitas.dssg.io/) is a tool that quantifies bias; it has 3 main interfaces:

- Web app.
- CLI.
- Python package: shown here.

In Aequitas we specify the score, label, and at least one categorical field (or numerical in ranges/cuts) and then three reports are created comparing against a reference group (often the majority):

- Group Metrics Results
- Bias Metrics Results
- Fairness Measures Results

Install it:

```bash
pip install aequitas
```

#### Exercise/Demo: Aequitas Workflow with COMPAS Dataset

Links to the demo/exercise:

- Demo repository: [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos).
- Exercise/demo notebook: [`aequitas_demo.ipynb`](https://github.com/mxagar/mlops-udacity-deployment-demos/blob/main/aequitas_demo.ipynb).
- A more detailed official demo of Aequitas with the COMPAS dataset: [compas_demo.ipynb](https://github.com/dssg/aequitas/blob/master/docs/source/examples/compas_demo.ipynb)


The [COMPAS dataset](https://github.com/dssg/aequitas/tree/master/examples) is used in the example; it contains a **recidivism** score, i.e., the tendency of a convict to re-offend. We have the 7214 observations with these 6 columns:

- `entity_id`
- `score`: score by the model, required by Aequitas.
- `label_value`: true label, required by Aequitas.
- `race`
- `sex`
- `age_cat`

:warning: Note: as far as I understand, Aequitas works with binary classifications?

The package seems a little bit dense and complex to understand straightforwardly, because it provides many statistics. Go to the notebook to understand how to make a basic use of Aequitas; the general workflow is:

- We create a `Group()` object, from which we extract a crosstab; the crosstab contains the data slicing statistics: for each categorical feature and each level/group in them we have a slice/group for which the statistics are computed.
- We create a `Bias()` object and compute disparities of the slices wrt. a reference group. We specify the reference group in `ref_groups_dict`; if we don't specify any reference group, the majority group/slice is taken. We get the same statistics as before + disparity statistics, differences wrt. reference group
- We compute the `Fairness()`: We get the same statistics as before + Fairnes true/false parity values.
- We can obtain summary values and plots, too.

#### Exercise: Aequitas Workflow with Car Evaluation Dataset

The exercise is similar to the previous one, with a bit more EDA. Have a look at it:

Exercise/demo notebook: [`car_aequitas_solution.ipynb`](https://github.com/mxagar/mlops-udacity-deployment-demos/blob/main/aequitas_car_exercise/).

### 2.7 Model Cards

Model cards are documentation pages with all the key aspects to be able to use it in all possible ways: extend, re-train, etc. There is no industry standard, but we should include:

- Model details: type, training/hyperparameter details, links to longer docu
- Intended use and users.
- Metrics: overall performance and key slice performance.
- Relevant figures: learning curves, etc.
- Data: information on the training and the validation splits; how they were acquired & processed, etc.
- Bias: inherent in data or model.
- Caveats, if there are any.

![Model Card Example](./pics/model-card-cropped.jpg)

Links:

- [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993.pdf)
- [Google on Model Cards](https://modelcards.withgoogle.com/about)

Example Model Card:

> **Model Details**
> 
> Justin C Smith created the model. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.
> 
> **Intended Use**
>
> This model should be used to predict the acceptability of a car based off a handful of attributes. The users are prospective car buyers.
> 
> **Metrics**
> 
> The model was evaluated using F1 score. The value is 0.8960.
> 
> **Data**
> 
> The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation). The target class was modified from four categories down to two: "unacc" and "acc", where "good" and "vgood" were mapped to "acc".
> 
> The original data set has 1728 rows, and a 75-25 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.
> 
> **Bias**
> 
> According to Aequitas bias is present at the unsupervised and supervised level. This implies an unfairness in the underlying data and also unfairness in the model. From Aequitas summary plot we see bias is present in only some of the features and is not consistent across metrics.

### 2.8 Performance Testing: Final Exercise

Exercise repository: [Performance_testing_FinalExercise](https://github.com/mxagar/mlops-udacity-deployment-demos/tree/main/Performance_testing_FinalExercise).

:construction:

## 3. Data and Model Versioning



## 4. CI/CD



## 5. API Deployment with FastAPI


