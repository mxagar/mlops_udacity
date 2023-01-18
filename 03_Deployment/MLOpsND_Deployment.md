# Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree

These are my notes of the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

The nanodegree is composed of four modules:

1. Clean Code Principles
2. Building a Reproducible Model Workflow
3. Deploying a Scalable ML Pipeline in Production
4. ML Model Scoring and Monitoring

Each module has a folder with its respective notes. This folder and file refer to the **third** module: **Deploying a Scalable ML Pipeline in Production**.

I have extended this module with material from other courses:

- [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025); repository: [data_science_udacity](https://github.com/mxagar/data_science_udacity).
- [Deployment of Machine Learning Models](https://www.udemy.com/course/deployment-of-machine-learning-models) by Soledad Galli & Christopher Samiullah; repository: [deploying-machine-learning-models](https://github.com/mxagar/deploying-machine-learning-models).
- [Complete Tensorflow 2 and Keras Deep Learning Bootcamp](https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/) J.M. Portilla; repository: [data_science_python_tools](https://github.com/mxagar/data_science_python_tools/tree/main/19_NeuralNetworks_Keras)

Although I provide the links to the repositories with other notes, everything is self-contained here, too.

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
    - [2.3 Data Slicing Use Cases and Overall Workflow](#23-data-slicing-use-cases-and-overall-workflow)
    - [2.4 Unit Testing for Data Slicing](#24-unit-testing-for-data-slicing)
      - [Exercise: Data Slicing with the Iris Dataset](#exercise-data-slicing-with-the-iris-dataset)
    - [2.5 Model Bias](#25-model-bias)
    - [2.6 Investigating Model Bias: The Aequitas Package](#26-investigating-model-bias-the-aequitas-package)
      - [Exercise/Demo: Aequitas Workflow with COMPAS Dataset](#exercisedemo-aequitas-workflow-with-compas-dataset)
      - [Exercise: Aequitas Workflow with Car Evaluation Dataset](#exercise-aequitas-workflow-with-car-evaluation-dataset)
    - [2.7 Model Cards](#27-model-cards)
    - [2.8 Performance Testing, Final Exercise: Manual Data Slicing](#28-performance-testing-final-exercise-manual-data-slicing)
  - [3. Data and Model Versioning](#3-data-and-model-versioning)
    - [3.1 Introduction Data Version Control (DVC): Installation \& Resemblance to Git](#31-introduction-data-version-control-dvc-installation--resemblance-to-git)
      - [Installation on Mac](#installation-on-mac)
      - [Resemblance to Git](#resemblance-to-git)
    - [3.2 Tracking with DVC: Local Remote](#32-tracking-with-dvc-local-remote)
    - [3.3 Remote Storage](#33-remote-storage)
      - [Example: GDrive Remote Storage](#example-gdrive-remote-storage)
    - [3.4 Pipelines with DVC](#34-pipelines-with-dvc)
      - [Exercise: Defining a Pipeline](#exercise-defining-a-pipeline)
        - [Solution](#solution)
    - [3.5 Experiment Tracking with DVC](#35-experiment-tracking-with-dvc)
  - [4. CI/CD: Continuous Integration and Continuous Deployment](#4-cicd-continuous-integration-and-continuous-deployment)
    - [4.1 Software Engineering Principles: Automation, Testing, Versioning](#41-software-engineering-principles-automation-testing-versioning)
      - [Automation](#automation)
      - [Testing - in the Context of Machine Learning](#testing---in-the-context-of-machine-learning)
      - [Versioning](#versioning)
    - [4.2 Continuous Integration and Continuous Delivery: Definition](#42-continuous-integration-and-continuous-delivery-definition)
    - [4.3 Continuous Integration with Github Actions](#43-continuous-integration-with-github-actions)
    - [4.4 Continuous Deployment with Heroku](#44-continuous-deployment-with-heroku)
      - [Introduction to Heroku](#introduction-to-heroku)
      - [Continuous Deployment to Heroku: Demo](#continuous-deployment-to-heroku-demo)
    - [4.5 Deployment to Render Cloud Platform](#45-deployment-to-render-cloud-platform)
  - [5. API Deployment with FastAPI](#5-api-deployment-with-fastapi)
    - [5.1 Review of Python Type Hints](#51-review-of-python-type-hints)
    - [5.2 Fundamentals of FastAPI](#52-fundamentals-of-fastapi)
      - [Basic Example](#basic-example)
      - [Core Features of FastAPI](#core-features-of-fastapi)
      - [Example/Demo Using the App Above](#exampledemo-using-the-app-above)
      - [Example/Demo: Variable Paths and Testing](#exampledemo-variable-paths-and-testing)
    - [5.3 Local API Testing](#53-local-api-testing)
      - [Exercise: Testing Example](#exercise-testing-example)
    - [5.4 Heroku Revisited: Procfiles and CLI](#54-heroku-revisited-procfiles-and-cli)
      - [Heroku CLI](#heroku-cli)
    - [5.5 Live API Testing with Requests Package](#55-live-api-testing-with-requests-package)
      - [Exercise/Demo: Data Ingestion and Error Handling with FastAPI](#exercisedemo-data-ingestion-and-error-handling-with-fastapi)
    - [5.6 Extra](#56-extra)
  - [6. Project: Deploying a Model with FastAPI to Heroku](#6-project-deploying-a-model-with-fastapi-to-heroku)
  - [7. Excurs: Docker and AWS EC2](#7-excurs-docker-and-aws-ec2)
  - [8. Summary Project: Vanilla Deployments](#8-summary-project-vanilla-deployments)

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

### 2.3 Data Slicing Use Cases and Overall Workflow

Possible slicing groups:

- Features:
  - if categorical, each level/class can be a slice
  - if numerical, we can cut in ranges/regions (e.g., above/below mean) and each is a slice
- Label classes
- Length of audio/text
- Sources of data

We should select the slices which are *relevant* to the model.

The workflow is the following:

- We train our model on the entire dataset.
- We have now the target label + predicted outcome.
- We compute the overall metric as always, e.g., F1: `y_true` vs. `y_pred`.
- We take our slices and check how good the metric of each slice is compared to the overall!

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

### 2.8 Performance Testing, Final Exercise: Manual Data Slicing

Exercise repository: [Performance_testing_FinalExercise](https://github.com/mxagar/mlops-udacity-deployment-demos/tree/main/Performance_testing_FinalExercise).

In this exercise the [Raisin dataset](https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset) is used. In it, 900 observations of 7 raising features (all numerical) are collected; each observation has a target class which specifies one of two types of raisins: 'Kecimen' or 'Besni'.

Data slicing is applied manually to the dataset as follows:

- Data is split: `train`, `validation`.
- Logistic regression model is fit and F1 is computed: overall score.
- 3 features are chosen and their mean is computed; then, 2 buckets or ranges are defined for each of the 3 features: above and below the mean. Thus, we get 3x2 = 6 slices.
- For each slice, the F1 metric is computed again and compared to the overall F1.
- Finally, a model card is written.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, f1_score

data = pd.read_csv("./exercise_data/Raisin_Dataset.csv")

data.head()
data.head() # (900, 8)
data.Class.unique() # array(['Kecimen', 'Besni'], dtype=object)

# Take features + target
y = data.pop("Class")

# Split the data into train and validation, stratifying on the target feature.
X_train, X_val, y_train, y_val = train_test_split(data, y, stratify=y, random_state=23)

# Get a high level overview of the data.
# This will be useful for slicing: we take the mean values from here!
X_train.describe()

# Define and fit model
lr = LogisticRegression(max_iter=1000, random_state=23)
lb = LabelBinarizer()

# Binarize the target feature.
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)

# Train Logistic Regression.
lr.fit(X_train, y_train.ravel())

# Use sklearn's classification report to get an overall view of our classifier.
print(classification_report(y_val, lr.predict(X_val)))
# Average F1: 0.88

## Data Slicing + Performance

print("F1 score on MajorAxisLength slices:")
row_slice = X_val["MajorAxisLength"] >= 427.7
print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))
row_slice = X_val["MajorAxisLength"] < 427.7
print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))
# F1 score on MajorAxisLength slices:
# 0.0
# 0.9230769230769231

print("\nF1 score on MinorAxisLength slices:")
row_slice = X_val["MinorAxisLength"] >= 254.4
print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))
row_slice = X_val["MinorAxisLength"] < 254.4
print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))
# F1 score on MinorAxisLength slices:
# 0.65
# 0.9319371727748692

print("\nF1 score on ConvexArea slices:")
row_slice = X_val["ConvexArea"] >= 90407.3
print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))
row_slice = X_val["ConvexArea"] < 90407.3
print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))
# F1 score on ConvexArea slices:
# 0.30769230769230765
# 0.9174311926605505
```

Model card reported in the solution:

> #### Model Card
> ##### Model Details
> Logistic Regresion model using default scikit-learn hyperparameters. Trained with sklearn version 0.24.1.
> ##### Intended Use
> For classifying two types of raisins from Turkey.
> ##### Metrics
> F1 classification with a macro average of 0.85, 0.84 for the minority class, and 0.85 for the majority class.
> When analyzing across data slices, model performance is higher for raisins below the average size and much lower for raisins above the average.
> ##### Data
> Raisin dataset acquired from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset
> Originally from: Cinar I., Koklu M. and Tasdemir S., Classification of Raisin Grains Using Machine Vision and Artificial Intelligence Methods. Gazi Journal of Engineering Sciences, vol. 6, no. 3, pp. 200-209, December, 2020.
> ##### Bias
> The majority of raisins are below the average size. This could be a potential source of bias but more subject matter expertise may be necessary. Note to students: this is a useful call out, and in a real-world scenario should prompt you to engage in collaboration with subject matter experts so you can flesh this out.

## 3. Data and Model Versioning

DVC = Data Version Control. We use it to control the version of:

- data
- models
- etc.

either (1) locally or (2) remotely. Additionally, we can use it to 

- create pipelines 
- and track experiments.

Data provenance or data lineage aspects, i.e., complete record of data whereabouts:

- Origin: where data comes from.
- Movement: systems and location it was.
- Manipulation: how data has been transformed.

Data provenance is important because:

- We forget all the history and quality data if not recorded.
- Accountability: adjust to regulations.
- Future-proofing of data is only possibly if we have a complete record of all origins and movements.
- Clear understanding of data and models is possible only if we have a complete lineage.

Some links:

- [List of Provenance Tools](https://projects.iq.harvard.edu/provenance-at-harvard/tools)
- [Provenance Use Cases](https://confluence.csiro.au/public/PROMS/provenance-use-cases)

### 3.1 Introduction Data Version Control (DVC): Installation & Resemblance to Git

DVC relies on Git to do the version control and it doesn't store the data/models itself, but that is done by external or remote services like S3, HDFS, GDrive, etc.

DVC provides tools to track changes in large/binary files.

Links:

- Documentation: [https://dvc.org/doc/start](https://dvc.org/doc/start).
- Installation: [https://dvc.org/doc/install](https://dvc.org/doc/install).

#### Installation on Mac

There are several ways of installing DVC; we need an environment with Python 3.8+ to run the latest version, which is necessary for some remote storage functionalities.

```bash
conda activate mlops-nd # Python 3.8

# with brew (outdated)
brew install dvc

# or with pip
# BUT: we need to have an environment with python 3.8+
# to have the latest version of dvc
pip install -U "dvc[all]" # all: all remote storage interfaces are installed

# also a conda installation is possible,
# but it didn't work for me
```

#### Resemblance to Git

DVC os designed to have a very similar use as git:

```bash
# Initialize project, ./dvc/ folder is created
dvc init

# Add files/folders to tracking
dvc add

# Upload download data from the remote store, specified in dvc.yaml
dvc push
dvc pull

# This one is different than git commit
dvc commit
```

[DVC Reference](https://dvc.org/doc/command-reference): Typical workflow:

- `dvc init`
- `dvc add`
- Create a `dvc.yaml` file: it contains the processing pipeline + artifact outputs
- `dvc repro`: execute or restore any version of the pipeline
- `dvc push`, `dvc pull`: access remote storage

### 3.2 Tracking with DVC: Local Remote

```bash
# 0. Make sure we're on a git repo;
# if not, initialize first git, then dvc
git init
dvc init
# dvc init generates:
# .dvc/ folder
# .dvcignore: equivalent to .gitignore

# 1. Create a local remote folder
mkdir ~/data/remote
dvc remote add -d localremote ~/data/remote
# list available remotes
dvc remote list
# In addition to local remote folders, we can use
# real remote storage: S3, GDrive, etc.
# Check how config file changed
less .dvc/config

# 2. Track files
dvc add sample.csv
# We get a prompt saying that
# we should add .gitignore
# and sample.csv.dvc to git
# NOTE: sample.csv is added to .gitignore for us!
git add sample.csv.dvc .gitignore

# 2. Commit changes to git
git commit -m "initial commit of dataset using dvc"
# If our git repo is connected to a remote origin
# we can always do git push/pull

# 3. Send data to local remote
dvc push

# 4. Retrieve data from local remote
dvc pull

# 5. Change a dataset and track changes
vim sample.csv # now, change something
dvc add sample.csv
git add sample.csv.dvc
git commit -m "changes..."
dvc push

# 6. Manage remotes
# Change/add properties to a remote
dvc remote modify
# Rename a remote
dvc remote rename
# Change a defalut remote
dvc remote default # we get the name of the current default remote
dvc remote default <name> # new default remote
```

The `sample.csv.dvc` has content of the following form:

```
outs:
- md5: 82c893581e57f7e84418cc44e9c0a3d0
  size: 3856
  path: sample.csv
```

### 3.3 Remote Storage

The true potential of DVC is unlocked when we use remote storage; then, we can simply `git clone` any repository anywhere and push/pull remote datasets/models from GDrive, S3, or similar. Therefore, we can develop on one machine a deploy on another one without any issues, because the datasets and models are remotely stored.

Example:

```bash
dvc remote add s3remote s3://a/remote/bucket
```

Note: We can have multiple remote stores!

Links:

- [Data and Model Access](https://dvc.org/doc/start/data-management/data-and-model-access)
- [Supported Storage Types](https://dvc.org/doc/command-reference/remote/add#supported-storage-types)

#### Example: GDrive Remote Storage

```python
# To work with remote GDrive folders, we need the unique identifier,
# ie., the last long token in the URL after the last slash '/'
# Unique identifier: <UNIQUEID>
# Additionally, dvc-gdrive must be installed (see above)
dvc remote add driveremote gdrive://<UNIQUEID>
dvc remote modify driveremote gdrive_acknowledge_abuse true
# Check how config file changes
less .dvc/config

# Push to the gdrive remote
# Since the local remote is the default,
# we need to specify the new drive remote
dvc push --remote driveremote
# We open the prompted URL and log in to Google
# or are redirected to a log in window.
# If a verification code is given on the web
# we paste it back on the terminal
# Now, in GDrive, we should see version files

# If we do dvc push,
# it pushes to the default remote,
# which is usually the local remote! 
dvc push

# We can change the default remote
# to be a cloud storage
dvc remote list # we get the list of all remotes: localremote, driveremote
dvc remote default # we get the name of the current default remote
dvc remote default driveremote # new default remote
```

### 3.4 Pipelines with DVC

DVC also handles Direct Acyclic Graph (DAG) pipelines, like MLflow. In such pipelines, the output of a stage is the input of a next stage.

To that end, we need to create:

- `dvc.yaml`: where the stages or components of the pipeline are defined. This file is generated when we execute a stage with `dvc run`, as shown below.
- `params.yaml`: where the parameters are defined. We need to manually define this file manually; then, we can use the parameters defined in it as follows:

```python
import yaml
from yaml import CLoader as Loader

with open("./params.yaml", "rb") as f:
    params = yaml.load(f, Loader=Loader)

my_param = params["my_param"]
```

To create a pipeline stage, we need to execute `dvc run` as follows:

```bash
dvc run -n  clean_data \ # stage name, as defined in dvc.yaml
            -p param \ # parameter param defined in params.yaml
            -d clean.py -d data/data.csv \ # dependencies: files required to run stage
            -o data/clean_data.csv \ # output artifact directory + name
            python clean.py data/data.csv # command to execute the stage

# This command will create the dvc.yaml file
# with the stage clean_data
```

The only files we need to version control are `dvc.yaml` and `params.yaml`. With them, we have the pipeline, and we can track and reproduce it.

```bash
git add dvc.yaml params.yaml
git commit -m "adding dvc pipeline files"
```

Important links:

- [Creating pipelines](https://dvc.org/doc/start/data-management/data-pipelines)
- [Running pipelines](https://dvc.org/doc/command-reference/run)

#### Exercise: Defining a Pipeline

In the folder [`./lab/dvc_test`](./lab/dvc_test), the following files are added:

- `fake_data.csv`: fake dataset.
- `prepare.py`: it prepares the fake dataset by scaling the features.
- `train.py`: it trains a logistic regression model with the prepared fake dataset.

The exercise consists in 

1. creating the two stages of the pipeline using DVC
2. and reading the hyperparameters used in `train.py` from `params.yaml`.

In the following, the file contents are provided.

`fake_data.csv`

```
feature,label
3,0
4,0
5,0
7,1
8,1
1,0
9,1
10,1
2,0
2,0
3,0
4,1
1,0
0,1
8,1
10,0
7,1
6,1
5,0
```

Python files:

```python
### prepare.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./fake_data.csv")

X = df["feature"].values
y = df["label"].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1))
print(X)

np.savetxt("X.csv", X)
np.savetxt("y.csv", y)

### train.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X = np.loadtxt("X.csv")
y = np.loadtxt("y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)

lr = LogisticRegression(C=1.0)
lr.fit(X_train.reshape(-1, 1), y_train)

preds = lr.predict(X_test.reshape(-1, 1))
f1 = f1_score(y_test, preds)
print(f"F1 score: {f1:.4f}")

```

##### Solution

First, after creating the files, we need to properly 

```bash
# Create the files
# and version-control them properly
git add prepare.py train.py
dvc add fake_data.csv # fake_data.csv.dvc is created
git add fake_data.csv.dvc .gitignore # we ignore fake_data.csv
git commit -m "adding first version"
dvc remote default # make sure which default remote we're using
dvc push
```

Then, we create `param.yaml`:

```
C: 1.0
```

And modify `train.py`:

```python
import yaml
from yaml import CLoader as Loader
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

with open("./params.yaml", "rb") as f:
    params = yaml.load(f, Loader=Loader)

X = np.loadtxt("X.csv")
y = np.loadtxt("y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)

lr = LogisticRegression(C=params["C"])
lr.fit(X_train.reshape(-1, 1), y_train)

preds = lr.predict(X_test.reshape(-1, 1))
f1 = f1_score(y_test, preds)
print(f"F1 score: {f1:.4f}")
```

Finally, we create/run the pipeline stages as follows:

```bash
# Version-control params.yaml
git add params.yaml train.py
git commit -m "add params + use it ni train.py"

# Stage 1: prepare
# The outputs are automatically added to .gitignore
dvc run -n prepare \
        -d fake_data.csv -d prepare.py \
        -o X.csv -o y.csv \
        python ./prepare.py

# Stage 2: train, with the modified script
dvc run -n train \
        -d X.csv -d y.csv -d train.py \
        -p C \ # NOTE we're using the param C from params.yaml
        python ./train.py

# The first command generates dvc.yaml & dvc.lock
# and the second updates them
# We need to track those files
git add dvc.lock dvc.yaml
# The outputs are added to .gitignore;
# they don't have a .dvc file, but that's ok
# because they stage outputs!
# Then, we push
dvc push
```

### 3.5 Experiment Tracking with DVC

We can run several experiments with varied hyperparameters using DVC. To that end, first we need to define an `evaluate` stage which uses a model pickle which was the output of the stage `train`. The `evaluate` stage creates a metric `validation.json`, which is the output of our validation/evaluation script. The stage generation command is the following:

```bash
dvc run -n evaluate \
        -d validate.py -d model.pkl \
        -M validation.json \
        python validate.py model.pkl validation.json

# To see the metrics we can either open validation.json
# or execute the following command
dvc metrics show
```

We can use the same strategy to output plots as metrics, too. In order to dump JSON files:

```python
import json

metrics = dict()
metrics['m1'] = 0.99 # e.g., accuracy
metrics['m2'] = 0.79 # e.g., F1

with open ('./validation.json') as f:
    json.dump(metrics, f)
```

Then, to run **experiments**, we execute the following:

```bash
# param is a generic parameter defined in params.yaml
# which is presumably used in the train stage.
# Thus, this command executes the complete pipeline with
# the manually defined param value and
# outputs the metric associated to the 
dvc exp run --set-param param=100
```

Each experiment is given a unique name to ultimately choose the best one; we commit the best experiment. To compare experiments:

```bash
dvc exp diff
dvc exp show # a nice table is shown
```

Important links:

- [DVC Metrics, parameters, plots](https://dvc.org/doc/start/data-management/metrics-parameters-plots)
- [DVC Experiments](https://dvc.org/doc/start/experiment-management/experiments)

## 4. CI/CD: Continuous Integration and Continuous Deployment

There are many software engineering principles, like the SOLID principles:

- Single responsibility: every class/function should do only one thing.
- Open-closed: software should be open to extension but closed to modification.
- Liskov substitution: functions that use pointers to base classes should be able to use derived classes.
- Interface segregation: clients should not be forced to depend upon interfaces that they don't use.
- Dependency inversion: depend upon abstractions, not concretions.

In this section, the following principles are worked:

1. Automation
2. Testing, specifically, how it applies to automation
3. Versioning, specifically, how it applies to machine learning

Thanks to them, the following is achieved:

- **Continuous Integration**, using Github Actions
- **Continuous Deployment**, using Heroku

### 4.1 Software Engineering Principles: Automation, Testing, Versioning

#### Automation

Everything that can be automated, should be, because we safe time!

The principle of **Don't Repeat Yourself (DRY)** is a subset of the **Automation** principle.

Examples:

- Code formatting: [black](https://black.readthedocs.io/en/stable/)
- Pre-commit hooks: for instance, we set up that our code formatter runs prior to committing, every time we execute `git commit`
- Editor features: for instance, remove EOL (end of line) symbol from scripts.

Links on git hooks:

- [Git Hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
- [`https://pre-commit.com/`](https://pre-commit.com/)

#### Testing - in the Context of Machine Learning

Testing is like using seat belts.

Tests:

- Ensure that the code behaves as expected.
- Ensure that changes in the code don't alter its behavior unexpectedly.
- Build trust in our code.

In machine learning, tests can be:

- deterministic: e.g., number of features
- indeterministic / stochastic: e.g, mean of a feature

Interesting links:

- [Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/)
- [How to Trust Your Deep Learning Code](https://krokotsch.eu/posts/deep-learning-unit-tests/)
- [Unit Testing for Data Scientists](https://towardsdatascience.com/unit-testing-for-data-scientists-dc5e0cd397fb)
- [Great Expectations: Tools for Testing ML Datasets](https://greatexpectations.io/)

#### Versioning

Versioning is essential to track compatibilities and dependencies; necessary if we work in a team.

Semantic versioning: `Major.Minor.Patch`:

- Major: API changes
- Minor: backward compatible
- Patch: bugs fixed, minor features added

### 4.2 Continuous Integration and Continuous Delivery: Definition

Practical definitions:

- Continuous Integration = Automated Testing. In other words, we make sure that any change we implement can be integrated in the code base without breaking it, i.e., the new implementations are integrable. CI makes possible to deploy our code any time, i.e., continuous deployment!
- Continuous Deployment = Deploy code/applications verified by CI automatically, without time gaps from the implementation integration. That way, the latest version of an app is always available to the users.

### 4.3 Continuous Integration with Github Actions

See this file:

[`github_ci_cd_actions_howto.md`](https://github.com/mxagar/cicd_guide/blob/master/github_ci_cd_actions_howto.md)

### 4.4 Continuous Deployment with Heroku

#### Introduction to Heroku

Heroku is a cloud Platform-as-a-Service (PaaS). See my brief notes on [cloud computing](https://github.com/mxagar/deep_learning_udacity/blob/main/06_Deployment/DLND_Deployment.md#12-cloud-computing) service model types:

- Software as a Service (SaaS): Google Docs, GMail; as opposed to software as a product, in SaaS the application is on the cloud and we access it via browser. The user has the unique responsibility of the login and the administration of the application and the content.
- Platform as a Service (PaaS): Heroku; we can use PaaS to e-commerce websites, deploy an app which is reachable via web or a REST API, etc. usually, easy deployments at the application level are done. Obviously, the user that deploys the application has more responsibilities.
- Infrastructure as a Service (IaaS): AWS EC2; they offer virtual machines on which the user needs to do everything: virtual machine provisioning, networking, app deployment, etc.

Some alternatives to Heroku, which has become now a paid service:

- Elastic Beanstalk (AWS)
- Digital Ocean App Platform
- [Render](https://render.com/): there is a section on it below: [4.5 Deployment to Render Cloud Platform](#45-deployment-to-render-cloud-platform); it's suggested by Udacity as an alternative to Heroku since Heroku removed the free tier (on November 28, 2022).

Heroku in a nutshell:

- Heroku has two important elements tha are referred constantly:
  - **Dyno**: a lightweight container where the app is deployed; these containers can easily scale. There are several dyno types: eco, basic, standard, etc.
  - **Slug**: the complete app (pipeline, etc.) an its dependencies; usually there's a limit of 500 MB, but we can leverage dvc to downloaded pipelines/artifacts.
- You need at least the **Eco subscription** (5 USD/month for 1000h Dyno hours/month) and you get 1,000 dyno hours; eco dynos sleep when inactive.
- In this section, we'll use 1 web dyno to run the API.
- A `Procfile` contains the instructions to run the app, e.g.: `web: uvicorn main:app`
  - `web`: dyno type configuration
  - `uvicorn`: command to run; [Uvicorn](https://www.uvicorn.org/) is a python-based web server package, i.e., we create a web server which runs the application we want, specified below
  - `main:app`: the app to execute has the name `app` and is located in `main.py`
- **IMPORTANT**: since we run everything in dyno containers, everything works under the principles of containerization:
  - Statelessness: no data stored or cached; if we want to access/save anything, we need to connect to external services.
  - Processes are disposable: they can be started/stopped at any time. That enables fault tolerance, rapid inteation and scalability.
  - Heroku breaks the app lifecycle into 3 stages: **build, release, run**; we're always in one of the three.

Relevant links:

- [Getting Started on Heroku with Python](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Dynos and the Dyno Manager](https://devcenter.heroku.com/articles/dynos)
- [Dyno Types](https://devcenter.heroku.com/articles/dyno-types)

#### Continuous Deployment to Heroku: Demo

In this section, I started creating the repository [vanilla_deployments](https://github.com/mxagar/vanilla_deployments) and its associated Heroku app.

Basic notions of CD in Heroku:

- Heroku has multiple deployment options: Git (Github or Heroku Git), Docker; we'll use Github here.
- Interestingly, we can perform continuous deployment if we use the Git option: we connect our git repo to Heroku and we have a continuous delivery.
- Furthermore, we can couple our CI (Github Actions) with the CD (Deployment to Heroku): we can specify that the deployment occurs only when tests are successful.
- Be mindful of the **slug** limits = The app and all its dependencies:
  - It has a limit of 500 MB: code, pipeline, data, everything needs to fit in that limit!
  - We can trim unnecessary files by adding them to the file `.slugignore`
  - However, we can use dvc to download larger data files

To create a slug/app, we can work with the CLI or using the GUI.
Example pf creating a **slug/app** with the GUI and the repository [vanilla_deployments](https://github.com/mxagar/vanilla_deployments):

- Log in to Heroku (we need to have an account and a subscription)
- New: Create new app (Europe / US)
- Select name, e.g., `vanilla-deployment-test`
- Deployment method: we choose `Github`
  - Connect to Github, authorize / grant access
  - Select repository: `vanilla_deployments`
  - Connect
- Automatic deploys
  - Choose branch: `main` (or another)
  - Wait for CI to pass before deploy
  - Enable automatic deploys: every time we push to the branch and tests pass, the code is deployed on Heroku
- Then, the app will be deployed automatically right away; however, note that:
  - If we have no `Procfile`, the slug will be deployed, but nothing will happen (a default blank app is launched).
  - If we have no executable code, nothing will happen (a default blank app is launched).

We can **check the Heroku app in the web GUI**: We select the app in the main dashboard and:

- Open App: App management dashboard is shown
- More > View logs: logs of deployment are shown
  - If we have no `Procfile`, the slug will be deployed, but nothing will happen (a default blank app is launched); the logs will reflect that

**Apps vs. Pipelines:**

- A Pipeline is a group of apps that run the same code.
- A pipeline has four stages:
  - Development
  - Review
  - Staging
  - Production
- Main use case for a pipeline: throughly test an app before deployment. Example:
  - > A developer creates a pull request to make a change to the codebase.
  - > Heroku automatically creates a review app for the pull request, allowing developers to test the change.
  - > When the change is ready, it’s merged into the codebase’s master branch.
  - > The master branch is automatically deployed to the pipeline’s staging app for further testing.
  - > When the change is ready, a developer promotes the staging app to production, making it available to the app’s end users.
- The idea is that we automate all this process; e.g., when pushed to the master branch, a stage app is created, etc.

Relevant links:

- [Heroku Limits](https://devcenter.heroku.com/articles/limits)
- [Slug Compiler](https://devcenter.heroku.com/articles/slug-compiler)
- [Heroku Pipelines vs. Apps](https://devcenter.heroku.com/articles/pipelines)

### 4.5 Deployment to Render Cloud Platform

Heroku removed the free tier in November 28, 2022. Therefore, Udacity suggested an alternative to Heroku: [Render](https://render.com/). Render offers many services:

- Static websites
- Web services
- PostgreSQL
- Cron Jobs
- etc.

This section is a tutorial on how to deploy a PostgreSQL web app using Render. The tutorial uses the following example repository from Udacity: [render-cloud-example](https://github.com/udacity/render-cloud-example). I have copied the example to the module exercise repository, too: [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos).

For now, I leave the tutorial in the following PDF, without summarizing it here: [`Udacity_Render_Tutorial.pdf`](Udacity_Render_Tutorial.pdf).

## 5. API Deployment with FastAPI

Topics covered:

- Review of Python Type Hints
- Fundamentals of FastAPI: many FastAPI features rely on type-hints
- Local API Testing
- Heroku Revisited: Deployment of our REST API app
- Live API Testing

### 5.1 Review of Python Type Hints

Python is dynamically typed, i.e., the variable types are inferred during runtime. However, we can use type hints, which make possible applications that need to know the variable types before execution. FastAPI is one of them. However, note that type hints are hints, not declarations! Thus, no type enforcing/checking is done with the hints

```python
from typing import Union

## Basic structure
# a: Union[list,str]: a should be a string or a list; e.g., a path or many paths
# b: int = 5: b is an integer with default value 5
# -> str: foo returns a string
def foo(a: Union[list,str], b: int = 5) -> str:
    pass

## Example
def greeting(name: str) -> str:
    return 'Hello ' + name

## Type aliases
Vector = list[float]

def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

# passes type checking; a list of floats qualifies as a Vector.
new_vector = scale(2.0, [1.0, -4.2, 5.4])

## Complex type signatures with aliases

from collections.abc import Sequence

ConnectionOptions = dict[str, str]
Address = tuple[str, int]
Server = tuple[Address, ConnectionOptions]

def broadcast_message(message: str, servers: Sequence[Server]) -> None:
    pass

## New types

from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)

def get_user_name(user_id: UserId) -> str:
    pass


## User-defined classes

from typing import TypeVar, Generic
from logging import Logger

T = TypeVar('T')

# Generic[T] as a base class defines that the class LoggedVar
# takes a single type parameter T .
# This also makes T valid as a type within the class body.
# Note also that we use the class name Logger as type!
class LoggedVar(Generic[T]):
    def __init__(self, value: T, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.value = value

```

More on type hints:

- [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
- [Support for type hints](https://docs.python.org/3/library/typing.html)


### 5.2 Fundamentals of FastAPI

FastAPI creates REST APIs very fast using type hints; it's fast for development and execution.

FastAPI is only for APIs, in contrast to Flask; thus, it's optimized for that application.

#### Basic Example

Example of a basic implementation in `main.py`:

```python
from fastapi import FastAPI

# Instantiate the app.
# The app is an instantiation of FastAPI class,
# which is our application that contains the API
app = FastAPI()

# Define a GET on the specified endpoint.
# If we open the browser at localhost:8000
# we'll get {"greeting": "Hello World!"}
@app.get("/") # root domain: /
async def say_hello():
    return {"greeting": "Hello World!"}
```

To run the app, start the `uvicorn` web server:

```bash
uvicorn main:app --reload

# uvicorn: lightweight and fast web server
# main:app: start the FastAPI application called app located in main.py
# --reload: regenerate the API if main.py is modified
```

Then, open a browser and go to any of the defined endpoints; at the moment, there's only one endpoint:

```bash
# localhost, port 8000
http://127.0.0.1:8000

# we get {"greeting": "Hello World!"}
```

#### Core Features of FastAPI

- FastAPI uses type hints and also the [pydantic](https://docs.pydantic.dev/) package:
  - Type hints allow to automatically create documentation
  - Pydantic is used for defining data models, which are used to define request bodies with built-in parsing and validation
- Default API documentation is in `http://127.0.0.1:8000/docs`
- Using pydantic and type hints, API URL path parameters are seamlessly converted from URL strings to the desired types:
  - The API URL path structure is defined in the decorator.
  - The backend python function that follows the decorator catches the parsed parameters defined in the decorator and uses them as we want inside the function.
  - To create optional query parameters use `Optional` from the `typing` module.

A simple `main.py` which creates a FastAPI app using pydantic data models is the following:

```python
# Import Union since our Item object
# will have tags that can be strings or a list.
from typing import Union 

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object (class) with its components and their type.
# For that, we derive from the pydantic BaseModel.
# This is what Pydantic is for.
# Note: we can also use default values, as done with type hints!
class TaggedItem(BaseModel):
    # These key names will be used in the JSON/dict generated later
    name: str
    tags: Union[str, list] # a string or a list
    item_id: int

app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
# Note we are creating a POST method, in contrast
# to the previous GET.
# The endpoint is http://127.0.0.1:8000/items
# This piece of code converts the HTML body into a JSON
# with the structure defined in TaggedItem!
# Also, the pydantic data model definitions
# will be used to geenrate automatically the API documentation in
# http://127.0.0.1:8000/docs
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item

# A GET (query) that in this case just returns the item_id we pass, 
# but a future iteration may link the item_id here
# to the one we defined in our TaggedItem.
@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}
# Notes:
# 1. Parameters not declared in the path
# are automatically query parameters: count
# 2. Parameters with curly braces in the path
# are expected as parameters in the function
# definition, and FastAPI takes care of converting
# the path element to the funtion parameter.
# 3. If we want to make quer parameters optional
# we can use the Optional typing module
# 
# Example above:
# http://127.0.0.1:8000/items/42/?count=1
# This returns {"fetch": "Fetched 1 of 42"}
```

More on FastAPI and REST APIs:

- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Best practices for REST API design](https://stackoverflow.blog/2020/03/02/best-practices-for-rest-api-design/)
- [Pydantic schema](https://fastapi.tiangolo.com/tutorial/schema-extra-example/)


#### Example/Demo Using the App Above

This demo is in the reporsitory [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos), folder `core_features_of_fast_api`. That folder contains the following files:

- `main.py` (same code as above)
- `sample_request.py` (content added below)

To start the REST API app:

```bash
# Start uvicorn web server
# running app from main.py
# and reload if we change our code
uvicorn main:app --reload
```

Then, we can play with the API app using the browser:

- `http://127.0.0.1:8000/docs`: documentation; **we can test the endpoints here!**
- `http://127.0.0.1:8000/items/42/?count=2`: we get `{"fetch": "Fetched 2 of 42"}`
- `http://127.0.0.1:8000/items`: error, because it tries to POST without an input JSON. To use this endpoint, we can employ `sample_request.py`, which uses the `resquest` module to POST/send a JSON object to the API endpoint. Another option would be to use `curl`, as shown in the documentation.

The content of `sample_request.py`:

```python
import requests
import json

data = {"name": "Hitchhiking Kit",
        "tags": ["book", "towel"],
        "item_id": 23}

r = requests.post("http://127.0.0.1:8000/items/", data=json.dumps(data))

print(r.json())
# Note that the dictionary keys are the ones used in the pydantic BaseModel
# {'name': 'Hitchhiking Kit', 'tags': ['book', 'towel'], 'item_id': 23}
```

To use the POST method with `sample_request.py`, we run

```bash
python sample_request.py
```

#### Example/Demo: Variable Paths and Testing

This demo is in the reporsitory [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos), folder `params_inputs_FastAPI`. That folder contains the following files:

- `bar.py`
- `test_bar.py`

... with the following content:

```python
## bar.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Value(BaseModel):
    value: int

# Use POST action to send data to the server
# path: user defined path value
# query: query value
# body: data sent in the post, body; type is Value
@app.post("/{path}")
async def exercise_function(path: int, query: int, body: Value):
    return {"path": path, "query": query, "body": body}

## test_bar.py

import json
from fastapi.testclient import TestClient

# We import the app from bar.py
from bar import app

# Test client from FastAPI
client = TestClient(app)

def test_post():
    data = json.dumps({"value": 10})
    r = client.post("/42?query=5", data=data)
    print(r.json())
    # {'path': 42, 'query': 5, 'body': {'value': 10}}
    assert r.json()["path"] == 42
    assert r.json()["query"] == 5
    assert r.json()["body"] == {"value": 10}

# We don't really need the below function
# invokation, because we can use pytest!
# FastAPI TestClient works with pytest!
'''
if __name__ == '__main__':

    test_post()
'''
```

To run everything:

```bash
# To start the web server with the REST API app
uvicorn bar:app --reload

# In another terminal session, we can test.
# HOWEVER, we don't need the server running
# to use the FastAPI TestClient!
# That makes possible to perform integration tests
# before deployment!
pytest
```

### 5.3 Local API Testing

As introduced in the example above, we can write tests using the FastAPI `TestClient` class in a `test_app.py` file and run them using `pytest`. That means we don't need to start the web server with `uvicorn` for testing.

The `TestClient` behaves like the `requests` module which is interacting with a web server.

That integration with `pytest` and the functionality offered by `TestClient` make possible to perform CI with FastAPI.

Another example:

```python
# The `TestClient` behaves like the `requests` module
# which is interacting with a web server
from fastapi.testclient import TestClient

# Import our app from main.py
from main import app

# Instantiate the testing client with our app
client = TestClient(app)

# Write tests using the same syntax
# as with the requests module
def test_api_locally_get_root():
    r = client.get("/")
    # 200: successful
    assert r.status_code == 200
```

Interesting links:

- [FastAPI Testing Tutorial](https://fastapi.tiangolo.com/tutorial/testing/)

#### Exercise: Testing Example

This exercise is in the reporsitory [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos), folder `local_api_testing`. That folder contains the following files:

- `foo.py`
- `test_foo.py`


```python
### foo.py

from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}

### test_foo.py
from fastapi.testclient import TestClient

# Import the app from foo.py
from foo import app

# Instantiate the TestClient with the app
client = TestClient(app)

# Recommendation: write a test for each
# endpoint usage case;
# that facilitates rapid identification of what exactly
# is failing when a test breaks
def test_get_path():
    r = client.get("/items/42")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 42"}

def test_get_path_query():
    r = client.get("/items/42?count=5")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 5 of 42"}

def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200

```

To test it:

```bash
pytest
```

### 5.4 Heroku Revisited: Procfiles and CLI

Since we run everything in dyno containers in Heroku, everything works under the principles of containerization:

- Statelessness: no data stored or cached; if we want to access/save anything, we need to connect to external services.
- Processes are disposable: they can be started/stopped at any time. That enables fault tolerance, rapid inteation and scalability.
- Heroku breaks the app lifecycle into 3 stages: **build, release, run**; we're always in one of the three.

We introduced a simple `Procfile`; however we can/should make it more advanced:

```bash
# Simple:
# - web: type of dyno
# - uvicorn: run web server
# - main:app: from main.py execute app
# - reload: if anything changes in the code update running instance
web: uvicorn main:app --reload

# Advanced: we specify
# - host 0.0.0.0: listen on any network interface
# - port: system specified in PORT, and if not define, port 5000 used
# In Unix ${VARIABLE:-default} is used to assigna ddefault value
# if VARIABLE is not set
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
```

![Heroku Procfile](./pics/heroku_procfile.jpg)

Interesting links:

- [Heroku's Runtime Principles](https://devcenter.heroku.com/articles/runtime-principles)
- [Difference between 127.0.0.1 and 0.0.0.0](https://superuser.com/questions/949428/whats-the-difference-between-127-0-0-1-and-0-0-0-0/949522#949522)

#### Heroku CLI

To install the Heroku CLI, follow the [instructions](https://devcenter.heroku.com/articles/heroku-cli):

```bash
# Mac
brew tap heroku/brew && brew install heroku
# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

Then, we can check that `heroku` was installed and log in:

```bash
# Check heroku is installed by requesting the version
heroku --version
# Log in: browser is opened and we log in
heroku login
# Log in: credentials entered via CLI
heroku login -i
```

The following commands show how to use the Heroku CLI (assuming we're logged in). I have the app files in the local copy of the repository [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos), folder `heroku_cli_demo`:

`~/git_repositories/mlops-udacity-deployment-demos/heroku_cli_demo`

But the files are not committed to that repo; instead, a new repo is created and connected to Heroku.

This app creation replicates approximately what is done in [Section 4.4](#44-continuous-deployment-with-heroku) using the GUI, but this time via the CLI. Another difference is that we don' link an existing Github repository, but link a local git repository to the Heroku git. After executing what comes in the following, we can open the Heroku web URL and interact with the app deployed on that endpoint.

```bash

cd /path/to/project # go to local project folder
cd .../heroku_cli_demo
heroku # list of all possible commands

# Create a Heroku app via the CLI
# We specify it's a python app, but that's not necessary
# becuse Heroku will detect it automatically.
# In the case of python apps, it looks for
# requirements.txt and setup.py
heroku create <name-of-the-app> --buildpack heroku/python
heroku create heroku-cli-demo --buildpack heroku/python
# https://heroku-cli-demo.herokuapp.com/ | https://git.heroku.com/heroku-cli-demo.git

# List all our apps
heroku apps

# View our app buildpacks
heroku buildpacks --app <name-of-the-app>
heroku buildpacks --app heroku-cli-demo # heroku/python

# We should have/create in our folder the following files:
# - Procfile (example content above)
# - requirements.txt (uvicorn, fastapi, requests)
# - runtime.txt (if we'd like to specify a concrete Python version)
# - main.py (FastAPI app)
# - test_main.py (optional)
# If we're not in a git repo, we can start one.
git init
git add *
git commit -m "initial commit"

# Connect the local repo to heroku git
heroku git:remote --app <name-of-the-app>
heroku git:remote --app heroku-cli-demo
# set git remote heroku to https://git.heroku.com/heroku-cli-demo.git

# Push our branch to main
# This automatically will launch our app!
# After pushing, the app is executed
git push heroku main

# We can enter the app's virtual machine/container
heroku run bash --app <name-of-the-app>
heroku run bash --app heroku-cli-demo
# Now we're inside and can run bash commands
# pwd: /app
# ls: we see our files: main.py, Procfile, etc.
# vi
# ...
exit # exit the heroku app container

# On the Heroku web interface
# we select the app and go to its settings
# There we see in the Domains section the endpoint URL
#   https://heroku-cli-demo.herokuapp.com
# Note that we can use our own domains!
# We can interact with it as we have defined: we open a browser and insert
#   https://heroku-cli-demo.herokuapp.com/items/42
# We get
#   {"fetch":"Fetched 1 of 42"}

# If smoething goes wrong
heroku logs --tail

# We can see whether the app is running on the web GUI
# dashboard:
# - The hexagon is darker
# - If we click on the app, we see ON in the Overview Dyno
# If we want to SHUT DOWN the app, we can either
# 1. Do it via the web GUI: select app, Resources:
# click on the  command pencil and switch it OFF
# 2. Or via the CLI
heroku ps:scale web=0 --app <name-of-the-app>
heroku ps:scale web=0 --app heroku-cli-demo

# If we want to RESTART the app, we can either 
# 1. Do it via the web GUI: select app, Resources:
# click on the command pencil and switch it ON
# 2. Or via the CLI
heroku ps:scale web=1 --app <name-of-the-app>
heroku ps:scale web=1 --app heroku-cli-demo

```

Interesting links:

- [Heroku CLI Plugins](https://devcenter.heroku.com/articles/heroku-cli#useful-cli-plugins)
- [Heroku CLI commands](https://devcenter.heroku.com/articles/heroku-cli-commands)

### 5.5 Live API Testing with Requests Package

Interacting with REST APIs consists in interacting with HTTP methods: POST, GET, DELETE, PUT ([CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete)).

Fortunately, Python has a very useful package `requests` which simplifies that.

```python
import requests
import json

# POST: send a package to endpoint URL
# In this example we don't send any data
response = requests.post('/url/to/query')

# POST: send a package to endpoint URL
# In this example (1) we perform authentification
# (2) and we send a JSON of a dictionary (it could be a list, too,
# or a variable)
data = {'a': 1, 'b': 2}
response = requests.post('/url/to/query/',
                         auth=('usr', 'pass'),
                         data=json.dumps(data))

# Now we print the response
print(response.status_code) # 200 means everything OK
print(response.json()) # 

```

Interesting links:

- [Requests Documentation](https://requests.readthedocs.io/en/latest/)
- [List of public APIs](https://github.com/public-apis/public-apis)
- [On Authentication: API Keys vs. OAuth (Access Token)](https://stackoverflow.com/questions/38047736/oauth-access-token-vs-api-key)

#### Exercise/Demo: Data Ingestion and Error Handling with FastAPI

This exercise/demo is not deployed, but is run locally. It can be found in the repository [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos), folder `API_Deployment_with_FastAPI_FinalExercise`.

The folder contains two files:

- `main.py`
- `test_main.py`

The task is the following:

>  Create an API that implements a POST method called `ingest_data` that accepts this body parameter:
>
> ```
> from pydantic import BaseModel
> 
> class Data(BaseModel):
>     feature_1: float
>     feature_2: str
> ```
>
> Imagine that this data is going to be used in a machine learning model. We want to make sure that the data is in the ranges we expect so our model doesn't output bogus results. To accomplish this, use `raise` with the `HTTPException` that is included with FastAPI. Your exception should raise an appropriate status code and tell the users what the issue is. For `feature_1` assure that the input is non-negative. For `feature_2` assure that there are 280 characters or less.
> 
> Then, use FastAPI's test client to ensure that your API gives the expected status code when it fails.
> 
> Lastly, add some metadata to your API including a name, brief description, and a version number.
> 
> Hint: FastAPI includes `HTTPException`. 
> 
> For more information on FastAPI's error handling see [here](https://fastapi.tiangolo.com/tutorial/handling-errors/) and for metadata see [here](https://fastapi.tiangolo.com/tutorial/metadata/).

Solution:

```python
### main.py

from typing import Union, List
# We can aise HTML exceptions with fastapi.HTTPException
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

# Data structure definition
class Data(BaseModel):
    feature_1: float
    feature_2: str

# FastAPI app: 
app = FastAPI(
    title="Exercise API",
    description="An API that demonstrates checking the values of your inputs.",
    version="1.0.0",
)

# POST method that checks the input data
@app.post("/data/")
async def ingest_data(data: Data):
    # feature_1 must be positive
    if data.feature_1 < 0:
        raise HTTPException(status_code=400, detail="feature_1 needs to be above 0.")
    # feature_2 must be larger shorter than 280
    if len(data.feature_2) > 280:
        raise HTTPException(
            status_code=400,
            detail=f"feature_2 needs to be less than 281 characters. It has {len(data.feature_2)}.",
        )
    # If everything correct, return the same input data
    return data

# To run this:
#   uvicorn main:app --reload
# To test:
#   pytest

### test_main.py

import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# This will pass
def test_post_data_success():
    data = {"feature_1": 1, "feature_2": "test string"}
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 200

# This will pass, but note that we check we get a 400 status code (error)
def test_post_data_fail():
    data = {"feature_1": -5, "feature_2": "test string"}
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 400

# To run this:
# pytest

```

Interesting links:

- [Error handling with FastAPI](https://fastapi.tiangolo.com/tutorial/handling-errors)
- [Metadata with FastAP](https://fastapi.tiangolo.com/tutorial/metadata/)

### 5.6 Extra

## 6. Project: Deploying a Model with FastAPI to Heroku

See project repository: [census_model_deployment_fastapi](https://github.com/mxagar/census_model_deployment_fastapi).

## 7. Excurs: Docker and AWS EC2

## 8. Summary Project: Vanilla Deployments

I have create a repository which summarizes the usage different techniques and technologies for deployment of machine learning models/pipelines.

See project repository: [vanilla_deployments](https://github.com/mxagar/vanilla_deployments).