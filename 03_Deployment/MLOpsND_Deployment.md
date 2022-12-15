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
  - [3. Data and Model Versioning](#3-data-and-model-versioning)
  - [4. CI/CD](#4-cicd)
  - [5. API Deployment with FastAPI](#5-api-deployment-with-fastapi)

## 1. Introduction to Deployment

The module is taught by [Justin Clifford Smith, PhD](https://www.justincliffordsmith.com/). Principal data scientist at [Optum](https://www.optum.com/).

Deployment consists in putting the model to work in production. But not all deployments are the same: sometimes we want to keep learning during the usage of the model!

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



## 3. Data and Model Versioning



## 4. CI/CD



## 5. API Deployment with FastAPI


