# Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree

These are my notes of the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

The nanodegree is composed of four modules:

1. Clean Code Principles
2. Building a Reproducible Model Workflow
3. Deploying a Scalable ML Pipeline in Production
4. ML Model Scoring and Monitoring

Each module has a folder with its respective notes. This folder and file refer to the **fourth** module: **ML Model Scoring and Monitoring**.

Mikel Sagardia, 2022.  
No guarantees.

## Overview of Contents

- [Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree](#machine-learning-devops-engineer-personal-notes-on-the-udacity-nanodegree)
  - [Overview of Contents](#overview-of-contents)
  - [1. Introduction to Model Scoring and Monitoring](#1-introduction-to-model-scoring-and-monitoring)

## 1. Introduction to Model Scoring and Monitoring

Why is it important to monitor ML models?

- Check model behaves correctly
- Check accuracy is enough
- Check any dependency/compatibility issues
- Check data is correctly pushed to the model
- etc.

When the model is sold to the customer, an MLOps team deploy it. However, when there are problems with the model, the ML/DS Engineer is responsible for the 2nd level support, because they know the model the best. Therefore, even though the model has been shipped, we are still its *"responsible owners"*.

With model monitoring

- We enable predictive ML Operations: we can identify and predict model drift
- We detect problems fast and solve them efficiently
- We achieve transparency for the stakeholders
- We increase productivity and result quality
- We capture insights and ideas

Module instructor: [Bradford Tuckfield](https://bradfordtuckfield.com/); he has a company: [https://kmbara.com/](https://kmbara.com/)

