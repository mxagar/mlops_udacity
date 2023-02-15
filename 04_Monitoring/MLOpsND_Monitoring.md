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
    - [1.1 Requirements for a ML Monitoring System](#11-requirements-for-a-ml-monitoring-system)
    - [1.2 History of ML Scoring](#12-history-of-ml-scoring)
    - [1.3 Overview of the Final Project: A Dynamic Risk Assessment System](#13-overview-of-the-final-project-a-dynamic-risk-assessment-system)
  - [2. Model Re-Training and Re-Deployment](#2-model-re-training-and-re-deployment)

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

### 1.1 Requirements for a ML Monitoring System

Model scoring and monitoring is necessary every time **we deploy a model and use it continuously**.

Even though the external stakeholders are not aware of the need of monitoring, they want a system that works reliably &mdash; and that's possible only if we monitor our system. Additionally, even there exist clear roles of who does what in the industry, the responsibilities are often shared between different roles in the team (e.g., a data scientist is responsible for the monitoring infrastructure).

![ML Stakeholders](./pics/stakeholders.png)

We need to define a **continuous monitoring system** in which 

- The model is continuously scored and its predictions evaluated wrt. expected accuracy; i.e., we run **diagnostics** on the model 
- **Ingestion** of new data is enabled in the system
- The model is re-trained and re-deployed if its performance has decreased
- The complete process is tested/checked for operation success automatically
- We report (e.g., with APIs) the results

![ML Scoring and Monitoring: Needs](./pics/ml_scoreing_monitoring_needs.jpg)

Interesting links:

- [How To Know if Your Machine Learning Model Has Good Performance](https://machinelearningmastery.com/how-to-know-if-your-machine-learning-model-has-good-performance/)
- [Production Machine Learning Monitoring: Outliers, Drift, Explainers & Statistical Performance](https://towardsdatascience.com/production-machine-learning-monitoring-outliers-drift-explainers-statistical-performance-d9b1d02ac158)
- [Evaluating a machine learning model](https://www.jeremyjordan.me/evaluating-a-machine-learning-model/)
- [How to Structure a Data Science Team: Key Models and Roles to Consider](https://www.altexsoft.com/blog/datascience/how-to-structure-data-science-team-key-models-and-roles/)

### 1.2 History of ML Scoring

![ML Scoring History](./pics/ml_scoring_history.png)

### 1.3 Overview of the Final Project: A Dynamic Risk Assessment System

The project builds a model which predicts who are the customers that have high probability of leaving the services of a company, i.e., customer churn or attrition risk. This is a common business problem faced by all companies; the idea behind it is that it's easier to keep a customer than getting a new one, thus, it's very important to detect customers that might want to leave to prevent them to.

![Customer Churn](./pics/customer_churn.jpg)

However, it's not enough with the model: if the nature of the business varies a little bit (e.g., new services) or we have new customers in our pool, the efficiency of the model will change, so we need to re-train and re-deploy it.

![Project Features](./pics/project_features.jpg)

Altogether, the project builds a system with the following features:

- automated data ingestion
- checking for model drift
- retraining and re-deployment
- diagnosis of operational issues
- continuous monitoring with API's

## 2. Model Re-Training and Re-Deployment

