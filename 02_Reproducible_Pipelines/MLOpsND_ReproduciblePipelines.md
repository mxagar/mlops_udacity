# Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree

These are my notes of the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

The nanodegree is composed of four modules:

1. Clean Code Principles
2. Building a Reproducible Model Workflow
3. Deploying a Scalable ML Pipeline in Production
4. ML Model Scoring and Monitoring

Each module has a folder with its respective notes. This folder and file refer to the **second** module: **Building a Reproducible Model Workflow**.

This module uses exercises from the following forked repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Final note: This module uses Weights & Biases for tracking and MLflow for pipeline management, among others. I have another guide in which I show how MLflow can be used for tracking, pipeline management and model deployment (including model registries):

[mxagar/mlflow_guide](https://github.com/mxagar/mlflow_guide)

Mikel Sagardia, 2022.  
No guarantees.


## Overview of Contents

- [Machine Learning DevOps Engineer: Personal Notes on the Udacity Nanodegree](#machine-learning-devops-engineer-personal-notes-on-the-udacity-nanodegree)
  - [Overview of Contents](#overview-of-contents)
  - [1. Introduction to Reproducible Model Workflows](#1-introduction-to-reproducible-model-workflows)
    - [1.1 Machine Learning Operations (MLOps)](#11-machine-learning-operations-mlops)
      - [What Are MLOps and Reproducible Workflows?](#what-are-mlops-and-reproducible-workflows)
    - [1.2 Business Stakeholders: Which Roles Are Important in MLOps?](#12-business-stakeholders-which-roles-are-important-in-mlops)
    - [1.3 When Should We Use MLOps?](#13-when-should-we-use-mlops)
    - [1.4 MLOps Tools Used](#14-mlops-tools-used)
      - [Installation of Weights \& Biases and MLflow](#installation-of-weights--biases-and-mlflow)
    - [1.5 Module Project: Rental Price Prediction in New York](#15-module-project-rental-price-prediction-in-new-york)
  - [2. Machine Learning Pipelines](#2-machine-learning-pipelines)
    - [2.1 The Three Levels of MLOps](#21-the-three-levels-of-mlops)
    - [2.2 Argparse](#22-argparse)
    - [2.3 Versioning Data and Artifacts in Weights and Biases](#23-versioning-data-and-artifacts-in-weights-and-biases)
    - [2.4 Weights \& Biases: Example Notebook](#24-weights--biases-example-notebook)
    - [2.5 Weights \& Biases: Exercise 1, Versioning Data \& Artifacts and Using Them](#25-weights--biases-exercise-1-versioning-data--artifacts-and-using-them)
    - [2.6 ML Pipeline Components in MLFlow](#26-ml-pipeline-components-in-mlflow)
      - [Conda: `conda.yaml`](#conda-condayaml)
      - [Project Definition: `MLproject`](#project-definition-mlproject)
      - [Running the Project](#running-the-project)
    - [2.7 Introduction to YAML](#27-introduction-to-yaml)
    - [2.8 MLflow: Exercise 2, Defining and Running an MLflow pipeline](#28-mlflow-exercise-2-defining-and-running-an-mlflow-pipeline)
      - [Solution](#solution)
    - [2.9 Linking Together the Components](#29-linking-together-the-components)
      - [Pipeline Configuration: Hydra](#pipeline-configuration-hydra)
      - [Tracking Pipelines with Weights \& Biases](#tracking-pipelines-with-weights--biases)
    - [2.10 MLflow and Hydra: Exercise 3, Parametrized Pipeline](#210-mlflow-and-hydra-exercise-3-parametrized-pipeline)
      - [Solution](#solution-1)
    - [2.11 MLflow Project Development Recommendations](#211-mlflow-project-development-recommendations)
      - [Notes, Tips \& Tricks](#notes-tips--tricks)
      - [Troubleshooting Pipelines](#troubleshooting-pipelines)
      - [Remove all conda mlflow conda environments](#remove-all-conda-mlflow-conda-environments)
    - [2.12 Conda vs. Docker](#212-conda-vs-docker)
    - [2.13 Running in the Cloud](#213-running-in-the-cloud)
  - [3. Data Exploration and Preparation](#3-data-exploration-and-preparation)
    - [3.1 Exploratory Data Analysis (EDA)](#31-exploratory-data-analysis-eda)
    - [3.2 Pandas Profiling](#32-pandas-profiling)
    - [3.3 Set Up EDA Notebook with WandB and MLflow: Exercise 4](#33-set-up-eda-notebook-with-wandb-and-mlflow-exercise-4)
      - [Solution](#solution-2)
      - [Important Issues](#important-issues)
    - [3.4 Clean and Pre-Process the Data](#34-clean-and-pre-process-the-data)
    - [3.5 Pre-Processing Script with W\&B and MLflow: Exercise 5](#35-pre-processing-script-with-wb-and-mlflow-exercise-5)
      - [Solution](#solution-3)
      - [Running the Example](#running-the-example)
    - [3.6 Data Segregation](#36-data-segregation)
    - [3.7 Data Segregation Script with W\&B and MLflow: Exercise 6](#37-data-segregation-script-with-wb-and-mlflow-exercise-6)
      - [Solution](#solution-4)
      - [Run the Exercise](#run-the-exercise)
    - [3.8 Feature Stores](#38-feature-stores)
  - [4. Data Validation](#4-data-validation)
    - [4.1 A Primer on Pytest](#41-a-primer-on-pytest)
    - [4.2 Deterministic Tests: Example / Exercise 7](#42-deterministic-tests-example--exercise-7)
      - [Exercise 7: Deterministic Data Tests](#exercise-7-deterministic-data-tests)
      - [Solution](#solution-5)
      - [Important Notes / Comments / Issues](#important-notes--comments--issues)
    - [4.3 Non-Deterministic or Statistical Tests: Example / Exercise 8](#43-non-deterministic-or-statistical-tests-example--exercise-8)
      - [Exercise 8: Non-Deterministic Data Tests](#exercise-8-non-deterministic-data-tests)
      - [Solution](#solution-6)
    - [4.4 Parameters in Pytest: Example / Exercise 9](#44-parameters-in-pytest-example--exercise-9)
      - [Exercise 9: Parametrized Test Functions](#exercise-9-parametrized-test-functions)
      - [Solution](#solution-7)
    - [4.5 Alternative Tool for Data Validation: Great Expectations](#45-alternative-tool-for-data-validation-great-expectations)
  - [5. Training, Validation and Experiment Tracking](#5-training-validation-and-experiment-tracking)
    - [5.1 The Inference Pipeline](#51-the-inference-pipeline)
    - [5.2 Inference Pipelines with Scikit-Learn and Pytorch](#52-inference-pipelines-with-scikit-learn-and-pytorch)
      - [Pipelines with Pytorch](#pipelines-with-pytorch)
      - [Example / Exercise 10](#example--exercise-10)
      - [Solution to Exercise 10](#solution-to-exercise-10)
    - [5.3 Machine Learning Experimentation](#53-machine-learning-experimentation)
    - [5.4 Experiment Tracking with Weights and Biases: A Example with Jupyter Notebook](#54-experiment-tracking-with-weights-and-biases-a-example-with-jupyter-notebook)
    - [5.5 Experiment Tracking and Hyperparameter Optimization with Weights and Biases and Hydra](#55-experiment-tracking-and-hyperparameter-optimization-with-weights-and-biases-and-hydra)
      - [Choosing the Best Performing Model](#choosing-the-best-performing-model)
      - [Example / Exercise 11: Validate and Choose the Best Performing Model](#example--exercise-11-validate-and-choose-the-best-performing-model)
      - [Solution to Exercise 11](#solution-to-exercise-11)
    - [5.6 Export the Inference Pipeline / Artifact](#56-export-the-inference-pipeline--artifact)
      - [Example / Exercise 12: Export ML Inference Pipeline with MLflow](#example--exercise-12-export-ml-inference-pipeline-with-mlflow)
      - [Solution to Example / Exercise 12](#solution-to-example--exercise-12)
    - [5.7 Testing the Final Artifact and Storing It in the Model Registry](#57-testing-the-final-artifact-and-storing-it-in-the-model-registry)
      - [Example / Exercise 13: Testing the Final Model](#example--exercise-13-testing-the-final-model)
      - [Solution to Exercise 13](#solution-to-exercise-13)
    - [5.8 Using MLflow for Experiment Tracking](#58-using-mlflow-for-experiment-tracking)
  - [6. Final Pipeline, Release and Deploy](#6-final-pipeline-release-and-deploy)
    - [6.1 Exercise 14: Write an End-to-End Machine Learning Pipeline](#61-exercise-14-write-an-end-to-end-machine-learning-pipeline)
      - [Most Important Files](#most-important-files)
    - [6.2 Releases](#62-releases)
    - [6.3 Deployments with MLflow](#63-deployments-with-mlflow)
      - [Offline/Batch Deployment](#offlinebatch-deployment)
      - [Online/Realtime Deployment](#onlinerealtime-deployment)
      - [Docker Image](#docker-image)
    - [6.4 Other Options for Deployment](#64-other-options-for-deployment)
  - [7. Final Project: Build an ML Pipeline for Short-term Rental Prices in NYC](#7-final-project-build-an-ml-pipeline-for-short-term-rental-prices-in-nyc)

## 1. Introduction to Reproducible Model Workflows

Deployed machine learning pipelines must be efficient and reliable.

To achieve that, we need to perform a successful deployment; the first thing to that end is to answer several questions:

- Data aspects: Data pipeline
    - How will be the data for training and for production collected
    - How will data be managed in the operational setting?
    - How will be the data secured?
    - Do we need to transfer the data?
    - Do we need some quality levels for the data?
    - Will the data need some preparation? If some preparation needed, how will it be addressed in the operational setting?
    - Is the data structured?
    - Data rights: Do we have access? If not, do we need some dummy data?
    - Are we considering data biasing when performing any selections?
- Model operational aspects:
    - The model must be reliable at any time.
    - Does the customer require ML model transparency?
    - How do we plan to re-train a model that is already in production?
        - Cloud deployments are easier to re-train than edge deployments.
    - Definition of Done for a Data Scientist: model working in production producing Return on Investment (ROI) for the company.
    - Data engineers and ML engineers must work hand-by-hand.
    - Customer expectations, experience, etc. is fundamental.

Goals of the module:

1. Create a clean, organized, reproducible, end-to-end machine learning pipeline from scratch.
2. Transform, validate and verify your data (avoid “garbage in, garbage out”).
3. Run experiments, track data, code, and results.
4. Select the best performing model and promote it to production.
5. Create an artifact ready for deployment, and release your final pipeline.

Review python [decorators](https://book.pythontips.com/en/latest/decorators.html?highlight=decorators).

### 1.1 Machine Learning Operations (MLOps)

These two settings are completely different:

- Academic/competition seetings: research environment is enough.
- Real-world applications: production environment is the goal.

If we are aiming for a production environment, we're going to need to deploy our model. In that case, when modelling, we need to keep in mind the following:

- Production: A model is 100% useless until it's production.
- Usability: 70% accuracy in production is infinitely better than 90% accuracy that can't be used, e.g., because the processing is too slow.
- Dependability: we need to monitor to avoid performance drift.
- Reproducibility: the process must be transparent (understandable by all stakeholders and team members) and repeatable.

![ML in the Wild](./pics/ml-in-the-wild.png)

MLOps is a quite recent disciplines, although it has its roots in the 1990's. It could be argued that it was founded with the seminal paper by Sculley et a.: *Hidden Technical Debt in Machine Learning Systems*.

![History of MLOps](./pics/mlops_history.png)


#### What Are MLOps and Reproducible Workflows?

**Machine Learning Operations (MLOps)** are a set of best practices and tools for an efficient end-to-end development and operation of performant, scalable, reliable, automated and reproducible ML solutions in a real production setting.

With MLOps many things are automated and we don't need to deal so much with the infrastructure anymore, so we have more time for the two most important things in data science:

- Data processing: cleaning, feature engineering, etc.
- Modelling; without MLOps, the time allocated for modelling would have been much smaller.

![MLOps advantage](./pics/mlops_advantage.png)

In a real-world machine learning deployment, we don't only deploy the model pipeline, but take care of many other steps:

- Data labeling
- Data storage
- ML pipeline packaging
- Tracking of experiments, code & artifacts with each model pipeline
- Repositories of models
- Inference system, which pulls the appropriate model pipeline
- Monitoring, which triggers re-training if necessary and updates the data labelling process

![Real ML Workflow](./pics/real-ml-workflow.png)

**Reproducible Workflow** is an orchestrated, tracked and versioned workflow that can be reproduced and inspected.

[Gartner Report: Top Strategic Technology Trends for 2021](https://www.gartner.com/en/newsroom/press-releases/2020-10-19-gartner-identifies-the-top-strategic-technology-trends-for-2021#:~:text=Gartner%20research%20shows%20only%2053,a%20production%2Dgrade%20AI%20pipeline.): *" only 53% of projects make it from artificial intelligence (AI) prototypes to production"*

### 1.2 Business Stakeholders: Which Roles Are Important in MLOps?

There are different roles involved in MLOps and all need to have a clear common language as well as good communication in order a project to be successful. In small companies, a person might have several hats. However, which are these roles:

1. **Data Scientists and ML Engineers**: the main people responsible for the development of the model and the performance measurement.
2. **Data Engineers**: responsible for the data ingestion pipelines and the quality of the data, as well as provisioning the right data at inference time (in production).
3. **Software or Platform Engineers**: they are responsible for the production environment: front-en and back-end. They are necessary partners to think about deployment, and what are the constraints in terms of processing power, latency, throughput, infrastructure and so on. They basically build the production environment.
4. **DevOps Engineer**: they are responsible for handling the infrastructure: training servers, the different MLops tools, and any other infrastructure needed to train and deploy a model. They basically maintain the infrastructure.
5. **Product Managers**: they **define the right problem to solve, exploiting their knowledge about customers' needs**. They keep the project on time and on budget. For them, MLops is a tool allowing for faster and more reliable deployment.
6. **Customers**: MLops and reproducible workflows are hidden from them, but they are going to notice improved reliability and a faster pace of improvements and enhancements in the product.

### 1.3 When Should We Use MLOps?

MLOps is always helpful, but not always necessary; the more complex the project is, the larger is the need and medium-term benefit of MLOps.

Rule of thumb:

- Small personal side-projects and competitions don't need MLOps.
- Initial MVPs often don't require MLOps.
- As soon as we deploy a model, we need MLOps.

### 1.4 MLOps Tools Used

Udacity has strived for simplicity, proven efficiency and availability on desktop platforms:

![MLOps Tools](./pics/mlops_tools.png)

Tools used:

- Github
- Weights & Biases
- MLflow + Hydra
- Anaconda (as oposed to Docker?)
- scikit-learn & pytorch

#### Installation of Weights & Biases and MLflow

Create an account: I used my Github account: [https://wandb.ai/mxagar](https://wandb.ai/mxagar).

Create/activate a conda environment and install the following packages:

```bash
# Create an environment
# conda create --name udacity-mlops python=3.8 mlflow jupyter pandas matplotlib requests -c conda-forge
# ... or activate an existing one:
conda activate ds
# Install missing packages
conda install mlflow requests -c conda-forge
# Make sure pip is pointing to the pip in the conda environment
which pip
# Install Weights and Biases through pip
pip install wandb
# Log in to wandb
wandb login
# Log in on browser if not done
# Go to provided URL: https://wandb.ai/authorize
# Copy and paste API key on Terminal, as requested
# Done!
```

Test `wandb` and `mlflow`:

```bash
wandb --help
mlflow --help
```

**IMPORTANT NOTE: I have another guide on MLflow with additional details: [mxagar/mlflow_guide](https://github.com/mxagar/mlflow_guide).**

### 1.5 Module Project: Rental Price Prediction in New York

> A property management company is renting rooms and properties in New York for short periods on various rental platforms. They need to estimate the typical price for a given property based on the price of similar properties. The company receives new data in bulk every week, so the model needs to be retrained with the same cadence, necessitating a reusable pipeline.

![ML project pipeline](./pics/mlops-project-pipeline.png)

## 2. Machine Learning Pipelines

A machine learning pipeline is a sequence of independent, modular and reusable components. It can be represented as a Direct Acyclic Graph (DAG) in which the output artifact of a component is the input for the next one or another component.

Artifacts are ouput files of objects that need to be

- tracked (who did what when),
- versioned,
- stored.

Example 1: **ETL pipeline** = Extract, Transform, Load: it ingests the data from various sources, aggregates and cleans it and stores it in a database.

![ETL pipeline](./pics/etl-pipeline.png)

Example 2: **Training Pipeline** = It takes the raw dataset and produces an inference artifact, which is more than the model, rather the pipeline. The ETL pipeline is part of it. The artifact is stored in a registry.

![ETL pipeline](./pics/ml-pipeline.png)

In this section, tools for creating machine learning pipelines are introduced, with examples.

The tools are:

- Weights and Biases, `wandb`: it tracks pipeline runs as wells as artifacts generated in the different steps. We can upload and download artifact versions.
- MLflow, `mlflow`: each step in the pipeline is a mini-project with a well defined conda/docker environment which executes a script or code file in any language; example steps are: get the data, process the data, train, etc. With MLflow we can define sequences of steps with their parameters; additionally, in combination with `wandb` we can generate and track artifacts which glue the different steps of the pipeline.
- Hydra: hydra offers an extra layer on top of MLFlow with which we can parametrize the pipeline with a configuration file. Hydra can do much more, but it's not covered in the section.

The examples are in the repository [udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises). However, I copied them to `./lab/`. In particular the exercise/example 3 is very interesting: `WandB_MLflow_Hydra_exercise_3_parametrized_pipeline`. It covers a 2-step pipeline which fetches data, uploads, downloads and processes artifacts, etc. It is like a blueprint for any large ML pipeline.

### 2.1 The Three Levels of MLOps

**Level 0**: no production, proofs of concept, competitions; often just monolithic scripts/notebooks and no pipelines.

**Level 1**: Pipelines, not models.

- The target is not just a model, but an entire pipeline; thus, we can regenerate the model with the pipeline (easy to retrain).
- The pipeline has reusable components.
- The code, artifacts, experiments are tracked.
- The model is monitored in production to avoid the model drift.
- We can learn in production.
- Everything is standardized, thus, it's easier to hand over th epipeline between teams or to try new things just by changing small parts of the modularized components.

**Level 2**: Continuous Integration, Continuous Development.

- The development and deployment of pipeline components is automatized.
    - Changes in components are tested; if they pass, they're merged automatically and deployed into production.
- Continuous training.
- It requires larger teams and infrastructure, but it is very fast and efficient.
- It is often used in A/B testing, because rapid interations are possible only when we have high automation.
- Customers see continuous improvements: pipelines are iterated very quickly.
- This level appears in large companies with mature ML infrastructures; they are very serious players.

![Levels of MLOps](./pics/level-comparison.png)

### 2.2 Argparse

Logging refresher:

```python
# Instantiate the logger at global scope of the script
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# We can log messages instead of printing in any scope
# Mark the message as "debug" importance
logger.debug("This is a debug message")
# Mark the message as "info" importance
logger.info("This is an info message")
# Mark the message as "warning" importance
logger.warning("This is a warning")
# Mark the message as "error" importance
logger.error("This is an error")
```

Argparse is a Python module which can be used to parse script arguments.
In order to use it, we instantiate an `ArgumentParser` with a description and then simply `add_arguments` to it. Later, those arguments can be introduced to the script execution command and they are parsed for use and stored in a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple).

The following example shows how this works and provides a typical script structure in which we factorize the functionality to a function called from the `main` scope.

```python

import argparse

# The real code / functions
def do_something(args):
    pass

if __name__ == "__main__":

    # Instantiate parser
    parser = argparse.ArgumentParser(description="This is a brief description of this script")

    # Required argument
    parser.add_argument("--number_1",
        type=float, help="First number", required=True)
    
    # Optional argument: default value needed
    parser.add_argument("--number_2",
        type=int, help="Second number", required=False, default=3)

    # Optional argument: default value needed
    parser.add_argument("--artifact_name",
        type=str, help="Name of the artifact", required=False, default="artifact:latest")

    # Parse arguments
    # args is a namedtuple
    # https://docs.python.org/3/library/collections.html#collections.namedtuple
    args = parser.parse_args()

    # Now, we can access the arguments:
    print(f"number_1: {args.number_1}")
    print(f"number_2: {args.number_2}")
    print(f"number_2: {args.artifact_name}")

    do_something(args)

# We can perfom these calls/script execution commands
# Best practice: use "" for strings
#   python my_script.py --number_1 1.2 --number_2 3 -- artifact_name "my_artifact:debug"
#   python my_script.py --number_1 1.2
#   python my_script.py --help

```

### 2.3 Versioning Data and Artifacts in Weights and Biases

In Weights and Biases, we have

- runs,
- experiments,
- artifacts,
- and job types.

A **run** is a basic unit of tracking, i.e., one/multiple script or notebook executions. We can attach to it:

- Parameters
- Metrics
- Artifacts
- Images, plots

One run generates a row in the table of results; all rows can then be analyzed and visualized.

An **experiment** is a grouping of runs. We collect runs that have the same configuration or are part of the same pipeline execution (e.g., the complete pipeline execution can be an experiment). This grouping is optional and can be undone. We can compare experiments. Experiments are defined with `group` in a run.

A **project** is an heterogenous collection of runs, experiments and artifacts related to a goal. We look at one project at a time, and they can be public or private.

An **artifact** is any file/directory produced during a run; they are all versioned and uploaded if anything changes in their content.

We can also specify the `job_type`, which is a mere tag for filtering.

This is how we define a **run**:

```python
import wandb

# Defintion of a run
run = wandb.init(
        #name="my_run_name" usually left so that W&B chooses one
        project="my_project", # project
        group="experiment_1", # experiment
        job_type="data_cleaning" # job type
)
```

### 2.4 Weights & Biases: Example Notebook

The notebook `02_Reproducible_Pipelines/lab/WandB_examples/01_WandB_Upload_Artifact.ipynb` showcases very basic but useful examples of how projects, runs and artifacts are created locally and registered/uploaded to the W&B registry.

In order to use it, you need to have a Weight & Biases account; then run `wandb login` and log in via the web.

The commands shown here have an effect in the projects on our W&B account, accessible via the web. Thus, always check interatcively the W&B web interface to see the changes.

Whenever we execute `wandb.init()`, a `wandb` folder is created with W&B stuff; I add that folder to `.gitignore`.

Note that in my W&B account I created a group `datamix-ai`, which appars in the package output; however, I'm logged as `mxagar`. As far as I know, that has no effect.

**Overview of Contents**:

1. Create a file to be an artifact and instantiate a run
2. Instantiate an artifact, attach the file to it and attach the artifact to the run
3. Change the file and re-attach to artifact & run
4. Using runs with context managers

```python
### -- 1. Create a file to be an artifact and instantiate a run

# We must be logged in: $ wandb login
import wandb

# Create a file
with open("my_artifact.txt", "w+") as fp:
    fp.write("This is an example of an artifact.")

# Check that the file is in the local directory
!ls

# Instantiate a run
run = wandb.init(project="demo_artifact",
                 group="experiment_1")
# Now, we go to the W&B page and look for the project: [https://wandb.ai/mxagar/projects](https://wandb.ai/mxagar/projects).
# We will fin the project, from which hang the `experiment` and the `run` with the automatic name `eternal-planet-1`.
# In Jupyter, we also get a link to the run when we execute a run with `wandb.init()`.

# To check wand object and function options
#wandb.init?
#wandb.Artifact?

### -- 2. Instantiate an artifact, attach the file to it and attach the artifact to the run

# Instantiate an artifact
artifact = wandb.Artifact(
    name="my_artifact.txt", # does not need to be the name of the file
    type="data", # this is to group artifacts together
    description="This is an example of an artifact",
    metadata={ # metadata is an optional dictionary; we can use it for searching later on
        "key_1":"value_1"
    }
)

# We attach a file to the artifact; we can attach several files!
artifact.add_file("my_artifact.txt")

# We attach the artifact to the run
run.log_artifact(artifact)

# The fact that we attached the artuufact to the run doesn't mean that it has been uploaded to the W&B registry. W&B uploads stuff whenever we close a run (e.g., when exiting the notebook) or every a certain amount of time (auto-upload).

# We can manually finish the run to force W&B upload the artifacts
# We cannot use the run object anymore after finish()
run.finish()


### -- 3. Change the file and re-attach to artifact & run

# When we change and re-attach the file, we will have a new version in the W&B web interface. However, a new version is registered only if the file has changed!

# Change the file
with open("my_artifact.txt", "w+") as fp:
    fp.write("This is an example of an artifact changed.")

# Instantiate a run
run = wandb.init(project="demo_artifact",
                 group="experiment_1")

# Instantiate an artifact
artifact = wandb.Artifact(
    name="my_artifact.txt", # does not need to be the name of the file
    type="data", # this is to group artifacts together
    description="This is an example of an artifact",
    metadata={ # metadata is an optional dictionary; we can use it for searching later on
        "key_1":"value_1"
    }
)

# We attach a file to the artifact; we can attach several files!
artifact.add_file("my_artifact.txt")
run.log_artifact(artifact)

# We can manually finish the run to force W&B upload the artifacts
run.finish()


### -- 4. Using runs with context managers

# If we use contexts, it's easier to use several runs. Several runs make sense, for instance, when we're doing hyperparameter tuning. We don't need to do run.finish(), since that's handle by the context manager.

with wandb.init(project="demo_artifact", group="experiment_1") as run:

    with open("my_artifact.txt", "w+") as fp:
        fp.write("This is an example of an artifact.")

    artifact = wandb.Artifact(
        name="my_artifact.txt", # does not need to be the name of the file
        type="data", # this is to group artifacts together
        description="This is an example of an artifact",
        metadata={ # metadata is an optional dictionary; we can use it for searching later on
            "key_1":"value_1"
        }
    )
    
    artifact.add_file("my_artifact.txt")

with wandb.init(project="demo_artifact", group="experiment_1") as run:

    with open("my_artifact.txt", "w+") as fp:
        fp.write("This is an example of an artifact changed again.")

    artifact = wandb.Artifact(
        name="my_artifact.txt", # does not need to be the name of the file
        type="data", # this is to group artifacts together
        description="This is an example of an artifact",
        metadata={ # metadata is an optional dictionary; we can use it for searching later on
            "key_1":"value_1"
        }
    )
    
    artifact.add_file("my_artifact.txt")

```

### 2.5 Weights & Biases: Exercise 1, Versioning Data & Artifacts and Using Them

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`/lesson-1-machine-learning-pipelines/exercises/exercise_1`

I also copied the files to

`./lab/WandB_exercise_1_upload_use_artifact/`

The exercise uses 3 files, 2 of which need to be completed:

- `zen.txt`: text file used as artifact.
- `upload_artifact.py`: create run, attach the `zen.txt` as artifact and upload it.
- `use_artifact.py`: **use/download** different versions of the artifact uploaded.

All files use `logging` and `argparse`. They need to be executed with the command specified in the file docstring.

Basically, artifacts are registered, changed and re-registered. After that, different versions are used. Note that every time we handle different versions of artifacts, W&B creates two folders:

- `wandb/`
- `artifacts/`

File `upload_artifact.py`:

```python
'''This file creates and registers a Weights & Biases run using arguments passed from the CLI.

To run it:

python upload_artifact.py --input_file zen.txt \
           --artifact_name zen_of_python \
           --artifact_type text_file \
           --artifact_description "20 aphorisms about writing good python code"
           
Then, check it on the W&B web interface: https://wandb.ai/home Projects: exercise_1

If we change zen.txt and re-run the script, a new artifact version will appear in W&B.
But if we re-run the script without changing the file no new version will appear.

When we have different versions, we'll get a local folder artifacts/ which contains them. Whenever we'd like to use a version, the run object gets the correct path (see script use_artifact.py).
'''
import argparse
import logging
import pathlib
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Creating run exercise_1")

    # Create a W&B run in the project ``exercise_1``. Set the option ``job_type="upload_file"``:
    run = wandb.init(project="exercise_1",
                     job_type="upload_file")

    # Create an instance of the class ``wandb.Artifact``. Use the ``artifact_name`` parameter to fill
    # the keyword ``name`` when constructing the wandb.Artifact class.
    # Use the parameters ``artifact_type`` and ``artifact_desc`` to fill respectively the keyword
    # ``type`` and ``description``
    # HINT: you can use args.artifact_name to reference the parameter artifact_name
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type, 
        description=args.artifact_description,
    )

    # Attach the file provided as the parameter ``input_file`` to the artifact instance using
    # ``artifact.add_file``, and log the artifact to the run using ``run.log_artifact``.
    artifact.add_file(args.input_file)
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload an artifact to W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--input_file", type=pathlib.Path, help="Path to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)

    

```

File `use_artifact.py`:

```python
'''This script uses an artifact registered in Weights & Biases using arguments passed from the CLI.

To run it:

python use_artifact.py --artifact_name exercise_1/zen_of_python:v1

We can change between artifact versions with v0, v1, etc.

The different artifact versions are on the cloud and also locally, in the folder artifacts/
'''
import argparse
import logging
import pathlib
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Creating run in project exercise_1")
    run = wandb.init(project="exercise_1", job_type="use_file")

    logger.info("Getting artifact")

    # Get the artifact and store its local path in the variable "artifact_path"
    # From the W&B web interface (Usage):
    # artifact = run.use_artifact('datamix-ai/exercise_1/zen_of_python:v1', type='text_file')
    # Apparently, we don't necessarily need the team name datamix-ai
    artifact = run.use_artifact(args.artifact_name)
    # The file path is retreived, pointing to the correct version in folder artifacts/
    artifact_filepath = artifact.file()
    print(f"artifact_filepath = {artifact_filepath}")
    
    # We open the file
    # But 
    logger.info("Artifact content:")
    with open(artifact_filepath, "r") as fp:
        content = fp.read()

    print(content)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Use an artifact from W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name and version of W&B artifact", required=True
    )

    args = parser.parse_args()

    go(args)



```

### 2.6 ML Pipeline Components in MLFlow

We're going to use an MLflow component called **MLflow Project**. An **MLflow Project** is a package of data science code that is reusable and reproducible. It includes an API and CLI tools for running and chaining projects into workflows.

An MLflow project has 3 components:

1. The **code** we want to use. This is independent from the rest; it can be in any language! When putting together pipelines, we can even mix components written in different languages.
2. The **environment definition**: runtime dependencies; we can use either conda or docker. Udacity focuses on conda.
3. The **project definition**: contents of the project and how to interact.


#### Conda: `conda.yaml`

Conda is language agnostic, it handles also C++ packages. Also it's open source - we should not mix it with Anaconda.

We define the conda environment in YAML file:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - requests=2.24.0
  - pip=20.3.3
  - mlflow=1.14.1
  - hydra-core=1.0.6
  - pip:
    - wandb==0.10.21
```

In the `conda.yaml`:

- We define the environment: `download_data`
- Channels are distribution channels for packages; `conda-forge` contains many packages. By listing the channels this way, conda looks for the package in the specified order.
- Dependencies: all code dependencies must be specified; if some dependencies are not in conda, we add a section with `pip`. We should specify the exact version used in the development environment. Note that the `pip` section uses `==` instead of `=`. If we don't specify the version, conda will always get the latest version, and our code might fail.
- **IMPORTANT**: it seems I need to add `protobuf` as dependency sometimes (I haven't figured out the cause and when). See other `conda.yaml` that add `protobuf`.

More information on conda by the instructor Giacomo Vianello [Conda: Essential Concepts and Tips](https://towardsdatascience.com/conda-essential-concepts-and-tricks-e478ed53b5b). Concepts covered:

- Why conda
- Entry-level examples
- What is conda
- Conda and pip
- Conda Vs Anaconda Vs Miniconda
- How to install conda
- Getting Miniconda
- Installing packages, and environments
- The base environment
- Other environments
- The best way to use Jupyter and conda
- Removing environments
- Sharing an environment
- Channels
- The conda-forge channel
- Conda and pip
- Conda is slow
- Free up some disk space
- Conda and docker
- In-depth: RPATH and conda
- Conclusions

#### Project Definition: `MLproject`

```
name: download_data
conda_env: conda.yml

entry_points:
  main:
    parameters:
      file_url:
        description: URL of the file to download
        type: uri
      artifact_name:
        description: Name for the W&B artifact that will be created
        type: str

    command: >-
      python download_data.py --file_url {file_url} \
                              --artifact_name {artifact_name}
  other_script:
    parameters:
        parameter_one:
          description: First parameter
          type: str
    command: julia other_script.jl {parameter_one}
```

In the `MLproject` file:

- Note that even though the `MLproject` file is a YAML file, it has no ending.
- Note also that the filename needs to be `MLproject`.
- We define the name of the project (can be any name) and the `conda_env` file defined above.
- The section `entry_points` is very important: it defines all the commands that are available for our project. 
    - We must have a `main` section: default script to run.
    - The other sections are optional, they are other possible scripts.
    - Every entry point has `command` and its `parameters`.
    - A `parameter` has
        - `description`: any string
        - `type`: str, float, uri, path
        - `default`: if the parameter is optional, default value

#### Running the Project

When running the `MLproject` file via CLI, we pass the parameters with the option `-P` and the parameter name. **Important**: the project file name needs to be `MLproject` and we pass the folder name to `mlflow`.

```bash
# Run default script from a local folder
mlflow run ./my_project -P file_url=https://myurl.com \
    -P artifact_name=my_data.csv

# Run different entry point from a local folder
mlflow run ./my_project -e other_script -P parameter_one=27

# Run default script directly from Github (HEAD is used)
mlflow run git@github.com/my_username/my_repo.git \
    -P file_url=https://myurl.com \
    -P artifact_name=my_data.csv

# Run a specific release or tag from the repository (best practice, otherwise HEAD is used)
mlflow run git@github.com/my_username/my_repo.git \
    -v 2.5.3 \
    -P file_url=https://myurl.com \
    -P artifact_name=my_data.csv
```

However, note that it is also possible to use `mlflow` via its API. That makes sense if we want to define a pipeline consisting of several chained components. See Section 2.8 for that topic.

### 2.7 Introduction to YAML

In YAML we can define lists and dictionaries; we can also combine and nest them. List items are preceded by `-` and key-value pairs are signaled with `:`. Example with a nested dictionary and a nested list with it:

```yaml
a: a value
b:
  c: 1.2
  d: 1
  e: a string
c:
  - 1
  - 2
  - another string
  - - 1
    - 2
    - a
```

Especial symbols:

- `#`: comments
- `>-`: line breaks allowed (but if command, we need to add `\`, too)

If we want to parse YAML files: `pip install pyyaml`, then:

```python
import yaml
with open("conda.yml") as fp:
    d = yaml.safe_load(fp)
print(d) # dictionary is printed
```

### 2.8 MLflow: Exercise 2, Defining and Running an MLflow pipeline

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`/lesson-1-machine-learning-pipelines/exercises/exercise_2`

I also copied the files to

`./lab/WandB_MLflow_exercise_2_download_upload_artifact/`

The exercise comes with the python script `download_data.py`, which downloads a file and logs it into W&B.

The script can be run as

```bash
python download_data.py \
       --file_url https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv \
       --artifact_name iris \
       --artifact_type raw_data \
       --artifact_description "The sklearn IRIS dataset"
```

We need to transform it into an MLflow pipeline: `conda.yaml` + `MLproject`.

#### Solution

`conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - requests=2.24.0
  - pip=20.3.3
  - mlflow=1.14.1
  - hydra-core=1.0.6
  - pip:
    - wandb==0.10.21
```

`MLproject`:

```yaml
name: download_data
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      file_url:
        description: URL of the file to download
        type: uri
      artifact_name:
        description: Name for the W&B artifact that will be created
        type: str
      artifact_type:
        description: Type of the W&B artifact that will be created
        type: str
        default: raw_data
      artifact_description:
        description: Description of the W&B artifact that will be created
        type: str

    command: >-
      python download_data.py \
       --file_url {file_url} \
       --artifact_name {artifact_name} \
       --artifact_type {artifact_type} \
       --artifact_description {artifact_description}
```

Execution:

```bash
# Run default script from a local folder.
# We need to pass all non-default arguments required by the command script.
# We specify the folder where the procject file MLproject is.
# The conda environment is created and the command/script are executed. That creates a run in W&B.
mlflow run . \
    -P file_url=https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv \
    -P artifact_name=iris \
    -P artifact_description="This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length"
```

Check: project and artifact appear in Weights & Biases.

### 2.9 Linking Together the Components

The ML pipeline is a graph of components or modules that produce artifacts; the output artifact of a component is the input of another. Thus, **artifacts are the glue** of the pipeline. Additionally, note that there is no limit in the number of inputs & outputs of a component.

![MLflow pipeline](./pics/mlflow-pipeline.png)

We can use `mlflow` via its API; we call a main python script `mlflow.run(...)` several times: each `mlflow.run(...)` is equivalent to using the CLI `mlflow run ...`. That way, we can use several components or modules chained with their artifacts.

Thus:

- `mlflow` is used to call different components with their parameters in a sequence.
- W&B is used to register the artifacts.

The `mlflow.run()` command works as follows:

```python
import mlflow

# Equivalent to 
# mlflow run ./my_project -e main -P file_url="https://..." -P artifact_name="my_data.csv"
mlflow.run(
  # URI can be a local path or the URL to a git repository
  # here we are going to use a local path.
  # The MLproject file must be there.
  uri="my_project",
  # Entry point to call
  entry_point="main",
  # Parameters for that entry point
  parameters={
    "file_url": "https://...",
    "artifact_name": "my_data.csv"
  }
)
```

That python snippet is equivalent to:

```bash
mlflow run ./my_project -e main \
    -P file_url="https://..." \
    -P artifact_name="my_data.csv"
```

Therefore, a typical ML pipeline has several components and has the following structure:

```
ml_pipeline/
    download_data/
        conda.yaml
        MLproject
        run.py
    remove_duplicates/
        conda.yaml
        MLproject
        run.py
    conda.yaml
    main.py
    MLproject
```

The top directory contains all the components in folders, and additionally:

- `main.py`, which executes all the `mlflow.run()`
- `MLproject`, to define how to call `main.py`
- `conda.yaml`, the environment configuration for `main.py`; the component dependencies DO NOT go here!

In fact, each component directory is a MLflow project that can be run on its own.

Example `main.py`:

```python
import mlflow

mlflow.run(
  uri="download_data",
  entry_point="main",
  parameters={
    "file_url": "https://...",
    "output_artifact": "raw_data.csv",
    "output_artifact_type": "raw_data",
    "output_artifact_description": "Raw data"
  }
)

mlflow.run(
  uri="remove_duplicates",
  entry_point="main",
  parameters={
    "input_artifact": "raw_data.csv:latest",
    "output_artifact": "clean_data.csv",
    "output_artifact_type": "dedup_data",
    "output_artifact_description": "De-duplicated data"
  }
)
```

Notes on `main.py`:

- The output of a run is the input for the next run.
- Note that the input artifact of the second component has a `:latest` tag! This is because we use W&B and we always need to provide the version. If we want a specific version and not the latest, we need to specify it, e.g., `:v2`.

#### Pipeline Configuration: Hydra

A complex ML project can have many parameters; they belong to specific components, yet they are all related. In order to track and handle them all together, another configuration layer is usually defined, for instance with [Hydra](https://hydra.cc/docs/intro/). Note that Hydra is not ML specific, it works for any complex project.

We should always avoid hard-coding the parameters in the `main.py` script, because that makes difficult to re-use the code. Instead, we can have them in a `hydra` configuration YAML which is used by the rest of the project. That configuration file defines the parameters and their default values. We have these advantages:

- All parameters are documented and can be versioned.
- We can override the parameters if required.
- We can genrate multiple runs from the same command; example: hyper-parameter optimization.

The typical hydra YAML has sections that map each one to each of the components (in addition to the main script), but that's a convention -- we can organize it as we please.

Example configuration YAML for hydra; the project aims to train a random forest and has two components: (1) fetching the data and the (2) train random forest. Each section can have subsections.

Hydra configuration `config.yaml`:

```yaml
main:
  project_name: my_project
  experiment_name: dev
data:
  train_data: "exercise_6/data_train.csv:latest"
random_forest_pipeline:
  random_forest:
    n_estimators: 100
    criterion: gini
    max_depth: null
```

In the `main.py` file which calls `mlflow.run()` we just need to `import hydra` and add a decorator; with that, we have access to the configuration paramaters:

File `main.py`:

```python
import os
import mlflow
import hydra

@hydra.main(config_name="config") # config.yaml is read (not we DON't add .yaml)
def go(config): # config object has teh contents of config.yaml
    # Now here config is a dictionary with our configuration
    # For example, to access the parameter project_name in the data
    # section we can just do
    project_name = config["main"]["project_name"]

    # WandB run grouping
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"] # WandB project name
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"] # WandB experiment name

    mlflow.run(
        uri="test_data",
        parameters={
            "input_artifact": config["data"]["train_data"],
            "output_data": "raw_data.csv"
            # ...
        }
    )
    mlflow.run(
        uri="train_random_forest",
        parameters={
            "input_artifact": "raw_data.csv:latest",
            "n_estimators": config["random_forest_pipeline"]["random_forest"]["n_estimators"]
            # ...
        }
    )


if __name__=="__main__":
    go() # we DON'T pass config
```

Notes:

- Since we're using `hydra`, we don't need `argparse`, at least in `main.py`.
- `go()` is called *without* the `config` object.
- The `config` object is created when using the `hydra` decorator, which loads the `config.yaml`, even though we pass the filename without extension.

To use `hydra`, we need to change the `MLproject` file so that we can override the parameters written in the configuration file. That is done allowing only one option, `hydra_options`, which is echoed directly to the `main.py` script. If we don't explicitly override parameters, the default ones from the `hydra` `config.yaml` are going to be used.

Top level `MLproject` file:

```yaml
name: main
conda_env: conda.yml

entry_points:
  main:
    parameters:
      hydra_options:
        description: Hydra parameters to override
        type: str
        default: ''
    command: >-
      python main.py $(echo {hydra_options})
```

ML pipeline execution call with everything together:

```bash
# Run local pipeline
mlflow run .

# Run pipeline in path
mlflow run /path/to/project

# Run with an overidden parameter from config.yaml
# The sections and their subsections are separated with .
mlflow run . -P hydra_options="main.experiment_name=my_experiment"
mlflow run . -P hydra_options="random_forest.random_forest_pipeline.n_estimators=50"

# Run with several overidden parameters
# Parameters are separated with white space
mlflow run . \
    -P hydra_options="main.experiment_name=my_experiment main.project_name=test"
```

#### Tracking Pipelines with Weights & Biases

Each run in W&B is tracked; additionally, recall that we can group them together in experiments and projects. It is in fact a best practice. That is done by defining the environment variables `WANDB_PROJECT` and `WAND_RUN_GROUP`.

```python
import hydra
import mlflow
import os

@hydra.main(config_name="config")
def go(config):
    # WandB run grouping
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"] # WandB project name
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"] # WandB experiment name

    mlflow.run(
        # ...
    )
    # ...

if __name__ == "__main__":
    go()
```

However, note that:

- This is set for the main script only.
- Any component can override the project/experiment name in `wandb.init()`.

### 2.10 MLflow and Hydra: Exercise 3, Parametrized Pipeline

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`/lesson-1-machine-learning-pipelines/exercises/exercise_3`

I also copied the files to

`./lab/WandB_MLflow_Hydra_exercise_3_parametrized_pipeline/`

**This is a very interesting exercise, since a complete pipeline with 2 characteristic steps is realized. It's a nice blueprint for larger projects.** Things done:

- Fetch data from an URL
- Add/upload CSV artifact 
- Download CSV artifact
- Process data: t-SNE
- Upload processed data + image (t-SNE)

Everything is reflected in the W&B web interface.

The exercise comes with the following file structure, which contains an ML pipeline with two components:

```
.
├── MLproject # mlflow main
├── conda.yml # mlflow main
├── config.yaml # hydra
├── main.py # mlflow main
├── download_data # mlflow step/component
│    ├── MLproject
│    ├── conda.yml
     └── download_data.py # download the data
└── process_data # mlflow step/component
    ├── MLproject
    ├── conda.yml
    └── run.py # t-SNE visualization + dataframe with t-SNE features
```

We need to stitch these two components together by finishing what's missing in each file. Then, we execute the pipeline and check everything is registered in the W&B web interface.

#### Solution

Main `MLproject`:

```yaml
name: download_data
conda_env: conda.yml

entry_points:
  main:
    parameters:
      hydra_options:
        description: Hydra parameters to override
        type: str
        default: ''
    command: >-
      python main.py $(echo {hydra_options})
```

Main `conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  #- python=3.8
  - requests=2.24.0
  - pip=20.3.3
  - mlflow=1.14.1
  - hydra-core=1.0.6
  - pip:
    - wandb==0.10.21
```

Main `config.yaml` (hydra):

```yaml
main:
  project_name: experiment_3
  experiment_name: dev
data:
  file_url: https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv
```

File `main.py`:

```python
import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        }
    )

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "input_artifact": "iris.csv:latest",
            "artifact_name": "clean_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        }
    )

if __name__ == "__main__":
    go()
```

File `download_data/MLproject`:

```yaml
name: download_data
conda_env: conda.yml

entry_points:
  main:
    parameters:
      file_url:
        description: URL of the file to download
        type: uri
      artifact_name:
        description: Name for the W&B artifact that will be created
        type: str
      artifact_type:
        description: Type of the artifact to create
        type: str
        default: raw_data
      artifact_description:
        description: Description for the artifact
        type: str

    command: >-
      python download_data.py --file_url {file_url} \
                              --artifact_name {artifact_name} \
                              --artifact_type {artifact_type} \
                              --artifact_description {artifact_description}

```

File `download_data/conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - requests=2.24.0
  - pip=20.3.3
  - mlflow=1.14.1
  - hydra-core=1.0.6
  - pip:
    - wandb==0.10.21
```

File `download_data/download_data.py`

```python
#!/usr/bin/env python
import argparse
import logging
import pathlib
import wandb
import requests
import tempfile


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    # Derive the base name of the file from the URL
    basename = pathlib.Path(args.file_url).name.split("?")[0].split("#")[0] # iris.csv

    # Download file, streaming so we can download files larger than
    # the available memory. We use a named temporary file that gets
    # destroyed at the end of the context, so we don't leave anything
    # behind and the file gets removed even in case of errors
    logger.info(f"Downloading {args.file_url} ...")
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        
        logger.info("Creating run")
        #with wandb.init(project="exercise_3", job_type="download_data") as run:
        with wandb.init(job_type="download_data") as run:
            
            # Download the file streaming and write to open temp file
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

            # Make sure the file has been written to disk before uploading
            # to W&B
            fp.flush()

            logger.info("Creating artifact")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={'original_url': args.file_url}
            )
            artifact.add_file(fp.name, name=basename)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # This makes sure that the artifact is uploaded before the
            # tempfile is destroyed
            #artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()
    
    go(args)
```

File `process_data/MLproject`:

```yaml
name: download_data
conda_env: conda.yml

entry_points:
  main:
    parameters:
      input_artifact:
        description: Fully-qualified artifact name for the input artifact
        type: uri
      artifact_name:
        description: Name for the W&B artifact that will be created
        type: str
      artifact_type:
        description: Type of the artifact to create
        type: str
        default: raw_data
      artifact_description:
        description: Description for the artifact
        type: str

    command: >-
      python run.py --input_artifact {input_artifact} \
                    --artifact_name {artifact_name} \
                    --artifact_type {artifact_type} \
                    --artifact_description {artifact_description}
```

File `process_data/conda.yml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - requests=2.24.0
  - pip=20.3.3
  - seaborn=0.11.1
  - pandas=1.2.3
  - scikit-learn=0.24.1
  - matplotlib=3.2.2
  - pillow=8.1.2
  - pip:
    - wandb==0.10.21
    - protobuf==3.20
```

File `process_data/run.py`:

```python
#!/usr/bin/env python
import argparse
import logging
import seaborn as sns
import pandas as pd
import wandb

from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(project="experiment_3", job_type="process_data")
    #run = wandb.init(job_type="process_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    iris = pd.read_csv(
        artifact_path,
        skiprows=1,
        names=("sepal_length", "sepal_width", "petal_length", "petal_width", "target"),
    )

    target_names = "setosa,versicolor,virginica".split(",")
    iris["target"] = [target_names[k] for k in iris["target"]]

    logger.info("Performing t-SNE")
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    transf = tsne.fit_transform(iris.iloc[:, :4])

    iris["tsne_1"] = transf[:, 0]
    iris["tsne_2"] = transf[:, 1]

    g = sns.displot(iris, x="tsne_1", y="tsne_2", hue="target", kind="kde")

    logger.info("Uploading image to W&B")
    run.log({"t-SNE": wandb.Image(g.fig)})

    logger.info("Creating artifact")

    iris.to_csv("clean_data.csv")

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file("clean_data.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
```

Project execution:

```bash
# Local run with default parameters
mlflow run .
# Local run with overriden parameter
mlflow run . -P hydra_options="main.experiment_name=prod"
```

### 2.11 MLflow Project Development Recommendations

These are tipcs and tricks I collected while following the course.

#### Notes, Tips & Tricks

- Make sure that the indentation in the YAML files is correct.
- Make sure that the requirements of the conda env are correctly typed: =, ==, etc.
- Sometimes I needed to add protobuf as pip requirement for the environment.
- The first time a run with an environment is executed, the conda environment is created; the next times, the environment is already there, so it's faster. See how to delete the `mlflow` `conda` environments below.

#### Troubleshooting Pipelines

If the MLflow project has several steps and it doesn't work, we can try each step separately. However, note that the project name for `wandb` needs to be set in the python script, because it is usually passed via the environment. Also note that if we execute the python script manually, we're outside from the desired environment with the required dependencies.

```bash
mlflow run . \
-P input_artifact=iris.csv:latest \
-P artifact_name=clean_data.csv \
-P artifact_type=processed_data \
-P artifact_description="Cleaned data"
```

```bash
python run.py --input_artifact iris.csv:latest \
       --artifact_name clean_data.csv \
       --artifact_type processed_data \
       --artifact_description "Cleaned data"
```

#### Remove all conda mlflow conda environments

The first time a run with an environment is executed, the conda environment is created; the next times, the environment is already there, so it's faster. If we want to remove all `mlflow` `conda` environment, we can use this script.

```bash
 #!/bin/bash
 
 echo "Removing all conda environments with prefix 'mlflow-'"
 
 conda env list | cut -d " " -f1 | while read -r env ; do
     echo "Processing $env"
     if [[ $env == mlflow-* ]]; then
         conda env remove -n $env
     fi  
 done
```

### 2.12 Conda vs. Docker

I will spare the [Docker overview](https://docs.docker.com/get-started/overview/) details, since I've worked already with docker.

Mlflow offers the possibility of working with Docker instead of conda. For more details, look at: [Specifying an Environment: Docker](https://www.mlflow.org/docs/latest/projects.html#specifying-an-environment)

Conda, in contrast to Docker, is/has:

- Easier to use, no need of DevOps knowledge.
- No need of a separate registry for the images.
- Self-contained, but not completely isolated: still OS libraries are used.
- No integration with Kubernetes; that's only possible with Docker

### 2.13 Running in the Cloud

MLflow is written by [Databricks](https://databricks.com/); hence, it has full support for deployment at Databricks.

In summary, these are the options we have for running our pipelines in the cloud:

- If we have Databricks enterprise, we can just add the option `-b` to the `mlflow run` command. More details: [Run MLflow Projects on Databricks](https://docs.databricks.com/applications/mlflow/projects.html#run-mlflow-projects-on-databricks).
- We can upload the code to a repo, generate an AWS compute instance, clone the repo there and run it on the cloud. This is the manual way.
- We can use hydra launchers. These can distribute the work in different cores, computers, clusters, etc. Hydra is very powerful, but not covered here.
- Use Kubernetes with Kubeflow.

## 3. Data Exploration and Preparation

### 3.1 Exploratory Data Analysis (EDA)

Before developing any Machine Learning Pipeline, we need to perform our Exploratory Data Analysis (EDA). EDA helps us understand the dataset:

- Learn data types, ranges, distributions, etc.
- Plots: histograms, etc.
- Identify missing values, outliers
- Test assumptions
- Uncover data biases

EDA is quite an art, since it's specific for each dataset.

Even though EDA is interactive and is usually done with Jupyter notebooks, we can track it with Weights & Biases and MLflow.

To that end:

- Write an MLflow component that installs Jupyter + libs, and execute the EDA as a notebook from within this component.
- Track inputs and outputs of the notebook (artifacts) with Weights & Biases; for that we create a run and set `save_code=True`. This keeps the code in synch with the copy W&B. W&B tracks even the order in which the cells were run. BUT: please, write notebooks that can be run sequentially from top to bottom!

```python
run = wandb.init(
  project="my_exercise",
  save_code=True
)
```

### 3.2 Pandas Profiling

[Pandas profiling](https://github.com/ydataai/pandas-profiling) is a tool which can help during EDA. A **profile** is an interactive visualization of the characteristics of a dataframe. It goes beyond `df.describe()`.

```bash
conda install -c conda-forge pandas-profiling
```

The [pandas-profiling](https://github.com/ydataai/pandas-profiling) repository has very nice quick-start commands.

Here, an example with the Iris dataset is provided in a code snippet to be run on a Jupyter notebook.

`./lab/pandas_profiling/Pandas_Profiling.ipynb`:

```python
import pandas_profiling
import pandas as pd
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['species'])
y['species'].replace({0:iris['target_names'][0],
                      1:iris['target_names'][1],
                      2:iris['target_names'][2]},
                      inplace=True)
df = pd.concat([X,y],axis=0)

# Pandas profiling report
profile = pandas_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_widgets()
# Report with:
# - Warnings
# - Correlations
# - Variable information: histogram, descriptives, etc.
# - ...
```

### 3.3 Set Up EDA Notebook with WandB and MLflow: Exercise 4

This exercise shows how to perform Jupyter notebook tracking with W&B and MLflow. The key idea is that the notebook runs as well as the code can be tracked by W&B. MLflow is used just to automate the environment generation and launching of Jupyter Notebook.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`/lesson-2-data-exploration-and-preparation/exercises/exercise_4`

I also copied the files to

`./lab/WandB_MLflow_Jupyter_EDA_exercise_4_tracked_notebook/`

The dataset used is a modified version of the [Spotify Songs](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify).

First, upload the dataset from the repo:

```bash
cd exercise_4/ # where genres_mod.parquet is located
wandb artifact put \
      --name exercise_4/genres_mod.parquet \
      --type raw_data \
      --description "A modified version of the songs dataset" genres_mod.parquet
```

Then, we carry out these steps (exercise instructions):

- Upload the dataset to W&B with the command above.
- Write the `MLproject` file: `main` step, no `parameters`, `command`: `jupyter notebook`.
- Check & update the `conda.yaml` file.
- Execute: `mlflow run .`
- Wait for the environment to be created and the Jupyter to be opened.
- Create a notebook called EDA.
- Write code into the notebook:
    - Imports
    - `run = wandb.init(project="exercise_4", save_code=True)`
    - `ProfileReport`
    - Simple EDA.
    - `run.finish()`
- Shut down the notebook server manually clicking on the button.

#### Solution

`conda.yaml`:

```yaml
name: simple_eda
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - jupyter=1.0.0
  - jupyterlab=3.1.7
  - seaborn=0.11.1
  - pandas=1.2.3
  - pip=20.3.3
  #- pandas-profiling=2.11.0
  - pandas-profiling=3.2.0
  - pyarrow=2.0
  #- flask=2.1.0
  - ipywidgets=7.6.5
  - pip:
    - wandb==0.10.21
    - protobuf==3.20
```

`MLproject`:

```yaml
name: EDA
conda_env: conda.yml

entry_points:
  main:
    command: >-
      jupyter notebook
```

`EDA.ipynb`:

```python
import wandb

import pandas_profiling
import pandas as pd
import seaborn as sns

# Create a run for our project; name automatically generated
run = wandb.init(
  project="exercise_4",
  save_code=True
)

# Open the artifact: the name is not the filename,
# but the name we used when registering it
# To download the file we need to call .file()
artifact = run.use_artifact("exercise_4/genres_mod.parquet:latest")
df = pd.read_parquet(artifact.file())
df.head()

# Note: Jupyter Lab has sometimes issues; use Jupyter Notebook if you come up with them
profile = pandas_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_widgets()

## Minimal EDA

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)

# New feature
# This feature will have to go to the feature store.
# If you do not have a feature store,
# then you should not compute it here as part of the preprocessing step.
# Instead, you should compute it within the inference pipeline.
df['title'].fillna(value='', inplace=True)
df['song_name'].fillna(value='', inplace=True)
df['text_feature'] = df['title'] + ' ' + df['song_name']

# Do Some plotting

# Finish run to upload the results.
# Close the notebook and stop the jupyter server
# by clicking on Quit in the main Jupyter page (upper right or File Quit, etc.)
# NOTE: DO NOT use Crtl+C to shutdown Jupyter.
# That would also kill the mlflow job.
run.finish()

# Go to W&B web interface: select run.
# You will see an option {} in the left panel.
# Click on it to see the uploaded Jupyter notebook.
```

#### Important Issues

- Pandas profiler sometimes doesn't work with Jupyter lab; JS library needs to be re-compiled locally. Or use Jupyter notebook.
- I had several issues with dependencies I needed to install manually: protobuf, ipywidgets, newer version of pandas-profiling, jupyter, etc. All are reflected in the `conda.yaml`.
- Artifacts and their naming:
    - when we create an artifact with `wandb.Artifact()` we need an `artifact_name`, which doesn't need to be the `file_name`  
    - when we `run.use_artifact()`, we need `project_name/artifact_name:version`
    - when we ` artifact.add_file()` we need the real `file_name`
    - when we `put` or `get` an artifact via CLI, we use the `project_name/artifact_name:version` identifier (if `put`, no version is added)

### 3.4 Clean and Pre-Process the Data

We perform EDA and conclude the pre-processing steps we require; typical steps are:

- Imputation of missing values, if these will not be missing in production.
- Duplicate removal.
- Feature reduction: throw away features that are not available in production.

However, the pre-processing is related to the training data; if the production data requires it, it should go to the **inference pipeline**. In other words, this pre-processing is usually quite small, since we are just shaping the data to the form the model will see it in production. If the data needs larger modifications which will be also done to the data in production, then, we need to implement the transformations in the inference pipeline, obviously

![Pre-processing: Location](./pics/pre_processing_location.png)

Examples of operations that should not be part of the pre-processing step:

- Categorical encoding: if something needs to be carried out in inference and training, then it should go to the inference pipeline!
- Mssing value imputation when this should also happen in production, too.
- Dimensionality reduction (e.g., PCA), when this should happen in production, too.

Key idea: keep in mind the type of data that the model should see in production: its collection, transformations, etc. The model must  be trained with a dataset which is representative of that!

### 3.5 Pre-Processing Script with W&B and MLflow: Exercise 5

In this exercise, the artifact from the previous exercise 4 is downloaded, pre-processed, and a new artifact is registered (pre-processed data). Really no skills are learnt.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`/lesson-2-data-exploration-and-preparation/exercises/exercise_5`

I also copied the files to

`./lab/WandB_MLflow_Preprocessing_Script_exercise_5/`

For the exercise, the following files needed to be done:

- Complete `run.py` with instructions.
- Create and fill in `conda.yaml`.
- Create and fill in `MLproject`.

#### Solution

`conda.yaml`:

```yaml
name: clean_data
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - jupyter=1.0.0
  - jupyterlab=3.1.7
  - seaborn=0.11.1
  - pandas=1.2.3
  - pip=20.3.3
  - pyarrow=2.0
  - ipywidgets=7.6.5
  - pip:
    - wandb==0.10.21
    - protobuf==3.20
```

`MLproject`:

```yaml
name: clean_data
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      input_artifact:
        description: Fully-qualified artifact name for the input artifact
        type: str
        default: exercise_4/genres_mod.parquet:latest
      artifact_name:
        description: Name for the W&B artifact that will be created
        type: str
        default: preprocessed_data.csv
      artifact_type:
        description: Type of the artifact to create
        type: str
        default: clean_data
      artifact_description:
        description: Description for the artifact
        type: str
        default: Cleaned dataset

    command: >-
      python run.py --input_artifact {input_artifact} \
                    --artifact_name {artifact_name} \
                    --artifact_type {artifact_type} \
                    --artifact_description {artifact_description}
```

`run.py`:

```python
#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    # Open the artifact: the name is not the filename,
    # but the name we used when registering it
    # To download the file we need to call .file()
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    df = pd.read_parquet(artifact.file())
    
    logger.info("Performing pre-processing: duplicates + new data")
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # New feature
    # This feature will have to go to the feature store.
    # If you do not have a feature store,
    # then you should not compute it here as part of the preprocessing step.
    # Instead, you should compute it within the inference pipeline.
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # Persist cleaned dataset
    df.to_csv(args.artifact_name, sep=',', header=True, index=False)

    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.artifact_name)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # Run finish is not necessary here
    # because it finishes automatically
    # when the script ends.
    # In a Jupyter session it is necessary, though.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)

```

#### Running the Example

Since the MLflow project has default arguments, we can just run

```bash
mlflow run .
```

If we didn't have default arguments, we'd need to run

```bash
mlflow run . -P input_artifact="exercise_4/genres_mod.parquet:latest" \
             -P artifact_name="preprocessed_data.csv" \
             -P artifact_type=clean_data \
             -P artifact_description="Cleaned dataset"
```

### 3.6 Data Segregation

Data segregation comes after the **data validation** step, covered later.

![Data segregation](./pics/data_segregation.png)

Data segregation consists in splitting the dataset into the **train, validation and test sub-sets**.

![Data segregation: Train, Validation, Test](./pics/data_segregation_splits_simple.png)

We should take into account the following points:

- First split train/test, and then obtain the validation subset from within the train split.
- We can also perform k-fold validation, for which we don't need to fix the validation sub-set, since it is chosen later.
- The test split is used only to evaluate the model performance when we finish; never use the test split for anything else.
- The validation split is used for model validation, prevent overfitting and hyperparameter tuning; the validation split should be always different to the test split!

We can also go beyond the simple train/validation/test split: this can happen by sub-dividing the validation and test splits in strata groups: male vs. female, low income vs. high income, etc. This is possible if we have enough data. It is called complex sub-sampling.

![Data segregation: Multiple splits](./pics/data_segregation_splits_multiple.png)

### 3.7 Data Segregation Script with W&B and MLflow: Exercise 6

In this exercise, the artifact from the previous exercise 5 is downloaded: the pre-processed dataset; then, that dataset is segregated into train and test splits. Really no skills are learnt.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`/lesson-2-data-exploration-and-preparation/exercises/exercise_6`

I also copied the files to

`./lab/WandB_MLflow_Segregation_Script_exercise_6/`

For the exercise, the following files needed to be done:

- Complete `run.py` with instructions.
- Check the `conda.yaml` file.
- Check the `MLproject` file.

Run:

```bash
mlflow run . -P input_artifact="exercise_5/preprocessed_data.csv:latest" \
             -P artifact_root="data" \
             -P test_size=0.3 \
             -P stratify="genre"
```

#### Solution

`conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - pandas=1.2.3
  - pip=20.3.3
  - scikit-learn=0.24.1
  - pip:
    - wandb==0.10.21
    - protobuf==3.20
```

`MLproject`:

```yaml
name: split_data
conda_env: conda.yml

entry_points:
  main:
    parameters:
      input_artifact:
        description: Fully qualified name for the artifact
        type: str
      artifact_root:
        description: Name for the W&B artifact that will be created
        type: str
      artifact_type:
        description: Type of the artifact to create
        type: str
        default: raw_data
      test_size:
        description: Description for the artifact
        type: float
      random_state:
        description: Integer to use to seed the random number generator
        type: str
        default: 42
      stratify:
        description: If provided, it is considered a column name to be used for stratified splitting
        type: str
        default: "null"

    command: >-
      python run.py --input_artifact {input_artifact} \
                    --artifact_root {artifact_root} \
                    --artifact_type {artifact_type} \
                    --test_size {test_size} \
                    --random_state {random_state} \
                    --stratify {stratify}
```

`run.py`:

```python
#!/usr/bin/env python
import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_6", job_type="split_data")

    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split model_dev/test
    logger.info("Splitting data into train and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
                                                       df, # entire dataset: X, y
                                                       test_size=args.test_size,
                                                       random_state=args.random_state,
                                                       stratify=df[args.stratify] if arg.stratify != 'null' else None
                                                       )

    # Now we save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"{args.artifact_root}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # This waits for the artifact to be uploaded to W&B. If you
            # do not add this, the temp directory might be removed before
            # W&B had a chance to upload the datasets, and the upload
            # might fail
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the produced artifacts", required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
             "splitting",
        type=int,
        required=False,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'  # unfortunately mlflow does not support well optional parameters
    )

    args = parser.parse_args()

    go(args)
```

#### Run the Exercise

```bash
mlflow run . -P input_artifact="exercise_5/preprocessed_data.csv:latest" \
             -P artifact_root="data" \
             -P test_size=0.3 \
             -P stratify="genre"
```

### 3.8 Feature Stores

Feature engineering requires applying domain knowledge and it can make a huge difference in terms of model performance.

For instance, if we have the `height` and `weight`, the `BMI = height/weight^2` is an engineered feature which best predicts whether the subject is overweight.

Feature engineering cannot be in the pre-processing step, because if it were, we would not have those engineered features in the inference pipeline. Thus, one solution is to allocate it to the inference pipeline. However, this does not scale optimally: it's slow and changes stop the inference.

Another solution consists in having **feature stores**: this a new concept which allows for development-production symmetric by using engineered features. The feature computation is centralized and automated:

- Feature registry: Several data scientists can define formulas/rules to compute different features, available to everyone.
- Features are computed automatically according to the feature computation definitions. If there is any update which affects the features, these are automatically modified.
- Serving can be done for real-time applications (hot or fast serving - inference) or for high throughput applications (cold or batch serving - training).

As far as I understand, it is like a parallel component which serves features given a dataset. The users commit the feature computation code and the store automatically can serve features in cold/hot for different steps in the pipeline, i.e., training of inference.

However, note that the feature store deals more with the generation of new features. Encoding or scaling are not part of the feature store, but of the inference pipeline!

## 4. Data Validation

If we want to automatically avoid bad data entering our system, data validation is fundamental.

Data validation can be achieved with [pytest](https://docs.pytest.org/en/7.1.x/).

Data validation is equivalent to unit testing in software, but it is applied on incoming data. Basically, our data assumptions are checked to avoid inputting garbage, which would lead to outputting garbage (Garbage In Garbage Out, GIGO).

Examples of why data inconsistencies might appear:

- Rating value changes from 5-stars system to 10-points system with floating point values. Thus, the range and the type are different, as well as the interpretation.
- A value of a product can change from USD to 1,000 USD; thus, the range is different.
- The world changes and previous assumptions don't hold; e.g., before the pandemic households close to workplaces were expensive and their neighborhood means were high, but after the pandemic, with remote work, that is not always true.

Data checks or data validation is placed immediately before or after the data segregation (train/test split).

![Data Checks in the ML Pipeline](./pics/data-checks-in-ml-pipeline.png)


### 4.1 A Primer on Pytest

This section explains how to use pytest in the context of data validations. Several concepts introduced in the first module (Clean Code: Lesson 4, Production Ready Code) are re-visited. Have a look at the notes of the second module.

- Test files are typically stored in the `test/` folder, by convention; i.e., we can use another folder name.
- Test files must start with `test_*`.
- The test functions must start with `test_*`.
- In a `test_` function, we can use `assert` and `isinstance()` to perform data validation.

Typically, fixtures are used with pytest: decorators that can provide with input data.

```python
import pytest
import wandb

run = wandb.init()

# Since we specify scope="session"
# this fixture is run once for the complete session.
# As an effect, the variable `data` triggers the function data() only once
# and it yields the loaded dataframe.
# Using `data` as an argument is equivalent to:
# data_ = data()
# test_data_length(data_)
# Note that with scope="function"
# every test_ function using data would load the complete dataset every time
@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("my_project/artifact").file()
    df = pd.read_csv(local_path, low_memory=False)

    return df


def test_data_length(data):
    """
    We test that we have enough data to continue
    """
    assert len(data) > 1000
```

To run the code below, assuming the file is `test/test_dataset.py`:

```bash
# If the test_ files are not in ., we specify the path where they are
# -vv: verbose
pytest test/ -vv
```

### 4.2 Deterministic Tests: Example / Exercise 7

Deterministic tests check properties of the data without uncertainty; examples:

- Number of features/columns.
- Number of entries/rows.
- Ranges of numerical features.
- Possible categories in a categorical variable, or number of categories.

#### Exercise 7: Deterministic Data Tests

In this exercise, the artifact from the previous exercise 5 is downloaded: the pre-processed dataset; then, that dataset is checked with pytest.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-3-data-validation/exercises/exercise_7/`

I also copied the files to

`./lab/DataValidation_exercise_7_pytest_dataframe/`

We have the following files:

- `MLproject`
- `conda.yaml`: in theory given, in practice I needed to add some dependencies (see issues section below).
- `test_data.py`: file to be completed; two testing functions need to be defined: test categories are correct and some numerical column values are in their expected ranges.

Run:

```bash
cd path-to-mlflow-file
mlflow run . 
```

#### Solution

`MLproject`:

```yaml
name: download_data
conda_env: conda.yml

entry_points:
  main:
    # NOTE: the -s flag is necessary, otherwise pytest will capture all the output and it
    # will not be uploaded to W&B. Hence, the log in W&B will be empty.
    command: >-
      pytest -s -vv .

```

`conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pandas=1.2.3
  - pip=20.3.3
  - pytest=6.2.2
  - pip:
      - wandb==0.10.21
      - protobuf==3.20
```

`test_data.py`:

```python
import pytest
import wandb
import pandas as pd

# This is global so all tests are collected under the same
# run
run = wandb.init(project="exercise_7", job_type="data_tests")


@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("exercise_5/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path, low_memory=False)

    return df


def test_column_presence_and_type(data):
    
    # It is a nice practice to create dictionaries of values to check.
    # `pandas.api.types` provides many type checking functions we can use!
    required_columns = {
        "time_signature": pd.api.types.is_integer_dtype,
        "key": pd.api.types.is_integer_dtype,
        "danceability": pd.api.types.is_float_dtype,
        "energy": pd.api.types.is_float_dtype,
        "loudness": pd.api.types.is_float_dtype,
        "speechiness": pd.api.types.is_float_dtype,
        "acousticness": pd.api.types.is_float_dtype,
        "instrumentalness": pd.api.types.is_float_dtype,
        "liveness": pd.api.types.is_float_dtype,
        "valence": pd.api.types.is_float_dtype,
        "tempo": pd.api.types.is_float_dtype,
        "duration_ms": pd.api.types.is_integer_dtype,  # This is integer, not float as one might expect
        "text_feature": pd.api.types.is_string_dtype,
        "genre": pd.api.types.is_string_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Check that only the known classes are present
    known_classes = [
        "Dark Trap",
        "Underground Rap",
        "Trap Metal",
        "Emo",
        "Rap",
        "RnB",
        "Pop",
        "Hiphop",
        "techhouse",
        "techno",
        "trance",
        "psytrance",
        "trap",
        "dnb",
        "hardstyle",
    ]

    assert data["genre"].isin(known_classes).all()


def test_column_ranges(data):

    ranges = {
        "time_signature": (1, 5),
        "key": (0, 11),
        "danceability": (0, 1),
        "energy": (0, 1),
        "loudness": (-35, 5),
        "speechiness": (0, 1),
        "acousticness": (0, 1),
        "instrumentalness": (0, 1),
        "liveness": (0, 1),
        "valence": (0, 1),
        "tempo": (50, 250),
        "duration_ms": (20000, 1000000),
    }

    for col_name, (minimum, maximum) in ranges.items():
        # Do not forget .dropna() and .all()
        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )

```

#### Important Notes / Comments / Issues

- `pandas.api.types` provides many type checking functions we can use!
- It is a nice practice to create dictionaries of values to check.
- I had to add these dependencies to `conda.yaml`: `python=3.8`, `protobuf==3.20`.
- Get used to auxiliary function of sets: `isin()`, `issuperset()`, etc.
- Get used to aggregate functions like `all()` instead of using loops!

### 4.3 Non-Deterministic or Statistical Tests: Example / Exercise 8

If we want to measure the correctness of a random variable, we perform non-deterministic or statistical tests; for instance, hypothesis tests.

Typically, the current dataset is checked against the dataset used during the exploratory data analysis (EDA). Examples:

- Mean and std. of columns. We can extend that with T-tests.
- Distribution of values in a column.
- Are there outliers?
- Correlations between columns and columns with target. They will be different, but should not vary so much.

In the frequentist hypothesis testing framework, we have:

- A null hypothesis `H0`: assumption of the data; e.g., both two distributions have same means
- An alternative hypothesis `H1`: violation of the assumption; e.g., distributions have different means

If we use T-tests to compare means, we compute the T statistic and checks its p-value in the T distribution. Of course, we need to define the `alpha` significance level beforehand, depending on how many false positives/negatives we'd like to achieve.

If the p-value is below the `alpha` threshold (e.g., 0.05 for 2-sigma), then the `H0` can be rejected; in our case that means that the distributions of the current dataset and the one in the EDA have significantly different means.

![T-Test Statistical Test](./pics/stat-test.png)

A test function that accomplishes such a test would look like this:

```python
import scipy.stats


def test_compatible_mean(sample1, sample2):
    """
    We check if the mean of the two samples is not
    significantly different
    """
    ts, p_value = scipy.stats.ttest_ind(
        sample1, sample2, equal_var=False, alternative="two-sided"
    )

    # Pre-determined threshold
    # However, note that in a regular case in which we are performing a dataset-wise check
    # we need to compare several columns, thus a Bonferroni correction of the alpha is required.
    # alpha_prime = 1 - (1 - alpha)**(1 / len(tested_columns))
    alpha = 0.05

    assert p_value >= alpha, "T-test rejected the null hyp. at the 2 sigma level"

    return ts, p_value
```

Notes:

- `alpha = 0.05` means that we get 5/100 false positives (i.e., no rejections) if we continue repeating the test with different samplings.
- If we perform multiple tests between different groups we need to use a Bonferroni correction. That is usually the case when we check two datasets: multiple columns are tested against each other to state whether the datasets are different! Thus, the example above is not complete!
- The T-test is a parametric test, with many assumptions; maybe we can opt for non-parametric versions, such as the Kolgomorov-Smirnov test with 2 samples.
- A failing test does not imply the datasets are different; we need to understand we're using statistical tests which have an expected amount of error.
- Instead of `scipy`, we can also use other packages, such as [statmodels](https://www.statsmodels.org/stable/stats.html)
- Usually we perform such kind of tests when we compose a new training dataset; in that case, the new and the old training sets can be compared or the new training set and the old test set.


#### Exercise 8: Non-Deterministic Data Tests

In this exercise, the artifacts from the previous exercise 6 is downloaded: the pre-processed dataset split in train and test subsets; then, a statistical test is performed embedded in a pytest testing function. The statistical test is a Kolgomorov-Smirnov test with 2 samples, which is a non-parametric equivalent of the T-test with independent samples.

Usually we perform such kind of tests when we compose a new training dataset, but here we don't have that.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-3-data-validation/exercises/exercise_8/`

I also copied the files to

`./lab/DataValidation_exercise_8_pytest_statistical/`

We have the following files:

- `MLproject`
- `conda.yaml`: in theory given, in practice I needed to add some dependencies (see issues exercise 7).
- `test_data.py`: file to be completed; a testing function needs to be written: a call to the Kolgomorov-Smirnov test with 2 samples.

Run:

```bash
cd path-to-mlflow-file
mlflow run . 
```

#### Solution


`MLproject`:

```yaml
name: download_data
conda_env: conda.yml

entry_points:
  main:
    # NOTE: the -s flag is necessary, otherwise pytest will capture all the output and it
    # will not be uploaded to W&B. Hence, the log in W&B will be empty.
    command: >-
      pytest -s -vv .

```

`conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - protobuf==3.20
  - pandas=1.2.3
  - pip=20.3.3
  - pytest=6.2.2
  - scipy=1.6.1
  - pip:
      - wandb==0.10.21
```

`test_data.py`:

```python
import pytest
import wandb
import pandas as pd
import scipy.stats

# This is global so all tests are collected under the same
# run
run = wandb.init(project="exercise_8", job_type="data_tests")


@pytest.fixture(scope="session")
def data():
    """Pytest fixture which provides the dataset in two separate dataframes (train and test splits).

    Returns:
        tuple of two dataframes: train and test splits of the dataset
    """

    local_path = run.use_artifact("exercise_6/data_train.csv:latest").file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact("exercise_6/data_test.csv:latest").file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


def test_kolmogorov_smirnov(data):
    """The statistical test Kolgomorov-Smirnov
    is roughly the non-parametric equivalent of the T-test.

    Args:
        data (tuple of 2 dataframes): test and train splits automatically loaded in data() fixture.
    """
    
    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Let's decide the Type I error probability (related to the False Positive Rate)
    alpha = 0.05
    # Bonferroni correction for multiple hypothesis testing.
    # We are comparing 2 splits consisting of several columns; thus,
    # several tests are carried out and we need Bonferroni.
    # See blog post on this topic:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha_prime = 1 - (1 - alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:

        ts, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime

```

### 4.4 Parameters in Pytest: Example / Exercise 9

Pytest checks if we have a special file called `conftest.py` in our `tests/` directory. If so, all fixtures defined in that special file are shared in all `test_*` files. We can define CLI parameters passed to the `pytest` command and parsed within the `conftest.py` file. That way, we can define testing environments that are parametrized, i.e., they are reusable.

Example:

```python
import pytest
import pandas as pd

# Special function with which we can define parameters passed via CLI
def pytest_addoption(parser):
    # Parameter input_artifact created: its value can be passed via CLI:
    # pytest . --input_artifact dataset.csv
    # parser is a special fixture from pytest which connects to the CLI input
    parser.addoption("--input_artifact", action="store")

# 
@pytest.fixture(scope="session")
def data(request):
    # We access the parameters created with addoption via request.config.option
    input_artifact = request.config.option.input_artifact
    if input_artifact is None:
        pytest.fail("--input_artifact missing on command line")
    local_path = run.use_artifact(input_artifact).file()
    return pd.read_csv(local_path)
```

Now, we can run a parametrized set of tests in which we pass which dataset to load!

```bash
pytest . -vv --input_artifact example/my_artifact:latest
```

Note that parsed parameters are strings, so we need to cast them if they are numbers!

#### Exercise 9: Parametrized Test Functions

In this exercise, the artifacts from the previous exercise 6 are downloaded: the pre-processed dataset split in train and test subsets; then, a statistical test is performed embedded in a pytest testing function. The statistical test is a Kolgomorov-Smirnov test with 2 samples, which is a non-parametric equivalent of the T-test with independent samples.

In contrast to the previous exercise, we pass the names of the dataset splits and the significance level `alpha` as parameters in the CLI. Note that parsed parameters are strings, so we need to cast them if they are numbers!

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-3-data-validation/exercises/exercise_9/`

I also copied the files to

`./lab/DataValidation_exercise_9_pytest_parametrized/`

We have the following files:

- `MLproject`
- `conda.yaml`: in theory given, in practice I needed to add some dependencies (see notes in exercise 7).
- `test_data.py`: the test is performed here; we use the fixtures defined in `conftest.py`.
- `conftest.py`: this file needs to be extended to define a `ks_alpha` fixture, along with parameters that are parsed from the CLI.

Run:

```bash
cd path-to-mlflow-file
mlflow run . -P reference_artifact="exercise_6/data_train.csv:latest" \
             -P sample_artifact="exercise_6/data_test.csv:latest" \
             -P ks_alpha="0.3"
```

#### Solution

`MLproject`:

```yaml
name: exercise_9
conda_env: conda.yml

entry_points:
  main:
    parameters:
      reference_artifact:
        description: Fully-qualitied name for the artifact to be used as reference dataset
        type: str
      sample_artifact:
        description: Fully-qualitied name for the artifact to be used as new data sample
        type: str
      ks_alpha:
        description: Threshold for the (pre-trial) p-value for the KS test
        type: float
    # NOTE: the -s flag is necessary, otherwise pytest will capture all the output and it
    # will not be uploaded to W&B. Hence, the log in W&B will be empty.
    command: >-
      pytest -s -vv . --reference_artifact {reference_artifact} \
                      --sample_artifact {sample_artifact} \
                      --ks_alpha {ks_alpha}

```

`conda.yaml`:

```yaml
name: download_data
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pandas=1.2.3
  - pip=20.3.3
  - pytest=6.2.2
  - scipy=1.6.1
  - pip:
      - wandb==0.10.21
      - protobuf==3.20

```

`test_data.py`:

```python
import scipy.stats

# Note that we pass the fixtures as arguments!
def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data

    columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Bonferroni correction for multiple hypothesis testing
    # (see my blog post on this topic to see where this comes from:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))

    for col in columns:

        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime

```

`conftest.py`:

```python
import pytest
import pandas as pd
import wandb


run = wandb.init(project="exercise_9", job_type="data_tests")


def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")


@pytest.fixture(scope="session")
def data(request):

    reference_artifact = request.config.option.reference_artifact

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path = run.use_artifact(reference_artifact).file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact(sample_artifact).file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)

```

### 4.5 Alternative Tool for Data Validation: Great Expectations

[Great Expectations](https://greatexpectations.io/) is another we can employ to perform data validation. Its features:

- Tests are called expectations.
- It has a rich set of pre-defined tests.
- It has automatic profiling: we pass the dataset and automatic tests are generated; e.g., number of columns, etc. Then, we tweak the created tests.
- It provides with awesome HTML reports, which can be published.
- Unfortunately, it is a quite complex tool with a steep learning curve.

## 5. Training, Validation and Experiment Tracking

Contents of the section:

- What is an **inference pipeline**, and how to create one
- How to conduct experiments in an ordered and reproducible way
- How to test our final inference artifact
- Options for deployment of our inference artifact

![ML Pipeline: Inference Pipeline](./pics/ml_pipeline_inference.png)

### 5.1 The Inference Pipeline

The **inference pipeline** is a static version of two things that produce an inference or score:

- the data transformations
- and the trained model.

When we serialize and save to disk the inference pipeline, we call it **inference artifact**. The final inference pipeline is generated during the training and validation process, after the dataset has been split and the model type chosen. It's too simplistic to mix the inference pipeline with the model, because it's more than that (i.e., all data transformations necessary to feed the rows to the model and all the backward transformations necessary to interpret the output).

One of the principles of MLOps is the development/production symmetry. Having an inference artifact helps with that. It is very important to use the same code for development and production! If we change something, it affects both environment and we avoid errors. Also, deploying the pipeline is much easier, because it's self-contained.

Most ML frameworks offer tools to create inference pipelines that can be serialized:

- Scikit-Learn has `Pipelines`.
- Pytorch has `torch.script`, which can export scriptable transforms.

Note that it makes sense to use a feature store if the feature engineering 

- is considerable and slows down the inference more than desired,
- or it is shared across models.

### 5.2 Inference Pipelines with Scikit-Learn and Pytorch

In a `Pipeline` object we concatenate the necessary data transformations and append the model at the end; then, we can train the resulting object and perform an inference with it.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

pipe = Pipeline(
  steps=[
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
  ]
)

# OR:
# pipe = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
# make_pipeline is very practical to create easily sub-pipelines with steps that don't have a name

# Fit the entire pipeline: transformers + model
pipe.fit(X_train, y_train)

# Inference
pipe.predict(X_test)
pipe.predict_proba(X_test)
```

However, note that we will have a much more complex data pre-processing than that. To that end, we can use the **column transformer**, which segregates the columns of a dataframe in such a way that we can define a pipeline for each subgroup; then, all columns are merged again. With `ColumnTransformer` we can achieve sub-pipelines.

![Column Transformer](./pics/column-transformer.png)

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# Example dataframe from the sklearn docs
df = pd.DataFrame(
    {'city': ['London', 'London', 'Paris', 'Sallisaw'],
     'title': ["His Last Bow", "How Watson Learned the Trick",
               "A Moveable Feast", "The Grapes of Wrath"],
     'expert_rating': [5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3],
     'click': ['yes', 'no', 'no', 'yes']})
y = df.pop("click")
X = df

# Build a Column transformer
categorical_preproc = OneHotEncoder()
text_preproc = TfidfVectorizer()
# We can define sub-pipelines
numerical_preprocessing = make_pipeline(SimpleImputer(), StandardScaler())
preproc = ColumnTransformer(
    transformers=[
        ("cat_transform", categorical_preproc, ['city']), # [] because OneHotEncoder expects 2D data
        ("text_transform", text_preproc, 'title'), # no [] because TfidfVectorizer expects plain data
        ("num_transform", numerical_preprocessing, ['expert_rating', 'user_rating'])
    ],
    remainder='drop' # if a column wasn't referenced in the ColumnTransformer, drop it
)
pipe = make_pipeline(preproc, LogisticRegression())
pipe.fit(X, y)
```

#### Pipelines with Pytorch

Pipelines are also possible with other ML frameworks, e.g., pytorch.

The folder `./lab/pytorch_inference_pipeline` contains an example of how pipelines work, following these steps:

- A pre-trained ResNet is loaded
- A `Sequential` pipeline is created and `transforms` + model + `Softmax` are packed into it
- The pipeline is saved as an inference artifact with `torch.jit.script`
- The inference artifact is loaded
- An image is loaded, prepared and passed to the pipeline for inference

To use the example:

```bash
cd  .../lab/pytorch_inference_pipeline
conda activate cvnd
python transforms.py
# ResNet18 is donwloaded
# Inference pipeline is saved to disk
# Inference artifact is loaded as well as a test image
# Inference of the image is done
```

The whole inference script `transforms.py`:

```python
import torch
from torchvision import transforms
from torch.nn import Sequential, Softmax
from PIL import Image
import numpy as np

# Get a pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

# Define the inference pipeline
pipe = Sequential(
    # NOTE: for the pipeline to be scriptable with script,
    # you must use a list [256, 256] instead of just one number (256)
    transforms.Resize([256, 256]),
    transforms.CenterCrop([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    model,
    Softmax(1)
)

# Save inference artifact using torch.script
scripted = torch.jit.script(pipe)
scripted.save("inference_artifact.pt")

# NOTE: normally we would upload it to the artifact store

# Load inference artifact
pipe_reload = torch.jit.load("inference_artifact.pt")

# Load one example
# NOTE: these operations are usually taken care by the inference
# engine
img = Image.open("dog.jpg")
img.load()
# Make into a batch of 1 element
data = transforms.ToTensor()(np.asarray(img, dtype="uint8").copy()).unsqueeze(0)

# Perform inference
with torch.no_grad():
    logits = pipe_reload(data).detach()

proba = logits[0]

# Transform to class and print answer
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]
print(f"Classification: {classes[proba.argmax()]}")
```

#### Example / Exercise 10

In this exercise, the artifacts from the previous exercise 6 are downloaded: the pre-processed dataset split in train and test subsets; then, a pipeline is defined and trained.

Hydra is used for parameter configuration in a `config.yaml` file.

The exercise/example is very interesting and could be used as a boilerplate.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-4-training-validation-experiment-tracking/exercises/exercise_10/`

I also copied the files to

`./lab/InferencePipeline_exercise_10/`

The file structure is the following:

```
.
├── MLproject
├── conda.yml
├── config.yaml # hydra configuration
├── main.py # main script that calls the component random_forest
└── random_forest
    ├── MLproject
    ├── conda.yml
    ├── run.py # the pipeline is defined here; file to be modified

```

To run the project:

```bash
cd path-to-mlflow-file
mlflow run .
# Since the project is connected to W&B, the run logs can be seen on the web interface
# Confusion matrix and feature importances are also uploaded
```

#### Solution to Exercise 10

There are many files; all should be read. I only put here the most important ones. Note that I had to update the `conda.yaml` files to contain `python=3.8` and `protobuf==3.20`.

Hydra configuration file `config.yaml`:

```yaml
main:
  project_name: exercise_10
  experiment_name: dev
data:
  train_data: "exercise_6/data_train.csv:latest"
random_forest:
  n_estimators: 100
  criterion: 'gini'
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  min_weight_fraction_leaf: 0.0
  max_features: 'auto'
  max_leaf_nodes: null
  min_impurity_decrease: 0.0
  min_impurity_split: null
  bootstrap: true
  oob_score: false
  n_jobs: null
  random_state: null
  verbose: 0
  warm_start: false
  class_weight: null
  ccp_alpha: 0.0
  max_samples: null
```

`random_forest/MLproject`:

```yaml
name: decision_tree
conda_env: conda.yml

entry_points:
  main:
    parameters:
      train_data:
        description: Fully-qualified name for the training data artifact
        type: str
      model_config:
        description: JSON blurb containing the configuration for the decision tree
        type: str
    command: >-
      python run.py --train_data {train_data} \
                    --model_config {model_config}

```

Inference pipeline components `random_forest/run.py`:

```python
#!/usr/bin/env python
import argparse
import logging
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import wandb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_10", job_type="train")

    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    logger.info("Setting up pipeline")

    pipe = get_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train, y_train)

    logger.info("Scoring")
    score = roc_auc_score(
        y_val, pipe.predict_proba(X_val), average="macro", multi_class="ovo"
    )

    run.summary["AUC"] = score

    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]

    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])

    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")

    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)

    fig_feat_imp.tight_layout()

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_val,
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )


def get_inference_pipeline(args):
    # Our pipeline will contain a pre-processing step and a Random Forest.
    # The pre-processing step will impute missing values, encode the labels,
    # normalize numerical features and compute a TF-IDF for the textual
    # feature

    # We need 3 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    # - one for textual ("nlp") features

    # Categorical preprocessing pipeline.
    # NOTE: we sort the list so that the order of the columns will be
    # defined, and not dependent on the order in the input dataset
    categorical_features = sorted(["time_signature", "key"])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0), OrdinalEncoder()
    )

    # Numerical preprocessing pipeline
    numeric_features = sorted([
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ])

    ############# YOUR CODE HERE
    # USE make_pipeline to create a pipeline containing a SimpleImputer using strategy=median
    # and a StandardScaler (you can use the default options for the latter)
    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    
    # Textual ("nlp") preprocessing pipeline
    nlp_features = ["text_feature"]
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})

    ############# YOUR CODE HERE
    # USE make_pipeline to create a pipeline containing a SimpleImputer with strategy=constant and
                      # fill_value="" (the empty string), followed by our custom reshape_to_1d instance, and finally
                      # insert a TfidfVectorizer with the options binary=True
    nlp_transformer = make_pipeline(SimpleImputer(strategy="constant",fill_value=""),
                                    reshape_to_1d,
                                    TfidfVectorizer(binary=True))

    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features), # COMPLETE HERE using the categorical transformer and the categorical_features,
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform (i.e., we don't use)
    )

    # Get the configuration for the model
    with open(args.model_config) as fp:
        model_config = json.load(fp)
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    ############# YOUR CODE HERE
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    # CREATE a Pipeline instances with 2 steps: one step called "preprocessor" using the
    # preprocessor instance, and another one called "classifier" using RandomForestClassifier(**model_config)
    # (i.e., a Random Forest with the configuration we have received as input)
    # NOTE: here you should create the Pipeline object directly, and not make_pipeline
    # HINT: Pipeline(steps=[("preprocessor", instance1), ("classifier", LogisticRegression)]) creates a
    #       Pipeline with two steps called "preprocessor" and "classifier" using the sklearn instances instance1
    #       as preprocessor and a LogisticRegression as classifier
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config))
        ]
    )

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a JSON file containing the configuration for the random forest",
        required=True,
    )

    args = parser.parse_args()

    go(args)

```

### 5.3 Machine Learning Experimentation

Machine learning pipelines are difficult to build and often require many iterations and experimentations to solve issues that arise along the way, e.g.:

- Errors in the data processing: unexpected columns, unexpected values, etc.
- Failed data checks
- Class imbalance
- Hyperparameter optimization after the validation
- etc.

![ML Inference Pipeline Iterations](./pics/ml_inference_pipeline_iterations.png)

Due to that extremely iterative nature of ML environments, we should:

- Assume we are going to iterate a lot
- Give ourselves the time to iterate, still keeping the deadlines
- Be systematic: change one thing at the time to keep an intuition of causality
- Learn from every trial.
- Be clear on our objectives and stop once we reach them; in other words, don't strive for undefined perfectionism.

In order to bring order to that chaos, we need to:

- Version our data.
- Version our code; also track very precisely the versions of our dependencies.
- Track every experiment.

Additionally, each experiment needs to be reproducible; to that end, we need to uses randomness seeds whenever randomness is applied.

A typical experimentation loop or iteration sequence is the one done in **hyperparameter optimization**. 

![Experiments](./pics/experiments.png)

### 5.4 Experiment Tracking with Weights and Biases: A Example with Jupyter Notebook

In the repository

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

in the folder

`lesson-4-training-validation-experiment-tracking/demo/wandb_tracking`

there is a notebook `tracking_demo.ipynb` which shows how to track experiments with Weights and Biases.

I also copied the demo files to

`./lab/wandb_tracking`

The notebook shows how to:

- Create an experiment object in W&B
- Store hyperparameters associated with the experiment
- Log final score
- Log time series associated with the experiment, e.g., loss along the iterations
- Log an image

If we run the notebook, an experiment is created in the web interface; if we run it again, we get a new experiment object.

**Important note**: we should commit to our repository the code before running an experiment; that way, the code state is stored with a hash id and if we click on the **info button** of the run in the W&B web interface (left menu bar), we'll get the git checkout command that downloads the precise code status that was used for the experiment! For example:

    git checkout -b "crisp-armadillo-2" c48420c28324e7b1b52aa84523b514f9944a21a0

In that info panel, we can see also the configuration parameters we add to the run/experiment in the code.

```python

import wandb
import matplotlib.pyplot as plt
import numpy as np

run = wandb.init(project="tracking_demo")
# A new experiment/run is created with a random names

# Storing hyper parameters
# We store parameters in dictionaries which can be nested
run.config.update({
    "batch_size": 128,
    "weight_decay": 0.01,
    "augmentations": {
        "rot_angle": 45,
        "crop_size": 224
    }
})

# NOTE: if we have arguments to argparse, we can do:
# parser = argparse.ArgumentParser(description="Train a Random Forest")
# parser.add_argument("--batch_size", type=int, ...)
# parser.add_argument("--weight_decay", type=float, ...)
# args = parser.parse_arguments()
# run.config.update(args)

# Log a final score, i.e., one value
run.summary['accuracy'] = 0.9

# Log a time-varying metric
# The last value will also be reported in the table, unless
# we override it with run.summary['loss']
# Time varying metrics logged with log are plotted
for i in range(10):
    run.log(
        {
            "loss": 1.2 - i * 0.1
        }
    )

# Log multiple time-varying metrics
for i in range(10):
    run.log(
        {
            "recall": 0.8 + i * 0.01,
            "ROC": 0.1 + i**2 * 0.01
        }
    )

# Explicit x-axis.
# If we want to personalize the x-axis of the line plot,
# we can just track the x value we want to associate with a certain y value.
for i in range(10):
    run.log(
        {
            "precision": 0.8 + i * 0.01,
            "epoch": i
        }
    )

from numpy import sqrt 

# Credits: Trae Blain (https://gist.github.com/traeblain/1487795)
x1 = np.arange(-7.25, 7.25, 0.012)
y1 = np.arange(-5, 5, 0.012)
x, y = np.meshgrid(x1, y1)

eq1 = ((x/7)**2*sqrt(abs(abs(x)-3)/(abs(x)-3))+(y/3)**2*sqrt(abs(y+3/7*sqrt(33))/(y+3/7*sqrt(33)))-1)
eq2 = (abs(x/2)-((3*sqrt(33)-7)/112)*x**2-3+sqrt(1-(abs(abs(x)-2)-1)**2)-y)
eq3 = (9*sqrt(abs((abs(x)-1)*(abs(x)-.75))/((1-abs(x))*(abs(x)-.75)))-8*abs(x)-y)
eq4 = (3*abs(x)+.75*sqrt(abs((abs(x)-.75)*(abs(x)-.5))/((.75-abs(x))*(abs(x)-.5)))-y)
eq5 = (2.25*sqrt(abs((x-.5)*(x+.5))/((.5-x)*(.5+x)))-y)
eq6 = (6*sqrt(10)/7+(1.5-.5*abs(x))*sqrt(abs(abs(x)-1)/(abs(x)-1))-(6*sqrt(10)/14)*sqrt(4-(abs(x)-1)**2)-y)
equation=[eq1,eq2,eq3,eq4,eq5,eq6]

fig, sub = plt.subplots()
for f in equation:
    sub.contour(x, y, f, [0])

run.log({
    "batman": wandb.Image(fig)
})

# Very important to finish the run/experiment
# so that everything is uploaded correctly.
# After finishing, go to the W&B web interface and have a look at the experiment:
# metrics, plots, project experiments, 
run.finish()

```

### 5.5 Experiment Tracking and Hyperparameter Optimization with Weights and Biases and Hydra

In the repository

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

in the folder

`lesson-4-training-validation-experiment-tracking/demo/hydra_sweeps`

there is a project which shows how to track hyperparameter optimization experiments with Weights and Biases and Hydra. Basically, [hydra sweeps or multi-runs](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) are used; these allow to run the same application with multiple different configurations: we pass a grid of parameter values and hydra loops on the values and run an experiment with each combination.

I also copied the demo files to

`./lab/hydra_sweeps`

The project has the following structure:

```
.
├── MLproject
├── component
│   ├── MLproject
│   ├── conda.yml
│   └── noop.py # very simple component
├── conda.yml
├── config.yaml # parameters defined here: a, b
└── main.py
```

The project component code is in `noop.py`, but it is a dummy module: it simply logs a sequence of numbers to W&B and creates and artifact.

Note that we need to have some dependencies if hydra and hydra sweeps are used: `hydra-joblib-launcher==1.1.2`, `hydra-core=1.0.6`; additionally, I had to add the dependencies related to `python=3.8` and `protobuf`.

Even though we define the parameter values in the `confg.yaml`, the grid can be manually passed via CLI:

```bash
cd path-to-mlflow-file

# Default parameters from config.yaml used
mlflow run .

# We can overwrite default parameters with grids.
# For that, we use the -m option: multi-run.
# Example with 3 runs (i.e., on for a parameter value combination)
mlflow run . -P hydra_options="-m parameters.a=3,4,5"

# We can create a grid in which we have several values
# for multiple parameters.
# This will generate 6 runs: (a=3, b=2), (a=4, b=2), (a=3, b=3), (a=4, b=3), (a=3, b=4), (a=4, b=4)
mlflow run . -P hydra_options="-m parameters.a=3,4 parameters.b=2,3,4"

# Equivalent call to before, but with range()
# Note: don't use spaces for range()
mlflow run . -P hydra_options="-m parameters.a=3,4 parameters.b=range(2,4,1)"

# Equivalent to before
# BUT we run the experiments in parallel!
# For that, we need joblib
mlflow run . -P hydra_options="hydra/launcher=joblib parameters.a=3,4 parameters.b=range(2,4,1) -m"

```

Important notes:

- Use this functionality to perform grid search or hyperparameter optimization. In the web interface of W&B we can visualize the table of metrics at the project level. If we activate the metric columns only, we see which hyperparameters lead to the best results.
- The CLI length is limited to 250; if we need more characters for the definition of the parameter grids, we can define them in the `config.yaml` and then we run `mlflow` without parameter override!

#### Choosing the Best Performing Model

The following W&B link shows an open project in which a model was trained 183 times with different hyperparameter values, i.e., in a grid:

[https://wandb.ai/example-team/sweep-demo](https://wandb.ai/example-team/sweep-demo)

The associated Github repository is [pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion); the project is basically a FashionMNIST classification with a Pytorch CNN consisting of 2 convolutional layers. In the repository, we can see the grid defined in the configuration file:

```yaml
program: train.py
method: grid
metric:
  goal: minimize
  name: loss
parameters:
  dropout:
    values: [0.1, 0.2, 0.4, 0.5, 0.7]
  channels_one:
    values: [10, 12, 14, 16, 18, 20]
  channels_two:
    values: [24, 28, 32, 36, 40, 44]
  learning_rate:
    values: [0.001, 0.005, 0.0005]
  epochs:
    value: 27
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
```

We can play with the project to select the hyperparameter configuration that leads to the best desired metric, e.g. , the overall accuracy. Simply open the project in table mode and select the columns we want (id, accuracy) and sort all experiments according to the metric.

More interesting examples: [https://github.com/wandb/examples](https://github.com/wandb/examples).

#### Example / Exercise 11: Validate and Choose the Best Performing Model

In this exercise, the artifacts from the previous exercise 6 are downloaded: the pre-processed dataset split in train and test subsets; then, a pipeline is defined and trained.

The exercise is based on the previous exercise 10. However, here the hydra parameter configuration from `config.yaml` is more sophisticated: it includes many more definitions which are used in `random_forest/run.py`.

The exercise/example is very interesting and could be used as a boilerplate.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-4-training-validation-experiment-tracking/exercises/exercise_11/`

I also copied the files to

`./lab/InferencePipeline_exercise_11/`

The file structure is the following:

```
.
├── MLproject
├── conda.yml
├── config.yaml # hydra configuration
├── main.py # main script that calls the component random_forest
└── random_forest
    ├── MLproject
    ├── conda.yml
    ├── run.py # the pipeline is defined here; file to be modified

```

`config.yaml`:

```yaml
main:
  project_name: exercise_11
  experiment_name: dev
data:
  train_data: "exercise_6/data_train.csv:latest"
random_forest_pipeline:
  random_forest:
    n_estimators: 100
    criterion: 'gini'
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 1
    min_weight_fraction_leaf: 0.0
    max_features: 'auto'
    max_leaf_nodes: null
    min_impurity_decrease: 0.0
    min_impurity_split: null
    bootstrap: true
    oob_score: false
    n_jobs: null
    random_state: null
    verbose: 0
    warm_start: false
    class_weight: "balanced"
    ccp_alpha: 0.0
    max_samples: null
  tfidf:
    max_features: 100
  features:
    numerical:
      - "danceability"
      - "energy"
      - "loudness"
      - "speechiness"
      - "acousticness"
      - "instrumentalness"
      - "liveness"
      - "valence"
      - "tempo"
      - "duration_ms"
    categorical:
      - "time_signature"
      - "key"
    nlp:
      - "text_feature"

```

To run the project:

```bash
cd path-to-mlflow-file
mlflow run .
```

However, the exercise consists in running different experiments with different options.

#### Solution to Exercise 11

```bash
## Experiment 1
# We override max_depth
mlflow run . -P hydra_options="random_forest_pipeline.random_forest.max_depth=5"

## Experiment 2
# We override n_estimators
mlflow run . -P hydra_options="random_forest_pipeline.random_forest.n_estimators=10"

## Experiment 3
# Override max_depth with values 1,5,10
mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.max_depth=1,5,10"
# Override max_depth with values 1-10 increased by 2
mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.max_depth=range(1,10,2)"

## Experiment 4
# Sweep multiple parameters
mlflow run . -P hydra_options="hydra/launcher=joblib random_forest_pipeline.random_forest.max_depth=range(10,50,3) random_forest_pipeline.tfidf.max_features=range(50,200,50) -m"
```

At the end, we check the runs of the project in the table view: we hide all columns except the metric we want to optimize (AUC) and the hyperparameters we have varied (tfidf.max_features, random_forest.max_depth). We might decide not to choose the model with the best AUC, but the one with an AUC value close to the best but less complex (smaller depth and less features).

### 5.6 Export the Inference Pipeline / Artifact

We want to package our inference pipeline/artifact and export it to be understood by downstream tools that use it for serving.

To that end, MLflow has **MLflow Models**. That is a standard format accepted by many tools, with the following properties:

- We have different flavors: sklearn, pytorch, keras, ONNX
    - `mlflow.sklearn.save_model() / load_model()`
    - `mlflow.pytorch.save_model() / load_model()`
    - `mlflow.keras.save_model() / load_model()`
    - `mlflow.onnx.save_model() / load_model()`
    - We can also define our own wrappers.
    - See, for instance: [mlflow.sklearn.save_model](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.save_model).
- The exported pipeline can be loaded by mlflow again (imported)
- MLflow models are isolated in conda or docker environments

We can add two important elements to the exported pipeline:

- a signature, which contains a the input/output schema
- input examples for testing

MLflow figures out the correct conda environment automatically and generates the `conda.yaml` file. However, we can also explicitly override it with the `conda_env` option.

The exported model can be converted to a Docker image that provides a REST API for the model.

Example:

```python
from sklearn.pipeline import Pipeline
import mlflow.sklearn
from mlflow.models import infer_signature

# Define and fit pipeline
pipe = Pipeline(...)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

# Get signature and export inference artifact
signature = infer_signature(X_test, pred)
export_path = "model_dir"
mlflow.sklearn.save_model(
    pipe, # our pipeline
    export_path, # Path to a directory for the produced package
    signature=signature, # input and output schema
    input_example=X_test.iloc[:5] # the first few examples
)

artifact = wandb.Artifact(...)
# NOTE that we use .add_dir and not .add_file
# because the export directory contains several
# files
artifact.add_dir(export_path)
run.log_artifact(artifact)
```

#### Example / Exercise 12: Export ML Inference Pipeline with MLflow

In this exercise, the artifacts from the previous exercise 6 are downloaded: the pre-processed dataset split in train and test subsets; then, a pipeline is defined and trained.

The exercise is based on the previous exercises 10 and 11. The new part here is that in `random_forest/run.py` we have a function that packs the pipeline and uploads it as an artifact to W&B: `export_model()`. The exported pipeline is a serialized version, which is supported by many other tools.

The exercise/example is very interesting and could be used as a boilerplate.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-4-training-validation-experiment-tracking/exercises/exercise_12/`

I also copied the files to

`./lab/InferencePipeline_exercise_12/`

The file structure is the following:

```
.
├── MLproject
├── conda.yml
├── config.yaml # hydra configuration
├── main.py # main script that calls the component random_forest
└── random_forest
    ├── MLproject
    ├── conda.yml
    ├── run.py # the pipeline is defined here; file to be modified

```

`config.yaml`:

```yaml
main:
  project_name: exercise_12
  experiment_name: dev
data:
  train_data: "exercise_6/data_train.csv:latest"
random_forest_pipeline:
  random_forest:
    n_estimators: 100
    criterion: 'gini'
    max_depth: 13 # null
    min_samples_split: 2
    min_samples_leaf: 1
    min_weight_fraction_leaf: 0.0
    max_features: 'auto'
    max_leaf_nodes: null
    min_impurity_decrease: 0.0
    min_impurity_split: null
    bootstrap: true
    oob_score: false
    n_jobs: null
    random_state: null
    verbose: 0
    warm_start: false
    class_weight: "balanced"
    ccp_alpha: 0.0
    max_samples: null
  tfidf:
    max_features: 10 # 100
  features:
    numerical:
      - "danceability"
      - "energy"
      - "loudness"
      - "speechiness"
      - "acousticness"
      - "instrumentalness"
      - "liveness"
      - "valence"
      - "tempo"
      - "duration_ms"
    categorical:
      - "time_signature"
      - "key"
    nlp:
      - "text_feature"
  # Set this to "null" if you do not want to export (useful for experimentation)
  export_artifact: "model_export"

```

To run the project:

```bash
cd path-to-mlflow-file

mlflow run .

mlflow run . -P hydra_options="random_forest_pipeline.random_forest.max_depth=13 random_forest_pipeline.tfidf.max_features=10"
```

#### Solution to Example / Exercise 12

`run.py`; in particular, the function `export_model` was changed:

```python
#!/usr/bin/env python
import argparse
import logging
import os

import yaml
import tempfile
import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import wandb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train")

    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    logger.info("Setting up pipeline")

    pipe = get_training_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train, y_train)

    # Evaluate
    pred = pipe.predict(X_val)
    pred_proba = pipe.predict_proba(X_val)

    logger.info("Scoring")
    score = roc_auc_score(y_val, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    # Export if required
    if args.export_artifact != "null":

        export_model(run, pipe, X_val, pred, args.export_artifact)

    # Some useful plots
    fig_feat_imp = plot_feature_importance(pipe)

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_val,
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )


def export_model(run, pipe, X_val, val_pred, export_artifact):

    # Infer the signature of the model
    signature = infer_signature(X_val, val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        #### YOUR CODE HERE
        # Save the pipeline in the export_path directory using mlflow.sklearn.save_model
        # function. Provide the signature computed above ("signature") as well as a few
        # examples (input_example=X_val.iloc[:2]), and use the CLOUDPICKLE serialization
        # format (mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        mlflow.sklearn.save_model(
            pipe, # our pipeline
            export_path, # Path to a directory for the produced package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE, # default, better cross-system compatibility
            signature=signature, # input and output schema
            input_example=X_val.iloc[:2] # the first few examples
        )

        # Then upload the temp_dir directory as an artifact:
        # 1. create a wandb.Artifact instance called "artifact"
        # 2. add the temp directory using .add_dir
        # 3. log the artifact to the run
        artifact = wandb.Artifact(
            name="inference_pipeline",
            type="pipeline", 
            description="Inference pipeline",
        )
        artifact.add_dir(export_path)
        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted; this blocks the execution until then
        artifact.wait()


def plot_feature_importance(pipe):

    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])
    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_training_inference_pipeline(args):

    # Get the configuration for the pipeline
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # We need 3 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    # - one for textual ("nlp") features
    # Categorical preprocessing pipeline
    categorical_features = sorted(model_config["features"]["categorical"])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0), OrdinalEncoder()
    )
    # Numerical preprocessing pipeline
    numeric_features = sorted(model_config["features"]["numerical"])
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    # Textual ("nlp") preprocessing pipeline
    nlp_features = sorted(model_config["features"]["nlp"])
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=True, max_features=model_config["tfidf"]["max_features"]
        ),
    )
    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config["random_forest"])),
        ]
    )
    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    go(args)

```

### 5.7 Testing the Final Artifact and Storing It in the Model Registry

After the inference pipeline has been exported and uploaded as an artifact, we need to test it; we explicitly test the production artifact, since we want to make sure we test  what's going to be used in production!

![ML Pipeline: Model Testing](./pics/ml_pipeline_model_testing.png)

We might discover that our model is overfitting the training set; then, we need to address that separately.

Testing script example:

```python
# Download selected model (i.e., usually the best performing one)
# .download() needs to be used because we get an entire directory
model_export_path = run.use_artifact(args.model_export).download()

# Load pipeline
pipe = mlflow.sklearn.load_model(model_export_path)

# Perform inference
pred_proba = pipe.predict_proba(X_test)

# Evaluate; usually we do much more tests: confusion matrices, etc.
score = roc_auc_score(y_test, pred_proba, average="macro", multi_class="ovo")
run.summary["AUC"] = score
```

After the tests are done and we're happy, we **mark the model for production**: on the W&B web interface, we go to the project artifacts, select the desired one and add to it the tag `prod`.

Then, when getting the artifact use that tag:

```python
import wandb
run = wandb.init()
artifact = run.use_artifact('datamix-ai/exercise_12/inference_pipeline:prod', type='pipeline')
artifact_dir = artifact.download()
```

#### Example / Exercise 13: Testing the Final Model

This is a very simple pipeline with a component that tests and inference pipeline with the test  split of the dataset processed in exercise 6.

The exercise/example is very interesting and could be used as a boilerplate.

Repository:

[udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises)

Folder:

`lesson-4-training-validation-experiment-tracking/exercises/exercise_13/`

I also copied the files to

`./lab/InferencePipeline_exercise_13/`

The file structure is the following:

```
.
├── MLproject
├── conda.yml
└── run.py
```

To run it:

```bash
cd path-to-mlflow-file

mlflow run . -P model_export="exercise_12/inference_pipeline:latest" -P test_data="exercise_6/data_test.csv:latest"
```

#### Solution to Exercise 13

The solution is in `run.py`; basically the artifacts related to the test dataset and the inference pipeline are downloaded and used. The evaluation results (confusion matrix PNG and AUC value) are logged to W & B; we can see them in the web interface.

We can mark model export as production ready: `prod`; we navigate to it in the web interface and add the tag.

```python
#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_13", job_type="test")

    logger.info("Downloading and reading test artifact")
    ## Get the args.test_data artifact from W&B locally
    ## YOUR CODE HERE
    test_data_artifact = run.use_artifact(args.test_data)
    test_data_path = test_data_artifact.file()
    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("genre")

    logger.info("Downloading and reading the exported model")

    ## Get the args.model_export artifact from W&B locally. Since this artifact contains a directory
    # and not a single file, you will have to use .download() instead of .file()
    ## YOUR CODE HERE
    # args.model_export
    artifact = run.use_artifact('datamix-ai/exercise_12/inference_pipeline:prod', type='pipeline')
    model_export_path = artifact.download()

    # Load the model using mlflow.sklearn.load_model
    ## YOUR CODE HERE
    pipe = mlflow.sklearn.load_model(model_export_path)

    # Compute the prediction from the model using .predict_proba on the test set
    ## YOUR CODE HERE
    # args.test_data    
    pred_proba = pipe.predict_proba(X_test)

    logger.info("Scoring")
    score = roc_auc_score(y_test, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    logger.info("Computing confusion matrix")
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_test,
        y_test,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)

```

### 5.8 Using MLflow for Experiment Tracking

We have used WandB for experiment tracking, but MLflow has also that functionality. However, we have some differences:

- No multiple users supported.
- We need to deploy it on our cloud infrastructure.
- There is less functionality around the lineage of artifacts.

For the backend, we can use:

- Local filesystem
- Local database + file system
- Remote database + data lake

We have also a user interface called `mlflow ui`.

Example of tracking (tracking is similar to how it is done in W and B):

```python
experiment_id = mlflow.create_experiment("experiment 1")
with mlflow.run() as run:
    # Log an hyper parameter
    run.log_param("param1", 100)

    # Log a metric
    for i in range(10):
        run.log_metric("foo", i)
    
    # Log an artifact
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("output/test.txt", "w") as f:
        f.write("hello world!")
    run.log_artifacts("outputs")
```

More information: [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html).

## 6. Final Pipeline, Release and Deploy

In this section, we bring everything together and compose and release an entire pipeline.

We can usually do it in two ways:

1. We pack all the components in one repository; we have the main `MLproject` and `conda.yaml` as well as the ones related to the components in their folders.
2. We have one repository for each component and another one for the whole project; this approach is more complex, but it enables easier scalability for larger projects.

It is fundamental to have good skills in git; check:

- [git-cheat-sheet.pdf](https://about.gitlab.com/images/press/git-cheat-sheet.pdf)
- `./git-cheat-sheet.pdf`

### 6.1 Exercise 14: Write an End-to-End Machine Learning Pipeline

**This exercise is the boilerplate for simple ML pipelines with MLflow and W&B**. The pipeline is divided into the typical steps or components in a pipeline, carried out in order.

The exercises/example can be found in two repositories:

- The original from Udacity: [udacity-cd0581-building-a-reproducible-model-workflow-exercises](https://github.com/mxagar/udacity-cd0581-building-a-reproducible-model-workflow-exercises) `/lesson-5-final-pipeline-release-and-deploy/exercises/exercise_14/` 
- My own repository: [https://github.com/mxagar/music_genre_classification](https://github.com/mxagar/music_genre_classification)

The file structure:

```
.
├── MLproject
├── README.md
├── check_data
│   ├── MLproject
│   ├── conda.yml
│   ├── conftest.py
│   └── test_data.py
├── conda.yml
├── config.yaml
├── dataset
│   └── genres_mod.parquet
├── download
│   ├── MLproject
│   ├── conda.yml
│   └── download_data.py
├── evaluate
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── main.py
├── preprocess
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
├── random_forest
│   ├── MLproject
│   ├── conda.yml
│   └── run.py
└── segregate
    ├── MLproject
    ├── conda.yml
    └── run.py
```

The most important high-level files are `config.yaml` and `main.py`; there contain the parameters and the main pipeline execution order, respectively. Each component or pipeline step has its own project sub-folder.

Pipeline steps or components:

1. `download/`
    - A parquet file of songs and their attributes is downloaded from a URL; the songs need to be classified according to their genre.
    - The dataset it uploaded to Weights and Biases as an artifact.
2. `preprocess/`
    - Raw dataset artifact is downloaded and preprocessed: missing values imputed and duplicates removed.
3. `check_data/`
    - Data validation: pre-processed dataset is checked using `pytest`.
    - In the dummy example, the reference and sample datasets are the same, and only deterministic tests are carried out, but we could have used a reference dataset for non-deterministic tests.
4. `segregate/`
    - Train/test split is done and the two splits are uploaded as artifacts.
5. `random_forest/`
    - Component/step with which a random forest model is defined and trained.
    - The training split is subdivided to train/validation.
    - The model is packed in a pipeline which contains data preprocessing and the model itself
        - The data preprocessing differentiates between numerical/categorical/NLP columns with `ColumnTransformer` and performs imputations, reshapings, feature extraction (TDIDF) and scaling.
        - The model configuration is achieved via a temporary yaml file created in the `main.py` using the parameters from `config.yaml`.
    - That inference pipeline is exported and uploaded as an artifact.
    - Performance metrics (AUC) and images (confusion matrix, feature importances) are generated and uploaded as artifacts.
6. `evaluate/`
    - The artifacts related to the test split and the inference pipeline are downloaded and used to compute the metrics with the test dataset.

Obviously, not all steps need to be carried out every time; to that end, with have the parameter `main.execute_steps` in the `config.yaml`. We can override it when calling `mlflow run`.

Possible run commands:

```bash
cd path-to-main-mlflow-file
# All steps are executed, in the order defined in main.py
mlflow run .

# Download dataset and preprocess it
# The order is given by main.py
mlflow run . -P hydra_options="main.execute_steps='download,preprocess,check_data,segregate'"

# Just re-generate the inference pipeline and evaluate it
# The order is given by main.py
mlflow run . -P hydra_options="main.execute_steps='random_forest,evaluate'"

# Change the name of the project for production
mlflow run . -P hydra_options="main.project_name='music_genre_classification_prod'"
```

Since the repository is publicly released, anyone can **run the code remotely** as follows:

```bash
# Go to a new empty folder
cd new-empty-folder

# General command
mlflow run -v [commit hash or branch name] [URL of your Github repo]

# Concrete command for the exercise in section 6.1
# We point to the commit hash of tag 0.0.1
# Note: currently, we cannot put tag names in -v
mlflow run -v 82f17d94e0800811e81f4d55c0442d3189ed0a63 git@github.com:mxagar/music_genre_classification.git

# Project name changed
# We point to the branch main; usually, we should point to a branch like master / stable, etc.
# Note: currently, we cannot put tag names in -v
mlflow run git@github.com:mxagar/music_genre_classification.git -v main -P hydra_options="main.project_name=remote_execution"
```

Issues:

- `conda.yaml` modified/extended to contain `python=3.8` and `protobuf` dependencies.

#### Most Important Files

`config.yaml`:

```yaml
main:
  project_name: exercise_14 # production name: music_genre_classification_prod; add tag prod
  experiment_name: dev
  execute_steps:
    # In production we don't need all steps, so we can override execute_steps, e.g.:
    # mlflow run . -P hydra_options="main.execute_steps='random_forest'"
    # mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
    - download
    - preprocess
    - check_data
    - segregate
    - random_forest
    - evaluate
  # This seed will be used to seed the random number generator
  # to ensure repeatibility of the data splits and other
  # pseudo-random operations
  random_seed: 42
data:
  # We can use either a web link or a file path for files
  # but in the current implementation an URL passes to the requests module is required
  #file_url: "dataset/genres_mod.parquet"
  #file_url: "https://github.com/udacity/nd0821-c2-build-model-workflow-exercises/blob/master/lesson-2-data-exploration-and-preparation/exercises/exercise_4/starter/genres_mod.parquet?raw=true"
  file_url: "https://github.com/mxagar/music_genre_classification/blob/main/dataset/genres_mod.parquet?raw=true"
  reference_dataset: "exercise_14/preprocessed_data.csv:latest"
  # Threshold for Kolomorov-Smirnov test
  ks_alpha: 0.05
  test_size: 0.3
  val_size: 0.3
  # Stratify according to the target when splitting the data
  # in train/test or in train/val
  stratify: genre
random_forest_pipeline:
  random_forest:
    # Whenever we have model or other object with many parameters
    # we should write config files for them.
    # That is easier than passing parameters in the code or via CLI
    # and we can guarantee compatibility in the code in case the model API changes
    # (i.e., we would simply change the config file).
    n_estimators: 100
    criterion: 'gini'
    max_depth: 13
    min_samples_split: 2
    min_samples_leaf: 1
    min_weight_fraction_leaf: 0.0
    max_features: 'auto'
    max_leaf_nodes: null
    min_impurity_decrease: 0.0
    min_impurity_split: null
    bootstrap: true
    oob_score: false
    n_jobs: null
    # This is a different random seed than main.random_seed,
    # because this is used only within the RandomForest
    random_state: 42
    verbose: 0
    warm_start: false
    class_weight: "balanced"
    ccp_alpha: 0.0
    max_samples: null
  tfidf:
    max_features: 10
  features:
    numerical:
      - "danceability"
      - "energy"
      - "loudness"
      - "speechiness"
      - "acousticness"
      - "instrumentalness"
      - "liveness"
      - "valence"
      - "tempo"
      - "duration_ms"
    categorical:
      - "time_signature"
      - "key"
    nlp:
      - "text_feature"
  export_artifact: "model_export"

```

`main.py`:

```python
import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        #assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"), # path of component
            "main", # entry point
            parameters={ # parameters passed to MLproject
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            }
        )

    if "preprocess" in steps_to_execute:

        ## YOUR CODE HERE: call the preprocess step
        # --input_artifact {input_artifact}
        # --artifact_name {artifact_name}
        # --artifact_type {artifact_type}
        # --artifact_description {artifact_description}
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                # the project path is not necessary
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Preprocessed data: missing values imputed and duplicated dropped"
            }
        )
        
    if "check_data" in steps_to_execute:

        ## YOUR CODE HERE: call the check_data step
        # --reference_artifact {reference_artifact}
        # --sample_artifact {sample_artifact}
        # --ks_alpha {ks_alpha}
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            }
        )

    if "segregate" in steps_to_execute:

        ## YOUR CODE HERE: call the segregate step
        # --input_artifact {input_artifact}
        # --artifact_root {artifact_root}
        # --artifact_type {artifact_type}
        # --test_size {test_size}
        # --random_state {random_state}
        # --stratify {stratify}
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": "preprocessed_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                #"random_state": 42, # already default value in MLproject
                "stratify": config["data"]["stratify"]
            }
        )
        
    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        # Whenever we have model or other object with many parameters
        # we should write config files for them.
        # That is easier than passing parameters in the code or via CLI
        # and we can guarantee compatibility in the code in case the model API changes
        # (i.e., we would simply change the config file).
        # Here a yaml is created from the relevant section of the config file
        # and passed as a dictionary to the model later on.
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        ## YOUR CODE HERE: call the random_forest step
        # --train_data {train_data}
        # --model_config {model_config}
        # --export_artifact {export_artifact}
        # --random_seed {random_seed}
        # --val_size {val_size}
        # --stratify {stratify}
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["random_forest_pipeline"]["random_forest"]["random_state"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

        ## YOUR CODE HERE: call the evaluate step
        # --model_export {model_export}
        # --test_data {test_data}
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest"
            }
        )

if __name__ == "__main__":
    go()

```

### 6.2 Releases

A release is a static copy of the code with a tag.

Often the semantic versioning scheme is used:

`1.3.8`: `Major.Minor.Patch`

- Major: changes not backward compatible.
- Minor: changes backward compatible.
- Patch: bug-fixes or small changes, backward compatible.

We can create releases on Github: right column pannel, create release.

- We choose the branch and a tag; we can also create the tag when publishing the release.
- Select a title according to the semantic versioning scheme, e.g., `1.0.0` or `0.0.1`.
- Describe the changes / content of the release.

We then get 

- a tag for our code,
- a releases page with zipped or archived code files.

If we create a release with the tag, everyone can use our release!

We should specify a branch name or a commit hash:

```bash
# General command
mlflow run -v [commit hash or branch name] [URL of your Github repo]

# Concrete command for the exercise in section 6.1
# We point to the commit hash of tag 0.0.1
# Note: currently, we cannot put tag names in -v
mlflow run -v 82f17d94e0800811e81f4d55c0442d3189ed0a63 git@github.com:mxagar/music_genre_classification.git

# Project name changed
# We point to the branch main; usually, we should point to a branch like master / stable, etc.
# Note: currently, we cannot put tag names in -v
mlflow run git@github.com:mxagar/music_genre_classification.git -v main -P hydra_options="main.project_name=remote_execution"
```

### 6.3 Deployments with MLflow

The produced inference pipeline/artifact is stored in the model repository; in our case that is W&B.

Then, the serving solution fetches that artifact from the repository and exposes it to the users with the proper interfaces.

![Deployments with MLflow](./pics/deployment.png)

There are two ways of using a model/pipeline-artifact in production:

1. Realtime (online): one answer at a time through an API. We optimize for speed.
2. Batch (offline): we receive several requests in a batch and provide their answer at once. We optimize for throughput: amount of requests per unit time.

With MLflow, we can use both approach, realtime and batch.

#### Offline/Batch Deployment

```bash
cd .../new_empty_folder
# In our case: cd test_inference

# Get the artifact
# We can check in the web interface the address of the desired artifact:
# Go to project, artifacts, model; make sure to mark it with prod tag for aligning with conventions
# --root: directory to which the artifact is downloaded, model/
# General call (user is optinal if logged in):
# wandb artifact get [<user>/]<project_name>/<inference_artifact>:<tag> --root model
wandb artifact get datamix-ai/music_genre_classification_prod/model_export:prod --root model

# The artifact should be in the specified folder: model/
# The artifact consists of
# - the MLflow file
# - the automatically generated environment file
# - the example JSON with the example samples we used during creation
# - a pickle of the model or the equivalent serialized object
cd model
tree
# .
# ├── MLmodel
# ├── conda.yaml
# ├── input_example.json
# └── model.pkl

# Predict offline/batch
# mlflow models: mlflow deployment API; predict = offline serving
# -t: format, json or csv
# -i: input file in specified format
# -m model: folder where the artifact should be
mlflow models predict -t json -i model/input_example.json -m model
# The conda env is created in the first call
# We get
# ["Rap", "RnB"]

# Another example
# With this, we get a prediction for each row in data_test.csv
wandb artifact get music_genre_classification_prod/model_export:prod --root model
wandb artifact get music_genre_classification_prod/data_test.csv:latest
mlflow models predict -t csv -i artifacts/data_test.csv/v0/data_test.csv -m model
```

#### Online/Realtime Deployment

First, we need to get the model artifact as we did before; then, we call `mlflow models serve` instead of `mlflow models predict`. The command `mlflow models serve` creates a microservice with a REST API which we can access in multiple ways, e.g., with Python using `resquests`.

The service creation with the REST API:

```bash
# Get the inference/model artifact
cd .../new_empty_folder # test_inference
# wandb artifact get [<user>/]<project_name>/<inference_artifact>:<tag> --root model
wandb artifact get music_genre_classification_prod/model_export:prod --root model

# Predict online/realtime
# mlflow models: MLflow deployment API; serve = online
# -m model: folder where the artifact should be
mlflow models serve -m model &
# We get the message:
# Listening at: http://127.0.0.1:5000
# That is, we can post JSONs as before to the following interface and get predictions:
# http://127.0.0.1:5000/invocations
# IMPORTANT NOTE: use 127.0.0.1 and not localhost,
# because sometimes it is not converted correctly by the requests package
```

Accessing the REST API with python: We create a script `test_inference.py` and execute it as `python test_inference.py`; the content of `test_inference.py`:

```python
import requests
import json

with open("model/input_example.json") as fp:
    data = json.load(fp)

print(data)

# JSON has an issue with fields that contain NaN/nan
# They need to be converted to a null string in the JSON file itself (nan -> null)
# or to None in Python.
# One possible solution in Python is the following:
import math
import numpy as np

def to_none(val):
    if not isinstance(val,str):
        if math.isnan(val):
            return None
    return val

def fix_nan(arr):
    arr_new = arr
    # Assuming 2D tables with 1D objects in cells, i.e., no lists in cells
    for row in range(len(arr)):
        for col in range(len(arr[0])):
            arr_new[row][col] = to_none(arr_new[row][col])
    return arr_new

data['data'] = fix_nan(data['data'])

# NaN has been converted to None
print(data)

# Get the prediction/inference through the REST API
# Do not use localhost, use 127.0.0.1, otherwise we could have issues (403 response)
results = requests.post("http://127.0.0.1:5000/invocations", json=data)

# We should get a response 200: correct
print(results)

# Result
print(results.json()) # ['Rap', 'RnB']
```

#### Docker Image

We can create a Docker image which can be instantiated on a Cloud platform (e.g., AWS) as a container. If we expose its port 5000, then the model is listening to the world!

More information: [mlflow-models-build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker).

First, make sure you have Docker installed and running, and that you downloaded the inference artifact with `wandb artifact get`. Then, create the image and test it:

```bash
# Build a generic Docker image:
# Specify the image --name and the folder of the model as downloaded (-m)
mlflow models build-docker -m model --name "music_genre_classification"

# Mount the model stored in './model' and serve it
docker run --rm -p 5000:8080 -v ./model:/opt/ml/model "music_genre_classification"

# Now, the server is listening under
# [url to the deployed machine]:5000/invocations
```

Unfortunately, I had issues with the Java certifications suing my Apple M1.

One helpful link to write the Dockerfile manually could be this one (not tried): [Using MLFlow and Docker to Deploy Machine Learning Models](https://medium.com/@paul.bendevis/using-mlflow-and-docker-to-deploy-machine-learning-models-4f7888005e24).


### 6.4 Other Options for Deployment

MLflow models can be deployed using other tools:

- Spark
- Seldon
- Algrithmia
- BentoML
- Valohai
- etc.

We can also dockerize the MLflow model manually and deploy it on a cloud platform (e.g., AWS).

## 7. Final Project: Build an ML Pipeline for Short-term Rental Prices in NYC

See the repository: [ml_pipeline_rental_prices](https://github.com/mxagar/ml_pipeline_rental_prices).

Also, see the template repository: [music_genre_classification](https://github.com/mxagar/music_genre_classification).