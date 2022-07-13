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

1. [Lesson 1: Introduction to Reproducible Model Workflows](#1.-Introduction-to-Reproducible-Model-Workflows)
2. [Lesson 2: Machine Learning Pipelines](#2.-Machine-Learning-Pipelines)
3. Lesson 3: Data Exploration and Preparation
4. Lesson 4: Data Validation
5. Lesson 5: Training, Validation and Experiment Tracking
6. Lesson 6: Final Pipeline, Release and Deploy
7. Project: Build an ML Pipeline for Short-term Rental Prices in NYC

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

[Gartnet Report: Top Strategic Technology Trends for 2021](https://www.gartner.com/en/newsroom/press-releases/2020-10-19-gartner-identifies-the-top-strategic-technology-trends-for-2021#:~:text=Gartner%20research%20shows%20only%2053,a%20production%2Dgrade%20AI%20pipeline.): *" only 53% of projects make it from artificial intelligence (AI) prototypes to production"*

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

#### Installation of Weights & Biases

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

### 1.5 Module Project: Rental Price Prediction in New York

> A property management company is renting rooms and properties in New York for short periods on various rental platforms. They need to estimate the typical price for a given property based on the price of similar properties. The company receives new data in bulk every week, so the model needs to be retrained with the same cadence, necessitating a reusable pipeline.

![ML project pipeline](./pics/mlops-project-pipeline.png)

## 2. Machine Learning Pipelines

A machine learning pipeline is a sequence of independent, modular and reusable components. It can be represented as a Direct Acyclic Graph (DAG) in which the output artifact of a component is the input for the next one or another component.

Artifacts are ouput files of objects that need to be

- tracked (who did what when),
- versioned,
- stored.

Example 1: **ETL pipeline** = Extract, Transform, Load: it ingests the data from varois sources, aggregates and cleans it and stores it in a database.

![ETL pipeline](./pics/etl-pipeline.png)

Example 2: **Training Pipeline** = It takes the raw dataset and produces an inference artifact, which is more than the model, rather the pipeline. The ETL pipeline is part of it. The artifact is stored in a registry.

![ETL pipeline](./pics/ml-pipeline.png)

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
In order to use it, we instantiate an `ArgumentParser` with a description and then simply `add_arguments` to it. Later, those arguments can be introduced to the script execution comman and they are parsed for use and stored in a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple).

The following example shows how this works and provides a tpical script structure in which we factorize the functionality to a function called from the `main` scope.

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

### 2.4 Weights & Biases: Exercise 1, Versioning Data & Artifacts and Using Them

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

### 2.5 ML Pipeline Components in MLFlow

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

### 2.6 Introduction to YAML

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

### 2.7 MLflow: Exercise 2, Defining and Running an MLflow pipeline

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

### 2.8 Linking Together the Components

The ML is a graph of components or modules that produce artifacts; the output artifact of a component is the input of another. Thus, **artifacts are the glue** of the pipeline. Additionally, note that there is no limit in the number of inputs & outputs of a component.

![MLflow pipeline](./pics/mlflow-pipeline.png)



