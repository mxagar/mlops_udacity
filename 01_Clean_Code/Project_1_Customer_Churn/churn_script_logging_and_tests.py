'''This module tests the functions in the module churn_library.py.
That tested module  defined the necessary data science processing steps
to build and save models that predict customer churn using the
Credit Card Customers dataset from Kaggle:

https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code

Altogether, 5 unit tests are defined using pytest:
- test_import(import_data)
- test_eda(perform_eda)
- test_encoder_helper(encoder_helper)
- test_perform_feature_engineering(perform_feature_engineering)
- test_train_models(train_models)

Clean code principles are guaranteed in the project
- Modularized code
- PEP8 conventions
- Error handling
- Testing
- Logging

PEP8 conventions checked with:

>> pylint churn_script_logging_and_tests.py.py # 7.86/10
>> autopep8 churn_script_logging_and_tests.py.py

Since the filename of the tester module does not have a prefix `test_`,
we need to notify pytest the filename to collect all tests:

>> pytest churn_script_logging_and_tests.py

To install pytest:

>> pip install -U pytest

The script expects the proper dataset to be located in `./data`

Additionally:

- Any produced models are stored in `./models`
- Any plots of the EDA and the classification results are stored in `./images/`
- Logs are stored in `./logs/`

Author: Mikel Sagardia
Date: 2022-06-08
'''

import os
from os import listdir
from os.path import isfile, join
import logging
import joblib
import numpy as np
import pytest

#import churn_library_solution as cls
import churn_library as cl

# Logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='w',
    # https://docs.python.org/3/library/logging.html
    # logger - time - level - our message
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

# Fixtures of the churn library functions.
# These fixture could go in a conftest.py file.
# Fixtures are predefined variables passed to test functions;
# in this case, most variables are functions to be tested.
@pytest.fixture
def dataset_path():
    '''Dataset path'''
    return "./data/bank_data.csv"

@pytest.fixture
def import_data():
    '''import_data function from churn_library'''
    return cl.import_data

@pytest.fixture
def eda_path():
    '''Path where EDA images are saved'''
    return './images/eda'

@pytest.fixture
def expected_eda_images():
    '''List of saved EDA image filenames'''
    return ['corr_heatmap.png',
            'total_trans_ct_dist.png',
            'age_dist.png',
            'churn_dist.png',
            'marital_status_dist.png']

@pytest.fixture
def perform_eda():
    '''perform_eda function from churn_library'''
    return cl.perform_eda

@pytest.fixture
def category_lst():
    '''List of categorical features'''
    return ['Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']

@pytest.fixture
def response():
    '''Response/Target variable name'''
    return "Churn"

@pytest.fixture
def encoder_helper():
    '''encoder_helper function from churn_library'''
    return cl.encoder_helper

@pytest.fixture
def num_features():
    '''Number of final features'''
    return 19

@pytest.fixture
def perform_feature_engineering():
    '''perform_feature_engineering function from churn_library'''
    return cl.perform_feature_engineering

@pytest.fixture
def models_path():
    '''Path where models are stored'''
    return './models'

@pytest.fixture
def results_path():
    '''Path where result images are stored'''
    return './images/results'

@pytest.fixture
def expected_models():
    '''List of stored model names'''
    return ['rfc_model_best.pkl',
            'rfc_model.pkl',
            'logistic_model.pkl']

@pytest.fixture
def expected_result_images():
    '''List of saved result images'''
    return ['rf_classification_report.png',
            'lr_classification_report.png',
            'feature_importance.png',
            'roc_plots.png']

@pytest.fixture
def train_models():
    '''train_models function from churn_library'''
    return cl.train_models

def df_plugin():
    '''Initialize pytest dataset container df as None'''
    return None

def splits_plugin():
    '''Initialize pytest splits container as None
    splits = (X_train, X_test, y_train, y_test)
    '''
    return None

def pytest_configure():
    '''Create a dataframe object 'pytest.df' in namespace
    as well as 'pytest.splits'
    '''
    pytest.df = df_plugin() # we can access & modify pytest.df in test functions!
    pytest.splits = splits_plugin()

# Tests

def test_import(import_data, dataset_path):
    '''
    Test data import

    input:
        import_data (function object): function to be tested
        dataset_path (function object): fixture function which returns the path
            of the dataset to be imported
    '''
    try:
        df = import_data(dataset_path)
        # Assign to pytest namespace object for further use
        pytest.df = df
        logging.info("TESTING import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("TESTING import_eda: ERROR - The file wasn't found")
        raise err

    try:
        assert pytest.df.shape[0] > 0
        assert pytest.df.shape[1] > 0
    except AssertionError as err:
        logging.error("TESTING import_data: ERROR - File has no rows / columns")
        raise err


def test_eda(perform_eda, eda_path, expected_eda_images):
    '''
    Test perform_eda_function
    input:
        perform_eda (function object): function to be tested
        eda_path (function object): fixture function which returns the path
            where the EDA images are to be saved
        expected_eda_images (function object): fixture function which returns a
            list of all the expected EDA image filenames
    '''
    # After perform_eda(df) we should get these images:
    perform_eda(pytest.df)
    filenames = [f for f in listdir(eda_path) if isfile(join(eda_path, f))]

    try:
        assert len(filenames) >= len(expected_eda_images)
    except AssertionError as err:
        logging.error("TESTING perform_eda: ERROR - Missing EDA images")
        raise err

    for image in expected_eda_images:
        try:
            assert image in filenames
        except AssertionError as err:
            logging.error("TESTING perform_eda: ERROR - The image %s is missing", image)
            raise err

    logging.info("TESTING perform_eda: SUCCESS")

def test_encoder_helper(encoder_helper, category_lst, response):
    '''
    Test encoder_helper
    input:
        encoder_helper (function object): function to be tested
        category_lst (function object): fixture function which returns a list
            of categorical features to be encoded
        response (function object): fixture function which returns the name
            of the target/response
    '''
    try:
        pytest.df, cat_columns_encoded = encoder_helper(pytest.df, category_lst, response)
    except KeyError as err:
        logging.error("TESTING encoder_helper: ERROR - Missing categorical column")
        raise err

    for col in category_lst:
        col_name = col+"_"+response
        try:
            assert col_name in pytest.df.columns and col_name in cat_columns_encoded
        except AssertionError as err:
            logging.error("TESTING encoder_helper: ERROR - Missing categorical column %s", col_name)
            raise err
        try:
            assert np.sum(pytest.df[col_name].isnull()) < 1
        except AssertionError as err:
            logging.error("TESTING encoder_helper: ERROR - Unexpected NA values in %s", col_name)
            raise err

    logging.info("TESTING encoder_helper: SUCCESS")

def test_perform_feature_engineering(perform_feature_engineering, num_features, response):
    '''
    Test encoder_helper
    input:
        perform_feature_engineering (function object): function to be tested
        num_features (function object): fixture function which returns the number
            of features in the final dataset
        response (function object): fixture function which returns the name
            of the target/response
    '''
    # Feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(pytest.df,
        response=response)
    # Save training splits
    pytest.splits = (X_train, X_test, y_train, y_test)
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] == num_features
        assert X_test.shape[0] > 0
        assert X_test.shape[1] == X_train.shape[1]
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error("TESTING perform_feature_engineering: ERROR - Unexpected sizes for X & y!")
        raise err

    logging.info("TESTING perform_feature_engineering: SUCCESS")

def test_train_models(train_models,
                      models_path,
                      results_path,
                      expected_models,
                      expected_result_images):
    '''
    Test train_models
    input:
        train_models (function object): function to be tested
        models_path (function object): fixture function which returns the path
            where the created models are to be found
        results_path (function object): fixture function which returns the path
            where the created result images are to be found
        expected_models (function object): fixture function which returns
            the names of the models that should have been stored by train_models
        expected_result_images (function object): fixture function which returns
            the names of the result images that should have been stored
            by train_models
    '''
    # Unpack training splits stored in the pytest namespace
    X_train, X_test, y_train, y_test = pytest.splits

    # Peroform training
    train_models(X_train, X_test, y_train, y_test)

    # Check that models are correctly stored
    model_filenames = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    try:
        assert len(model_filenames) >= len(expected_models)
    except AssertionError as err:
        logging.error("TESTING train_models: ERROR - Missing models")
        raise err

    for model in expected_models:
        try:
            assert model in model_filenames
        except AssertionError as err:
            logging.error("TESTING train_models: ERROR - The model %s is missing", model)
            raise err

    # Check that the models can be loaded
    for model_name in model_filenames:
        if model_name in expected_models:
            try:
                model = joblib.load(join(models_path, model_name))
            except Exception as err:
                logging.error("TESTING train_models: ERROR - Model %s cannot be loaded", model_name)
                raise err

    # Check that the result images were correctly saved
    result_filenames = [f for f in listdir(results_path) if isfile(join(results_path, f))]
    try:
        assert len(result_filenames) >= len(expected_result_images)
    except AssertionError as err:
        logging.error("TESTING train_models: ERROR - Missing models")
        raise err

    for result in result_filenames:
        try:
            assert result in result_filenames
        except AssertionError as err:
            logging.error("TESTING train_models: ERROR - The result image %s is missing", result)
            raise err

    logging.info("TESTING train_models: SUCCESS")

if __name__ == "__main__":
    # Without logging, we would run
    # >> pytest
    # or, in this case, since the file does not start with test_*
    # >> pytest churn_script_logging_and_tests.py
    # However, logging does not occur when invoking pytest that way.
    # If we want to have logging with pytest, we either configure the TOML / INI
    # or we define the line below in __main__ and execute the tests with
    # >> python churn_script_logging_and_tests.py
    # Sources:
    # https://stackoverflow.com/questions/4673373/logging-within-pytest-tests
    # https://stackoverflow.com/questions/31793540/how-to-save-pytests-results-logs-to-a-file
    pytest.main(args=['-s', os.path.abspath(__file__)])
