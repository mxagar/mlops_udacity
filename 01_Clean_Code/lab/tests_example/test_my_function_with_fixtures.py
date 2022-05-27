# File: test_my_function_with_fixtures.py
# Pytest filename starts with "test_...."
# Whereever we define fixtures, pytest must be imported!
import pytest

##################################
"""
Function to test (usually, this is on another file!)
"""
import pandas as pd
def import_data(pth):
    df = pd.read_csv(pth)
    return df

##################################
"""
Fixture - The test function test_import_data() will 
use the return of path() as an argument
"""
@pytest.fixture(scope="module")
def path():
    #return "./data/bank_data.csv"
    return "../wine_quality/winequality-red.csv"

##################################
"""
Test method
"""
def test_import_data(path):
    try:
        df = import_data(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df