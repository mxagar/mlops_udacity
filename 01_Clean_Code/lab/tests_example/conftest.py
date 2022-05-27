# With fictures, we need to import pytest
# By convention, all fuxtures defined in conftest.py
# are made available to all detected test files and functions!
import pytest

@pytest.fixture
def input_value():
	return 4
