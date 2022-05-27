from my_module import square

# We need to import pytest
# because we're using one of its decorators
import pytest

# We define the variable input to be a parameter of the test
# passed to the testing function.
# The test function is executed so many times as the number of values
# defined in the parameter list
@pytest.mark.parametrize(
    'inputs',
    [2, 3, 4]
)

def test_square_return_value_type_is_int(inputs):
    # When
    subject = square(inputs)

    # Then
    assert isinstance(subject, int)