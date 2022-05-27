from my_module import square

# input_value is defined as a fixture in conftest.py
# but we could've had defined it in this file, too - just cut & paste code piece
def test_square_gives_correct_value(input_value):
    # When
    subject = square(input_value)

    # Then
    assert subject == 16