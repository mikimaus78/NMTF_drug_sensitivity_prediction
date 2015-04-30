"""
Tests for the methods in code/baselines/random_forests.py.
"""

import numpy, pytest
import NMTF_drug_sensitivity_prediction.code.baselines.random_forests as random_forests


def test_construct_datapoints_overall():
    X = numpy.array([[1,2,3],[4,5,6]])
    M = numpy.array([[1,1,0],[1,0,1]])

    # Rows are cell lines, columns are drugs
    drug_features = numpy.array([[1,2],[3,4],[5,6]])
    cell_line_features = numpy.array([[7,8],[9,10]])
    
    expected_X = [[1,2,7,8],[3,4,7,8],[1,2,9,10],[5,6,9,10]]
    expected_Y = [1,2,4,6]
    (X,Y) = random_forests.construct_datapoints_overall(X,M,drug_features,cell_line_features)
    
    assert numpy.array_equal(expected_X,X)
    assert numpy.array_equal(expected_Y,Y)



def test_assert_no_empty_rows_columns():
    # Empty row
    Ms = [
        numpy.array([[0,1,1],[1,1,1]]),    
        numpy.array([[1,1,1],[0,0,0]]),
        numpy.array([[1,1,1],[1,1,1]])
    ]
    with pytest.raises(AssertionError) as error:
        random_forests.assert_no_empty_rows_columns(Ms)
    assert str(error.value) == "Fully unobserved row in M, row 1."
    
    # Empty column
    Ms = [
        numpy.array([[0,1,1],[1,1,1]]),    
        numpy.array([[1,1,1],[0,1,0]]),
        numpy.array([[1,1,0],[1,1,0]])
    ]
    with pytest.raises(AssertionError) as error:
        random_forests.assert_no_empty_rows_columns(Ms)
    assert str(error.value) == "Fully unobserved column in M, column 2."
    
    # Nothing wrong - no exception expected
    Ms = [
        numpy.array([[0,1,1],[1,1,1]]),    
        numpy.array([[1,1,1],[0,1,0]]),
        numpy.array([[1,1,0],[1,1,1]])
    ]
    try:
        random_forests.assert_no_empty_rows_columns(Ms)
    except(AssertionError):
        pytest.fail("Unexpected AssertionError raised.")
        