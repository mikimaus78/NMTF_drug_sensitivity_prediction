"""
Tests for the methods in code/baselines/naive_averages.py.
"""

import numpy, pytest
import NMTF_drug_sensitivity_prediction.code.baselines.naive_averages as naive_averages


def test_assert_no_empty_rows_columns():
    # Empty row
    Ms = [
        numpy.array([[0,1,1],[1,1,1]]),    
        numpy.array([[1,1,1],[0,0,0]]),
        numpy.array([[1,1,1],[1,1,1]])
    ]
    with pytest.raises(AssertionError) as error:
        naive_averages.assert_no_empty_rows_columns(Ms)
    assert str(error.value) == "Fully unobserved row in M, row 1."
    
    # Empty column
    Ms = [
        numpy.array([[0,1,1],[1,1,1]]),    
        numpy.array([[1,1,1],[0,1,0]]),
        numpy.array([[1,1,0],[1,1,0]])
    ]
    with pytest.raises(AssertionError) as error:
        naive_averages.assert_no_empty_rows_columns(Ms)
    assert str(error.value) == "Fully unobserved column in M, column 2."
    
    # Nothing wrong - no exception expected
    Ms = [
        numpy.array([[0,1,1],[1,1,1]]),    
        numpy.array([[1,1,1],[0,1,0]]),
        numpy.array([[1,1,0],[1,1,1]])
    ]
    try:
        naive_averages.assert_no_empty_rows_columns(Ms)
    except(AssertionError):
        pytest.fail("Unexpected AssertionError raised.")
        
        
def test_f_row():
    X = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    M_training = numpy.array([[1,1,1,0],[0,0,1,1],[0,1,1,0]])
    M_test = numpy.array([[0,0,0,1],[0,1,0,0],[1,0,0,1]])
    # Row averages: [2.0,7.5,10.5]
    expected_MSE = (2.**2 + 1.5**2 + 1.5**2 + 1.5**2) / 4.
    MSE = naive_averages.f_row(X,M_training,M_test)
    assert expected_MSE == MSE
    
    
def test_f_column():
    X = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    M_training = numpy.array([[1,1,1,0],[0,0,1,1],[0,1,1,0]])
    M_test = numpy.array([[0,0,0,1],[0,1,0,0],[1,0,0,1]])
    # Column averages: [1.0,6.0,7.0,8.0]
    expected_MSE = (4.**2 + 0.**2 + 8.**2 + 4.**2) / 4.
    MSE = naive_averages.f_column(X,M_training,M_test)
    assert expected_MSE == MSE
    
   
def test_f_overall():
    X = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    M_training = numpy.array([[1,1,1,0],[0,0,1,1],[0,1,1,0]])
    M_test = numpy.array([[0,0,0,1],[0,1,0,0],[1,0,0,1]])
    # Overall average: 6.0
    expected_MSE = (2.**2 + 0.**2 + 3.**2 + 6.**2) / 4.
    MSE = naive_averages.f_overall(X,M_training,M_test)
    assert expected_MSE == MSE