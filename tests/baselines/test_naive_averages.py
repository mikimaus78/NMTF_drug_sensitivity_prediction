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
    expected_performances = {
        'MSE' : (2.**2 + 1.5**2 + 1.5**2 + 1.5**2) / 4.,
        'R^2' : 0.7074829931972789, 
        'Rp' : 0.88220718990898173
    }
    performances = naive_averages.f_row(X,M_training,M_test)
    assert expected_performances == performances
    
    
def test_f_column():
    X = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    M_training = numpy.array([[1,1,1,0],[0,0,1,1],[0,1,1,0]])
    M_test = numpy.array([[0,0,0,1],[0,1,0,0],[1,0,0,1]])
    # Column averages: [1.0,6.0,7.0,8.0]
    expected_performances = {
        'MSE' : (4.**2 + 0.**2 + 8.**2 + 4.**2) / 4.,
        'R^2' : -1.6122448979591835, 
        'Rp' : -0.15132998169159548
    }
    performances = naive_averages.f_column(X,M_training,M_test)
    assert expected_performances == performances
    
   
def test_f_overall():
    X = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    M_training = numpy.array([[1,1,1,0],[0,0,1,1],[0,1,1,0]])
    M_test = numpy.array([[0,0,0,1],[0,1,0,0],[1,0,0,1]])
    # Overall average: 6.0
    expected_performances = {
        'MSE' : (2.**2 + 0.**2 + 3.**2 + 6.**2) / 4.,
        'R^2' : -0.33333333333333326, 
        'Rp' : 0.0
    }
    performances = naive_averages.f_overall(X,M_training,M_test)
    assert expected_performances == performances