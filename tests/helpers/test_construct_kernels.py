"""
Tests for the helper functions for constructing the kernels.
"""

import numpy, math
import NMTF_drug_sensitivity_prediction.code.helpers.construct_kernels as construct_kernels

def test_remove_homogenous_features():
    # Remove features test
    matrix = numpy.array([[1,2,2,2],[1,1,2,2],[1,3,3,2]])
    names = ["1","2","3","4"]
    
    expected_new_matrix = numpy.array([[2,2],[1,2],[3,3]])
    expected_new_names = ["2","3"]
    
    new_matrix,new_names = construct_kernels.remove_homogenous_features(matrix,names)
    assert numpy.array_equal(expected_new_matrix,new_matrix)
    assert numpy.array_equal(expected_new_names,new_names)
    
    # No features removed test
    matrix = numpy.array([[4,2,2,5],[1,1,2,2],[1,3,3,2]])
    names = ["1","2","3","4"]
    
    new_matrix,new_names = construct_kernels.remove_homogenous_features(matrix,names)
    assert numpy.array_equal(matrix,new_matrix)
    assert numpy.array_equal(names,new_names)
    
    
def test_sort_matrix_rows():
    matrix = numpy.array([[1,2,2,2],[1,1,2,2],[1,3,3,2]])
    names = ["beta","gamma","alpha"]
    
    expected_new_matrix = numpy.array([[1,3,3,2],[1,2,2,2],[1,1,2,2]])
    new_matrix = construct_kernels.sort_matrix_rows(matrix,names)
    assert numpy.array_equal(expected_new_matrix,new_matrix)
    

def test_intersection():
    # Test intersection    
    a1 = [1,10,8,6,3,4]
    a2 = [8,3,1,5,6]
    
    expected_intersection = [1,3,6,8]
    intersection = construct_kernels.intersection(a1,a2)
    assert numpy.array_equal(expected_intersection,intersection)
    
    # Test no intersection
    a1 = [1,10,8,6,3,4]
    a2 = [2,9,7,5]
    
    expected_intersection = []
    intersection = construct_kernels.intersection(a1,a2)
    assert numpy.array_equal(expected_intersection,intersection)
    
    
def test_remove_matrix_rows():
    matrix = numpy.array([[2,2,2],[1,1,1],[5,5,5],[3,3,3],[4,4,4]])
    names = ["2","1","5","3","4"]
    names_to_keep = ["1","4","5"]
    
    expected_matrix = numpy.array([[1,1,1],[5,5,5],[4,4,4]])
    new_matrix = construct_kernels.remove_matrix_rows(matrix,names,names_to_keep)
    assert numpy.array_equal(expected_matrix,new_matrix)
    
    # No rows removed
    names_to_keep = names
    new_matrix = construct_kernels.remove_matrix_rows(matrix,names,names_to_keep)
    assert numpy.array_equal(matrix,new_matrix)


def test_standardise_columns():
    matrix = numpy.array([[1,2,3,4,5],[10,9,8,9,10]])
    #means = [5.5,5.5,5.5,6.5,7.5]
    #stds = [4.5,3.5,2.5,2.5,2.5]
    expected_matrix_standardised = numpy.array([[-1,-1,-1,-1,-1],[1,1,1,1,1]])
    matrix_standardised = construct_kernels.standardise_columns(matrix)
    assert numpy.array_equal(expected_matrix_standardised,matrix_standardised)


def test_jaccard_kernel():
    values = numpy.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,1,1,1]])
    expected_kernel = numpy.array([
        [0,1,1,0.5,0.25],
        [1,0,0,1,1],
        [1,0,0,1,1],
        [0.5,1,1,0,0.75],
        [0.25,1,1,0.75,0]
    ])
    kernel = construct_kernels.jaccard_kernel(values)
    assert numpy.array_equal(expected_kernel,kernel)


def test_jaccard():
    # Test no 1's
    a1 = [0,0,0,0]
    a2 = [0,0,0,0]
    expected_jaccard = 1
    jaccard = construct_kernels.jaccard(a1,a2)
    assert expected_jaccard == jaccard

    # Normal case
    a1 = [1,1,1,0,1,0,1]
    a2 = [1,0,1,0,0,1,0]
    expected_jaccard = 2./6.
    jaccard = construct_kernels.jaccard(a1,a2)
    assert expected_jaccard == jaccard


def test_gaussian_kernel():
    values = numpy.array([[0.1,0.2,0.3,0.4],[-0.1,-0.2,-0.3,-0.4],[0.1,0.2,0.3,0.4]])
    expected_kernel = numpy.array([
        [0,1-math.exp(-0.6),0],
        [1-math.exp(-0.6),0,1-math.exp(-0.6)],
        [0,1-math.exp(-0.6),0],
    ])
    kernel = construct_kernels.gaussian_kernel(values)
    assert numpy.array_equal(expected_kernel,kernel)


def test_gaussian():
    a1 = [0.5,1,1.5]
    a2 = [4.0,2.0,0.0]
    sigma_2 = math.sqrt(3)/4.
    
    expected_gaussian = math.exp( - 62. / (2*math.sqrt(3)) )
    gaussian = construct_kernels.gaussian(a1,a2,sigma_2)
    assert expected_gaussian == gaussian