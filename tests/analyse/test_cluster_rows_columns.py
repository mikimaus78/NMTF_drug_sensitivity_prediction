"""
Test the methods used for clustering the rows and columns of the Sanger
dataset and computing the number of non-empty clusters.
"""
import NMTF_drug_sensitivity_prediction.code.analyse.cluster_rows_columns as cluster_rows_columns

import numpy

def test_cluster():
    # Simple test case with 1 cluster
    K = 1
    R = [[1,2,3],[4,5,6]]
    M = [[1,1,0],[1,0,1]]
    expected_clustering = [[1],[1]]
    clustering = cluster_rows_columns.cluster(R,M,K)
    assert numpy.array_equal(expected_clustering,clustering)

def test_non_empty():
    clustering_results = numpy.array([[1,0,0],[1,0,0],[0,0,1]])
    expected_non_empty = 2
    non_empty = cluster_rows_columns.non_empty(clustering_results)
    assert expected_non_empty == non_empty
    
def test_run_clustering_K():
    # Simple test case with 1 cluster
    K = 1
    R = [[1,2,3],[4,5,6]]
    M = [[1,1,0],[1,0,1]]
    N = 10
    expected_average_cluster_count = 1
    
    average_cluster_count = cluster_rows_columns.run_clustering_K(R,M,K,N)
    assert expected_average_cluster_count == average_cluster_count
    
    # Test case with 2 clusters, and two identical points
    K = 2
    R = [[1,2,3],[1,2,3]]
    M = [[1,1,1],[1,1,1]]
    N = 10
    expected_average_cluster_count = 1
    
    average_cluster_count = cluster_rows_columns.run_clustering_K(R,M,K,N)
    assert expected_average_cluster_count == average_cluster_count
    
def test_run_all_clusterings():
    R = [[1,2,3],[1,2,3]]
    M = [[1,1,1],[1,1,1]]
    N = 10
    K_range = [1,2]
    expected_average_counts = [1,1]
    
    average_counts = cluster_rows_columns.run_all_clusterings(R,M,K_range,N)
    assert numpy.array_equal(expected_average_counts,average_counts)