"""
Run the Kmeans clustering algorithm on the complete Sanger dataset.
We plot the average number of non-empty clusters as a function of the total
number of clusters K.
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")

from kmeans_missing.code.kmeans import KMeans
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger

from matplotlib import pyplot

K_range_rows = [1]+range(10,200+1,10)+range(220,622,20) # 622 cell lines
K_range_columns = xrange(1,139+1)   # 139 drugs
N = 10                              # Run Kmeans N times for each K, and take the average

# Cluster the matrix R with known values M, with K clusters
def cluster(R,M,K):
    kmeans = KMeans(R,M,K)
    kmeans.initialise()
    kmeans.cluster()
    return kmeans.clustering_results

# Compute the number of non-empty clusters
def non_empty(clustering_results):
    cluster_counts = [sum(column) for column in clustering_results.T]
    return len([count for count in cluster_counts if count != 0])
    
# Run the clustering N times for a value of K and return the average no. of non-empty clusters
def run_clustering_K(R,M,K,N):
    cluster_counts = []
    for n in range(1,N+1):
        print "Clustering for K=%s, n=%s." % (K,n)
        clustering_results = cluster(R,M,K)
        cluster_count = non_empty(clustering_results)
        cluster_counts.append(cluster_count)
    average_cluster_count = sum(cluster_counts) / float(len(cluster_counts))   
    print "K: %s. Average cluster count: %s. Cluster counts: %s." % (K,average_cluster_count,cluster_counts)
    return average_cluster_count
    
# Run the clustering for varying numbers of K
def run_all_clusterings(R,M,K_range,N):
    average_counts = []
    for K in K_range:
        print "Clustering for K=%s." % (K)
        average_count = run_clustering_K(R,M,K,N)
        average_counts.append(average_count) 
    print "Average cluster counts: %s." % (average_counts)
    return average_counts
    
# Plot the average clustering counts against K
def plot(K_range,average_counts,title):
    pyplot.plot(K_range,average_counts,color="blue")
    pyplot.plot(K_range,K_range,color="green")
    pyplot.title(title,y=1.03)
    pyplot.xlabel("K",fontsize=12)
    pyplot.ylabel("Average number of non-negative clusters",fontsize=12)
    pyplot.show()
    
    
if __name__ == "__main__":
    # Load in Sanger dataset
    (_,X_min,M,_,_,_,_) = load_Sanger(standardised=False)
        
    # Cluster rows and plot it
    average_counts_rows = run_all_clusterings(R=X_min, M=M, K_range=K_range_rows, N=N)
    #average_counts_rows = [1.0, 8.1, 14.0, 17.5, 21.5, 23.7, 22.7, 26.8, 28.9, 31.5, 31.5, 33.3, 33.4, 34.8, 36.7, 38.8, 39.6, 41.7, 40.8, 43.0, 42.6, 46.8, 44.6, 44.4, 50.0, 53.7, 49.9, 56.4, 49.8, 55.2, 60.3, 59.8, 56.6, 56.0, 59.9, 62.7, 60.5, 64.2, 60.9, 63.1, 65.2, 63.8]
    plot(K_range_rows,average_counts_rows,title="Number of non-empty clusters for rows")
    
    # Cluster columns and plot it
    #average_counts_columns = run_all_clusterings(R=X_min, M=M, K_range=K_range_columns, N=N)
    #average_counts_columns = [1.0, 2.0, 3.0, 3.9, 4.8, 5.2, 6.6, 7.3, 7.3, 8.7, 9.5, 9.8, 10.0, 10.2, 11.6, 10.8, 12.5, 12.3, 12.0, 13.9, 13.0, 14.7, 14.6, 14.5, 16.3, 15.1, 16.2, 15.9, 17.5, 18.4, 16.8, 18.9, 17.2, 19.5, 19.5, 17.6, 19.8, 18.5, 19.3, 19.0, 20.2, 20.2, 20.7, 21.5, 21.0, 22.3, 22.5, 22.5, 21.7, 21.8, 22.5, 24.0, 24.2, 21.7, 22.8, 24.4, 23.6, 24.6, 22.4, 23.7, 24.0, 26.2, 23.9, 26.0, 25.1, 23.7, 26.4, 25.5, 25.2, 26.6, 27.9, 26.6, 26.4, 29.1, 27.2, 25.6, 26.2, 28.9, 29.1, 27.2, 30.7, 28.7, 29.0, 29.0, 28.5, 30.2, 30.0, 30.0, 30.0, 32.2, 32.4, 30.5, 30.4, 32.5, 31.1, 32.5, 32.3, 31.2, 30.9, 32.9, 31.5, 33.0, 30.3, 33.7, 33.2, 31.9, 31.3, 35.4, 36.0, 32.5, 32.2, 33.2, 31.7, 33.7, 31.9, 31.7, 31.6, 34.4, 36.4, 36.8, 34.8, 34.5, 31.7, 33.7, 32.6, 35.4, 34.5, 34.4, 36.0, 34.5, 35.5, 33.7, 35.5, 34.0, 36.0, 35.3, 36.5, 34.5, 38.0]
    #plot(K_range_columns,average_counts_columns,title="Number of non-empty clusters for columns (drugs)")
    