"""
Train a Random Forest on the drug/cell line features and use it to predict 
missing values.

We get three classes of classifiers:
 -  One per row (trained on drug features)
 -  One per column (trained on cell line features)
 -  One overall (trained on drug+cell line features)

Performance: 
    Drug:       
    Cell line:  
    Overall:        3.90916816246 (3.9060722858915335 1st fold) (excl gene expression, too big for memory)
    
Standardised:
    Drug:       
    Cell line:  
    Overall:        1.02499538091 (1.0199794068076518 1st fold) (excl gene expression, too big for memory)
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
import ml_helpers.code.mask as mask
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, load_features

from sklearn.ensemble import RandomForestRegressor

# Settings
standardised = False
no_trees = 10

location_features = '/home/thomas/Dropbox/Biological databases/Sanger_drug_sensivitity/'


# Method for doing cross-validation on the drug sensitivity dataset. We split
# M into <no_folds> folds, and then either:
#   train a classifier per row (on drug features)               classifier='row'
#   train a classifier per column (on cell line features)       classifier='column'
#   train one classifier overall (on drug+cell line features)   classifier='overall'

def run_cross_validation(X,M,drug_features,cell_line_features,no_folds,classifier='overall'):
    (I,J) = M.shape
    folds_M = mask.compute_folds(I,J,no_folds,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    MSEs = []
    for fold in range(0,no_folds):
    #for fold in [0]:
        print "Fold %s." % (fold+1)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        if classifier == 'overall':
            (X_train,Y_train) = construct_datapoints_overall(X,M_training,drug_features,cell_line_features)
            (X_test,Y_test) = construct_datapoints_overall(X,M_test,drug_features,cell_line_features)
        
        MSE = train_RF(X_train,Y_train,X_test,Y_test)
        MSEs.append(MSE)
    return MSEs

    
# Test the folds to ensure none has an entirely empty row or column.
def assert_no_empty_rows_columns(Ms):
    for M in Ms:
        (I,J) = M.shape
        OmegaI = [[] for i in range(0,I)]
        OmegaJ = [[] for j in range(0,J)]
        for i,j in itertools.product(range(0,I),range(0,J)):
            if M[i,j]:
                OmegaI[i].append(j)
                OmegaJ[j].append(i)
        for i,omega_i in enumerate(OmegaI):
            assert len(omega_i) != 0, "Fully unobserved row in M, row %s." % i
        for j,omega_j in enumerate(OmegaJ):
            assert len(omega_j) != 0, "Fully unobserved column in M, column %s." % j    
    

# Construct the data points from the mask, the sensitivity values, and the features
def construct_datapoints_overall(X,M,drug_features,cell_line_features):
    indices = mask.nonzero_indices(M)
    Y = [X[i][j] for (i,j) in indices]
    X = [numpy.append(drug_features[j],cell_line_features[i]) for (i,j) in indices]
    return (X,Y)
    

# Train a RF on the list of data points X and output values Y.
def train_RF(X_train,Y_train,X_test,Y_test):
    random_forest = RandomForestRegressor(n_estimators=no_trees)
    random_forest.fit(X_train,Y_train)
    
    Y_fit = random_forest.predict(X_train)
    print sum([ (y_fit - y_train)**2 for (y_fit,y_train) in zip(Y_fit,Y_train)]) / float(len(Y_fit))
    
    Y_pred = random_forest.predict(X_test)
    MSE = sum([ (y_pred - y_test)**2 for (y_pred,y_test) in zip(Y_pred,Y_test)]) / float(len(Y_pred))
    return MSE
    
    
if __name__ == "__main__":
    random.seed(0)    
    
    # Load in the Sanger dataset
    (X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    no_folds = 5
    
    # Load in the features
    file_1d2d = location_features+"drug_features/1d2d/drug_1d2d_filtered_std"
    file_fingerprints = location_features+"drug_features/fingerprints/drug_PubChem_fingerprints_filtered.csv"
    file_targets = location_features+"drug_features/targets/drug_targets_binary_filtered"
    
    file_copy_variation = location_features+"cell_line_features/copy_variation_std"    
    file_gene_expression = location_features+"cell_line_features/gene_expression_std"    
    file_mutation = location_features+"cell_line_features/mutation"    
    
    drug_features = numpy.append(
        numpy.append(load_features(file_1d2d),
                     load_features(file_fingerprints,delim=','),axis=1),
        load_features(file_targets),axis=1)
        
    # Gene expression profile takes up too much memory
    #cell_line_features = numpy.append(
    #    numpy.append(load_features(file_copy_variation),
    #                 load_features(file_gene_expression),axis=1),
    #    load_features(file_mutation),axis=1)
    cell_line_features = numpy.append(load_features(file_mutation),load_features(file_mutation),axis=1)
    
    # Run the cross-validation on the data
    overall_MSEs = run_cross_validation(X,M,drug_features,cell_line_features,no_folds,classifier='overall')
    print "Overall RF: %s (%s)." % ((sum(overall_MSEs)/float(len(overall_MSEs))),overall_MSEs)