"""
Train a Random Forest on the drug/cell line features and use it to predict 
missing values.

We get three classes of classifiers:
 -  One per row (trained on drug features)
 -  One per column (trained on cell line features)
 -  One overall (trained on drug+cell line features)


Unstandardised: 
(1 tree)    
    Drug:       
    Cell line:  
    Overall:
        {'R^2': 0.52081818610268837, 'MSE': 5.6041680134813916, 'Rp': 0.75234436348493916}
        (MSE training fit: 2.65074125939, 2.64108089865, 2.7062477235, 2.66424191233, 2.62473277383)
        
(10 trees)
    Drug:       
    Cell line:  
    Overall:
        {'R^2': 0.66382279330707594, 'MSE': 3.9316221019364432, 'Rp': 0.81790339620515162}
        (MSE training fit: 1.48528274561, 1.46113098926, 1.47355839296, 1.48968450238, 1.47096323709)
    
    
Standardised:
(1 tree)    
    Drug:       
    Cell line:  
    Overall:
        {'R^2': -0.56502818605963934, 'MSE': 1.5646201311354495, 'Rp': 0.1551035487975784}
        (MSE training fit: 0.753144988716, 0.734979315975, 0.752811727733, 0.763343843335, 0.757657307427)      
         
(10 trees)
    Drug:       
    Cell line:  
    Overall:        
        {'R^2': -0.022346523236208293, 'MSE': 1.0220555459037743, 'Rp': 0.25727955849020434}
        (MSE training fit: 0.410628305961, 0.402244048939, 0.406781104331, 0.412605650006, 0.408939117546)
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, load_features

from sklearn.ensemble import RandomForestRegressor

# Settings
standardised = False
no_trees = 1

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
    
    performances = []
    for fold in range(0,no_folds):
    #for fold in [0]:
        print "Fold %s." % (fold+1)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        if classifier == 'overall':
            construct_datapoints = construct_datapoints_overall
            train_RF = train_RF_overall
            
        (X_train,Y_train) = construct_datapoints(X,M_training,drug_features,cell_line_features)
        (X_test,Y_test) = construct_datapoints(X,M_test,drug_features,cell_line_features)
        
        performance = train_RF(X_train,Y_train,X_test,Y_test)
        performances.append(performance)
        
    return performances

    
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
    

##### Methods for one overall RF
# Construct the data points from the mask, the sensitivity values, and the features
def construct_datapoints_overall(X,M,drug_features,cell_line_features):
    indices = mask.nonzero_indices(M)
    Y = [X[i][j] for (i,j) in indices]
    X = [numpy.append(drug_features[j],cell_line_features[i]) for (i,j) in indices]
    return (X,Y)

# Train a RF on the list of data points X and output values Y.
def train_RF_overall(X_train,Y_train,X_test,Y_test):
    random_forest = RandomForestRegressor(n_estimators=no_trees)
    random_forest.fit(X_train,Y_train)
    
    Y_fit = random_forest.predict(X_train)
    print "MSE fit: %s." % statistics.MSE(Y_train,Y_fit)
    
    Y_pred = random_forest.predict(X_test)
    MSE = statistics.MSE(Y_test,Y_pred)
    R2 = statistics.R2(Y_test,Y_pred)
    Rp = statistics.Rp(Y_test,Y_pred)
    return {'MSE':MSE,'R^2':R2,'Rp':Rp}
    
    
##### Methods for one RF per row
# We now make each datapoint X into a pair (X,i) where i is the row index.
# We only use the cell line features
def construct_datapoints_row(X,M,drug_features,cell_line_features):
    indices = mask.nonzero_indices(M)
    Y = [X[i][j] for (i,j) in indices]
    X = [(cell_line_features[i],i) for (i,j) in indices]
    return (X,Y)
    
#TODO: this!
    
##### Methods for one RF per column

    
    
##############
if __name__ == "__main__":
    random.seed(0)    
    
    # Load in the Sanger dataset
    (X,_,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
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
    #cell_line_features = numpy.append(numpy.append(load_features(file_copy_variation),load_features(file_gene_expression),axis=1),load_features(file_mutation),axis=1)
    cell_line_features = numpy.append(load_features(file_mutation),load_features(file_mutation),axis=1)
    
    # Run the cross-validation on the data
    overall_performances = run_cross_validation(X,M,drug_features,cell_line_features,no_folds,classifier='overall')
    
    # Find the average performances
    def find_averages(performances):
        averages = {'MSE':0.0,'R^2':0.0,'Rp':0.0}
        n = float(len(performances))
        for performance in performances:
            for name in performance:
                averages[name] += performance[name]
        for name,value in averages.iteritems():
            averages[name] /= n
        return averages
        
    overall_averages = find_averages(overall_performances)
        
    print "Overall RF: %s." % (overall_averages)