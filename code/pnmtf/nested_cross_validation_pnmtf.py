"""
We make a custom cross-validation for doing PNMTF.

For K,L we search in the range:
    K: [10,15,20]
    L: [20,25,30]
For alpha,beta:
    alpha: [0.01,0.1,1,10]
    beta:  0
    
parameter_search_KL is of the format:
    [ { 'K': K, 'L': L }, ... ]
parameter_search_ab is of the format:
    [ { 'C1': C1, 'C2': C2 }, ... ]
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")

from nmtf_i_div.code.nmtf import NMTF
from pnmtf_i_div.code.pnmtf import PNMTF
from ml_helpers.code.parallel_matrix_cross_validation import ParallelMatrixCrossValidation
from ml_helpers.code import mask
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, negate_Sanger, load_kernels

import numpy, itertools, random

class PNMTFNestedCrossValidation:
    def __init__(self,X,M,K,parameter_search_KL,parameter_search_ab,train_config,file_performance,files_nested_performances_KL,files_nested_performances_ab):
        self.X = numpy.array(X,dtype=float)
        self.M = numpy.array(M)
        self.K = K
        self.train_config = train_config
        self.parameter_search_KL = parameter_search_KL
        self.parameter_search_ab = parameter_search_ab
        self.files_nested_performances_KL = files_nested_performances_KL   
        self.files_nested_performances_ab = files_nested_performances_ab       
        
        self.fout = open(file_performance,'w')
        (self.I,self.J) = self.X.shape
        assert (self.X.shape == self.M.shape), "X and M are of different shapes: %s and %s respectively." % (self.X.shape,self.M.shape)
        
        self.all_performances = {}      # Performances across all folds - dictionary from evaluation criteria to a list of performances
        self.average_performances = {}  # Average performances across folds - dictionary from evaluation criteria to average performance
        
    # Run the cross-validation
    def run(self):
        folds_test = mask.compute_folds(self.I,self.J,self.K,self.M)
        folds_training = mask.compute_Ms(folds_test)       

        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            # Run the cross-validation search for K,L using NMTF
            crossval = ParallelMatrixCrossValidation(
                method=NMTF,
                X=self.X,
                M=train,
                K=self.K,
                parameter_search=self.parameter_search_KL,
                train_config=self.train_config,
                file_performance=self.files_nested_performances_KL[i],
                P=5
            )
            crossval.run()
            (best_KL,_) = crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)
            print "Best parameters for fold %s using NMTF were %s." % (i+1,best_KL)
            
            # Add K,L to parameter_search_ab
            for params in self.parameter_search_ab:
                params['K'] = best_KL['K']
                params['L'] = best_KL['L']
            
            # Run the cross-validation search for alpha,beta using PNMTF
            crossval = ParallelMatrixCrossValidation(
                method=PNMTF,
                X=self.X,
                M=train,
                K=self.K,
                parameter_search=self.parameter_search_ab,
                train_config=self.train_config,
                file_performance=self.files_nested_performances_ab[i],
                P=5
            )
            crossval.run()
            (best_c1c2,_) = crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)
            print "Best parameters for fold %s using PNMTF were %s." % (i+1,best_c1c2)
            
            # Train the model and test the performance on the test set
            performance_dict = self.run_model(train,test,best_c1c2)
            self.store_performances(performance_dict)
            self.log_performance(fold=i+1,perf=performance_dict)  
            
        self.log()
            
    # Initialises and runs the model, and returns the performance on the test set
    def run_model(self,train,test,parameters):  
                    
        self.train_config['M_test'] = test                    
                              
        model = PNMTF(self.X,train,**parameters)
        model.train(**self.train_config)
        return model.predict(self.X,test)
        
    # Store the performances we get back in a dictionary from criterion name to a list of performances
    def store_performances(self,performance_dict):
        for name in performance_dict:
            if name in self.all_performances:
                self.all_performances[name].append(performance_dict[name])
            else:
                self.all_performances[name] = [performance_dict[name]]
              
    # Compute the average performance of the given parameters, across the K folds
    def compute_average_performances(self):
        performances = self.all_performances     
        average_performances = { name:(sum(values)/float(len(values))) for (name,values) in performances.iteritems() }
        self.average_performances = average_performances
        
    # Logs the performance of one fold on the test set
    def log_performance(self,fold,perf):
        message = "Finished fold %s, with performances %s. \n" % (fold,perf) 
        self.fout.write(message)
        self.fout.flush()
        
    # Logs the performances on the test sets 
    def log(self):
        self.compute_average_performances()
        message = "Average performances: %s. \nAll performances: %s. \n" % (self.average_performances,self.all_performances)
        self.fout.write(message)
        self.fout.flush()
        
        
# Run the PNMTF nested cross-validation
if __name__ == "__main__":
    # Settings
    standardised = False
    train_config = {
        'max_iterations' : 10000,
        'updates' : 1,
        'epsilon_stop' : 0.0,#0.0001,#
        'stop_validation' : True,
        'Kmeans' : True,
        'S_random' : True,
        'S_first' : True,
        'M_test' : None,
        'Rpred_Skl' : False
    }
    K_range = [10,20]
    L_range = [20,30]
    alpha_range = [0.01,0.1,1]
    beta_range = [0]
    no_folds = 5
    output_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/nested_cross_validation_pnmtf/results.txt"
    files_nested_performances_KL = [
        "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/nested_cross_validation_pnmtf/fold_%s_KL.txt" % fold for fold in range(1,no_folds+1)]
    files_nested_performances_ab = [
        "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/nested_cross_validation_pnmtf/fold_%s_ab.txt" % fold for fold in range(1,no_folds+1)]
    location_kernels = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/kernels/"

    # Load in the Sanger dataset
    (X,_,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    X_min = negate_Sanger(X,M)
    
    # Load in the kernels
    C1 = load_kernels(location_kernels,["copy_variation","gene_expression","mutation"]) #cell lines
    C2 = load_kernels(location_kernels,["1d2d_descriptors","PubChem_fingerprints","targets"]) #drugs
            
    #C1 = numpy.subtract(C1,1) #Give rewards for clustering similar things (negative values)
    #C2 = numpy.subtract(C2,1)
    # Ensure the diagonals are always 0's
    for c1 in C1:
        numpy.fill_diagonal(c1,0)
    for c2 in C2:
        numpy.fill_diagonal(c2,0)
    
    # Construct the parameter search
    parameter_search_KL = [{'K':K,'L':L} for (K,L) in itertools.product(K_range,L_range)]
    parameter_search_ab = [{'C1':numpy.multiply(C1,alpha),'C2':numpy.multiply(C2,beta)} for (alpha,beta) in itertools.product(alpha_range,beta_range)]

    # Run the cross-validation framework
    random.seed(9000)
    numpy.random.seed(42)
    nested_crossval = PNMTFNestedCrossValidation(
        X=X_min,
        M=M,
        K=no_folds,
        parameter_search_KL=parameter_search_KL,
        parameter_search_ab=parameter_search_ab,
        train_config=train_config,
        file_performance=output_file,
        files_nested_performances_KL=files_nested_performances_KL,
        files_nested_performances_ab=files_nested_performances_ab
    )
    nested_crossval.run()