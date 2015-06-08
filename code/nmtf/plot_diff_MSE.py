"""
Plot the diff MSE against the MSE and R^2 on the training and test datasets.
"""
import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
from matplotlib import pyplot
from nmtf_i_div.code.nmtf import NMTF
from ml_helpers.code import mask
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = False

K = 20
L = 30

max_iterations = 10000
updates = 1
epsilon_stop = 0.00001
Kmeans = True
S_random = True
S_first = True
Rpred_Skl = False

random.seed(0)   

no_folds = 5 # hold back 20% of the data for performance evaluation

# Load in the Sanger dataset
(_,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
(I,J) = X_min.shape
    
# Split into training and test datasets
folds_test = mask.compute_folds(I,J,no_folds,M)
folds_training = mask.compute_Ms(folds_test)
(M_train,M_test) = (folds_training[0],folds_test[0])
    
# Train the classifier
nmtf = NMTF(X_min,M_train,K,L)
nmtf.train(
    max_iterations=max_iterations,
    updates=updates,
    epsilon_stop=epsilon_stop,
    Kmeans=Kmeans,
    S_random=S_random,
    S_first=S_first,
    M_test=M_test,
    Rpred_Skl=Rpred_Skl
)

# Extract the performances
all_diff_MSE = nmtf.all_diff_MSE[2:] #get rid of 0th and 1st iteration (at 1st diff_MSE is huge)
all_MSE = nmtf.all_MSE[2:]
all_R2 = nmtf.all_R2[2:]
all_MSE_pred = nmtf.all_MSE_pred[2:]
all_R2_pred = nmtf.all_R2_pred[2:]
iterations = range(1,len(all_diff_MSE)+1)

# Plot diff_MSE against MSE
pyplot.plot(iterations,all_MSE,label="MSE",color="blue")
pyplot.plot(iterations,all_MSE_pred,label="MSE pred",color="green")
pyplot.legend(loc=2)
pyplot.twinx()
pyplot.plot(iterations,all_diff_MSE,label="Diff MSE",color="red")
pyplot.legend(loc=3)
pyplot.xscale('log')
pyplot.show()

# Plot diff_MSE against R^2
pyplot.subplot(1,1,1)
pyplot.plot(iterations,all_R2,label="R2",color="blue")
pyplot.plot(iterations,all_R2_pred,label="R2 pred",color="green")
pyplot.legend(loc=2)
pyplot.twinx()
pyplot.plot(iterations,all_diff_MSE,label="Diff MSE",color="red")
pyplot.legend(loc=3)
pyplot.xscale('log')
pyplot.show()