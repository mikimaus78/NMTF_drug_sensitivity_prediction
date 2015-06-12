"""
Run the NMTF algorithm on the full Sanger dataset.
Then output the matrices F, S, G, and predictions R_pred.
"""
import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
from pnmtf_i_div.code.pnmtf import PNMTF
import ml_helpers.code.mask as mask
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, load_kernels


# Settings
standardised = False

K = 20#50#
L = 5#20#

alpha = 1
beta = 1

max_iterations = 10000
updates = 1
epsilon_stop = 0.00001
stop_validation = True,
Kmeans = True
S_random = True
S_first = True
Rpred_Skl = False

random.seed(0)   

output_folder = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/pnmtf_FSG_%s_%s_%s_%s/" % (K,L,alpha,beta)
file_F = output_folder+"F.txt"
file_S = output_folder+"S.txt"
file_G = output_folder+"G.txt"

location_kernels = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/kernels/"


# Load in the Sanger dataset
(_,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    

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
    
# Multiply the kernels by alpha, beta
C1 = numpy.multiply(C1,alpha)
C2 = numpy.multiply(C2,beta)    

# We can also multiply the matrices together - not any better performance it seems, but may be worth running this all twice anyways
def multiply_matrices(l):
    result = l[0]
    for l2 in l[1:]:
        result *= l2
    return result
    
C1_mult = [multiply_matrices(C1)]
C2_mult = [multiply_matrices(C2)]
    
# Train the classifier
pnmtf = PNMTF(X_min,M,K,L,C1,C2)#C1_mult,C2_mult)
pnmtf.train(
    max_iterations=max_iterations,
    updates=updates,
    epsilon_stop=epsilon_stop,
    stop_validation=stop_validation,
    Kmeans=Kmeans,
    S_random=S_random,
    S_first=S_first,
    M_test=None,
    Rpred_Skl=Rpred_Skl
)
F = pnmtf.F
S = pnmtf.S
G = pnmtf.G
print "Statistics on training data: %s." % pnmtf.predict(X_min,M)

# Save F, S, G to the files
numpy.savetxt(file_F,F)
numpy.savetxt(file_S,S)
numpy.savetxt(file_G,G)