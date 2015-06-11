"""
Run the NMTF algorithm on the full Sanger dataset.
Then output the matrices F, S, G, and predictions R_pred.

Findings:
- S=1 converges slower than S=[0,1] randomly (similar for low K,L, slower for high)
  Especially so for the non-standardised dataset!
- Furthermore, for S=1 F,S,G are all roughly the same values; for S=[0,1] we
  get more differing values (more interpretation?)
- Kmeans initialisation (either with S=1 or S=[0,1]) gives very bad convergence
- Updating R_pred once per iteration rather than for each S_kl gives very
  similar performance and is much faster
- Using 'singleton' empty cluster reinitialisation is not any better (in fact, worse)
"""
import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
from nmtf_i_div.code.nmtf import NMTF
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger



# Settings
standardised = False

K = 10#50#
L = 5#20#

max_iterations = 10000
updates = 1
epsilon_stop = 0#0.0001
stop_validation = True
Kmeans = True
S_random = True
S_first = True
Rpred_Skl = False

random.seed(0)   
numpy.random.seed(0)

output_folder = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/nmtf_FSG_%s_%s/" % (K,L)
file_F = output_folder+"F.txt"
file_S = output_folder+"S.txt"
file_G = output_folder+"G.txt"
        
# Load in the Sanger dataset
(_,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    
# Train the classifier
nmtf = NMTF(X_min,M,K,L)
nmtf.train(
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
F = nmtf.F
S = nmtf.S
G = nmtf.G
print "Statistics on training data: %s." % nmtf.predict(X_min,M)

# Save F, S, G to the files
numpy.savetxt(file_F,F)
numpy.savetxt(file_S,S)
numpy.savetxt(file_G,G)