"""
Load in the drug sensitivity dataset.
Take out 10% of the values.
Use Non-Negative Matrix Factorisation (I-divergence implementation) to 
predict the missing values.
"""

import numpy, random, imp
NMF = imp.load_source('NMF','/home/thomas/Documenten/PhD/nmf_i_div/code/nmf.py').NMF

dataset_location = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/"
file_name = "gi50_no_missing.txt"

""" Generate <fraction> random indices, and construct a matrix M for this."""
def generate_M(I,J,fraction,seed=0):
    random.seed(seed)
    M = numpy.ones([I,J])
    values = random.sample(xrange(0,I*J),int(I*J*fraction))
    for v in values:
        M[v / J][v % J] = 0
    return M

if __name__ == "__main__":
    # Load in data
    data = numpy.loadtxt(dataset_location+file_name,dtype=str)
    drug_ids = data[1:,0]
    cell_lines = data[0,1:]
    X = numpy.array(data[1:,1:],dtype='f')
    (I,J) = X.shape # 2200 drugs, 60 cancer cell lines
    
    # Take out 10% of the values, random but seeded
    fraction = 0.1
    M = generate_M(I,J,fraction,seed=0)
    
    # Run NMF
    K = 10
    nmf = NMF(X,M,K)
    
    print nmf.Omega