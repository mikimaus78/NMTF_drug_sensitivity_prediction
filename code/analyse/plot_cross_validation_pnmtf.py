"""
Plot the results (MSE) of the cross-validation that we ran on the Sanger 
dataset using the PNMTF algorithm, for:
- K = 20
- L = 30
- alpha in [0,0.01,0.1,1,10,100].
- beta in  [0,0.01,0.1,1,10,100]

Second line can be thrown away; first line is of the format:
    Tried parameters {'K': K, 'L': L}. Average performances: {'R^2': R2, 'MSE': MSE, 'Rp': Rp}.
    
Outcome:
[[ 2.29498712  2.26282785  2.25841549  2.2577247   2.26520525  2.27614638]
 [ 2.32123469  2.28715635  2.30131062  2.26844042  2.26273177  2.27950516]
 [ 2.32637946  2.32418652  2.30286101  2.30409859  2.29129242  2.27995425]
 [ 2.35542774  2.34118043  2.29709462  2.31681315  2.31439038  2.29994312]
 [ 2.32861289  2.31861843  2.31988026  2.33162267  2.31207462  2.33450798]
 [ 2.35801093  2.32108355  2.34913675  2.32726361  2.32276517  2.31074424]]
"""

import ast, numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm                   # For colourmap

location_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/cross_validation_pnmtf/"
file_name_not_std = location_file+"crossval_pnmtf_notstd.txt"

# Extract the performances, and return a tuple of two lists: [(K,L)], [{'MSE':MSE,'R^2':R2,'Rp':Rp}]
def extract_performances(location_file):
    fin = open(location_file,'r')
    lines = fin.readlines()[:-1] # remove last line with best performance
    lines = [line for i,line in enumerate(lines) if i % 2 == 0]
    lines = [line.split("Tried parameters ")[1] for line in lines]
    lines = [line.split(". \n")[0].split(". Average performances: ") for line in lines]
    
    alpha_beta_values = [(dictionary['alpha'],dictionary['beta']) for dictionary in [ast.literal_eval(line[0]) for line in lines]]
    performances = [ast.literal_eval(line[1]) for line in lines]
    return (alpha_beta_values,performances)
    
def plot_performances(alpha_beta_values,performances):
    MSEs = [performance['MSE'] for performance in performances]
    alpha_values = sorted(set([alpha for (alpha,beta) in alpha_beta_values]))
    beta_values = sorted(set([beta for (alpha,beta) in alpha_beta_values]))
    
    MSEs_grid = numpy.zeros((len(alpha_values),len(beta_values)))
    for (alpha,beta),MSE in zip(alpha_beta_values,MSEs):
        index_alpha = alpha_values.index(alpha)
        index_beta = beta_values.index(beta)
        MSEs_grid[index_alpha,index_beta] = MSE
        
    print MSEs_grid
    
    # Plot the performances as a contour plot
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = numpy.meshgrid(beta_values,alpha_values)
    surf = ax.plot_surface(X,Y,MSEs_grid, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('beta',fontsize=12)
    ax.set_ylabel('alpha',fontsize=12)
    ax.set_zlabel('Mean Square Error',fontsize=12)  
    
    # Setting the scales to log does not work...
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pyplot.title("Average MSE of NMTF against alpha, beta for K=20, L=30",y=1.03)
    
    ax.tick_params(labelsize=10)
    
    pyplot.show()
    
    
if __name__ == "__main__":
    (alpha_beta_values,performances) = extract_performances(file_name_not_std)
    plot_performances(alpha_beta_values,performances)