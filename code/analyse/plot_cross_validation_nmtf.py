"""
Plot the results (MSE) of the cross-validation that we ran on the Sanger 
dataset using the NMTF algorithm, for:
- K in [10,20,30,40,50,60,70]
- L in [5,10,15,20,25,30]

Second line can be thrown away; first line is of the format:
    Tried parameters {'K': K, 'L': L}. Average performances: {'R^2': R2, 'MSE': MSE, 'Rp': Rp}.
"""

import ast, numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm                   # For colourmap

location_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/"
file_name_not_std = location_file+"crossval_nmtf_notstd.txt"

# Extract the performances, and return a tuple of two lists: [(K,L)], [{'MSE':MSE,'R^2':R2,'Rp':Rp}]
def extract_performances(location_file):
    fin = open(location_file,'r')
    lines = fin.readlines()
    lines = [line for i,line in enumerate(lines) if i % 2 == 0]
    lines = [line.split("Tried parameters ")[1] for line in lines]
    lines = [line.split(". \n")[0].split(". Average performances: ") for line in lines]
    
    K_L_values = [(dictionary['K'],dictionary['L']) for dictionary in [ast.literal_eval(line[0]) for line in lines]]
    performances = [ast.literal_eval(line[1]) for line in lines]
    return (K_L_values,performances)
    
def plot_performances(K_L_values,performances):
    MSEs = [performance['MSE'] for performance in performances]
    K_values = sorted(set([K for (K,L) in K_L_values]))
    L_values = sorted(set([L for (K,L) in K_L_values]))
    
    MSEs_grid = numpy.zeros((len(K_values),len(L_values)))
    for (K,L),MSE in zip(K_L_values,MSEs):
        index_K = K_values.index(K)
        index_L = L_values.index(L)
        MSEs_grid[index_K,index_L] = MSE
        
    print MSEs_grid
    
    # Plot the performances as a contour plot
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = numpy.meshgrid(L_values,K_values)
    surf = ax.plot_surface(X,Y,MSEs_grid, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('L',fontsize=12)
    ax.set_ylabel('K',fontsize=12)
    ax.set_zlabel('Mean Square Error',fontsize=12)  
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pyplot.title("Average MSE of NMTF against K,L",y=1.03)
    
    ax.tick_params(labelsize=10)
    
    pyplot.show()
    
    
if __name__ == "__main__":
    (K_L_values,performances) = extract_performances(file_name_not_std)
    plot_performances(K_L_values,performances)