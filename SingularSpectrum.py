"""
Singular Spectrum Analysis class, 
More information regarding the analysis can be found here:
https://en.wikipedia.org/wiki/Singular_spectrum_analysis

"""

import scipy.linalg as linalg
import numpy as np
from matplotlib import pyplot as plt

class SSA():
    """Singular Spectrum Analysis decomposition algorithn for a time series.
    
    Attributes
    -----------------  
    - dim: embedding dimension to consider
    - n_components: number of principal eigenvectors to consider for reconstruction
    - eps: epsilon, to be used to computes the value to ensure convergence
    - n_iter: maximum number of iteration
    
    Methods
    -------------
    - decomposition    
    - series_reconstruction
    - predict
    - nans
    - plot_forecast
    - plot_singular_spectrum
    
    """
    
    def __init__(self, dim, n_components = 2, eps = 0.0001, n_iter=10000):
        """Initializer"""
        self.dim = dim
        self.n_compo = list(range(n_components))
        self.eps = eps
        self.n_iter = n_iter
        
    def decomposition(self, y):
        """Singular Spectrum Analysis decomposition"""
        
        # Total size of the time series
        n_points = len(y)
        t = n_points - (self.dim - 1)
        
        # Construct the Hankel matrix, remove last rows and scale it
        mx = linalg.hankel(y, np.zeros(self.dim))
        mx = mx[:-self.dim+1,:] / np.sqrt(t)
        
        # Singular Value Decomposition.
        # s: singular values
        _, s, v = linalg.svd(mx, full_matrices=False, lapack_driver='gesvd')
        
        # Principal components
        # vector: matrix of singular vector
        # pc: matrix of principal components
        vector = np.asarray(np.matrix(v).T)
        pc = np.asarray(np.matrix(mx) * vector)

        return pc, s, vector
    
    def series_reconstruction(self, y):
        """Series reconstruction for SSA decomposition using vector of 
        components"""
        
        # Compute the principal components matrix, the singular values and the
        # matrix of singular vectors
        pc, _, vector = self.decomposition(y)
                
        # Consider only the first n_compo vectors
        pc_comp = np.asarray(np.matrix(pc[:, self.n_compo]) * 
                             np.matrix(vector[:, self.n_compo]).T)

        # Initialize variables
        y_r = np.zeros(y.shape[0])
        times = np.zeros(y.shape[0])
        t = y.shape[0] - (self.dim - 1)
        
        # Reconstruction loop
        for i in range(self.dim):
            y_r[i : t + i] = y_r[i : t + i] + pc_comp[:, i]
            times[i : t + i] = times[i : t + i] + 1
            
        y_r = (y_r / times) * np.sqrt(t)
            
        return y_r

    def predict(self, x, n_forecast):
        """Data prediction based on SSA over a certain number of 
        observations"""
        
        # Interval to compare with
        e = self.eps * (np.max(x) - np.min(x))
        
        # Data transformation
        mean_x = x.mean()
        x = x - mean_x
        
        # Initialize forecast
        xf = self.nans(n_forecast)

        # Loop through the number of points to forecast
        for i in range(n_forecast):
            
            # Previous value as initial estimation
            x = np.append(x, x[-1])
            yq = x[-1]
            y = yq + 2 * e
            
            # Maximum number of iterations
            n_iter = self.n_iter
            
            while abs(y - yq) > e:
                yq = x[-1]
                
                xr = self.series_reconstruction(x)
                
                y = xr[-1]
                x[-1] = y
                
                # Iteration control
                n_iter -= 1
                if n_iter <= 0:
                    print('The SSA prediction algorithm has reached its maximum number of iterations')
                    break
    
            xf[i] = x[-1]
            
        xf = xf + mean_x
                    
        return xf 

    def nans(self, n_dim):
        """Array of nans"""
        return np.nan * np.ones(n_dim)
    
    def plot_forecast(self, y, n_forecast):
        """Plots the variable y and add the forecast to it, into a graph"""
        
        # Computes forecast
        yf = self.predict(y, n_forecast)
        
        # Initialize x axis
        x0 = range(len(y))
        x1 = range(len(y), len(y) + n_forecast)
        
        # Graph
        plt.figure()
        plt.plot(x0, y)
        plt.plot(x1, yf, 'r--')
        plt.xlabel('Observations')
        plt.legend(['Data', 'Forecast'])
        
    def plot_singular_spectrum(self, y):
        """Plots the singular spectrum, i.e. the eigenvalues variance 
        normalized and sorted"""
        
        # Computes the singular values
        _, s, _ = self.decomposition(y)
        
        # Normalized cumulative explanation
        s_norm = s / s.sum() * 100
        
        # Plot cumulative graph
        plt.figure()
        plt.plot(range(self.dim), s_norm, 'r', marker = '*')
        plt.title('Singular Spectrum')
        plt.xlabel('Eigenvalue number')
        plt.ylabel('Eigenvalue')