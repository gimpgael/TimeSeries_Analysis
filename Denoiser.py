# -*- coding: utf-8 -*-
"""
Simple class to be used to denoise time series. The objective is to keep 
implementing methods, in order to have a generic tool.
"""

# Import librairies
import numpy as np
import pandas as pd
import pywt

class Denoiser():
    """Algorithm, quite generic, but designed to use several methodologies for
    time series denoising. The idea is to keep adding methods to this tool, 
    that can then be used working.
    
    Attributes
    ----------------- 
    - period: level used for denoising. Depending on the methodology used, the
    meaning can be quite different
    
    Methods
    -----------------     
    - fit_wavelet: Haar wavelet approach
    - fit_ma: Simple moving average approach
    """
    
    def __init__(self, period):
        """Initialize the algorithm"""
        
        self.period = period
        
    def fit_wavelet(self, x):
        """Denoise the x time series, using the wavelet Haar approach"""

        # Initialize result
        res = x.reshape(-1,1).copy()

        # Loop
        for i in range(self.period + 1):
            
            # Computes the wavelet transformation - need to do for each iteration
            coeffs = pywt.wavedec(x, 'haar', level = self.period)
            coeffs_new = list(coeffs)
            
            # Loop to fix all coefficients except the one we want to use at 0
            for j in range(self.period + 1):
                if j != i:
                    coeffs_new[j] *= 0
                    
            # Reconstructed signal
            rec = pywt.waverec(coeffs_new, 'haar').reshape(-1,1)
            res = np.append(res, rec, axis = 1)
        
        # Return all but initial signal
        return res[:,1:]
    
    def fit_ma(self, x):
        """Denoise the x time series, using a moving average approach"""
        
        # Goes through a DataFrame, to use build in methods
        x = pd.DataFrame(data = x, columns = ['init'])
        
        # Create the moving average, and the residual        
        x['ma'] = x['init'].rolling(window = 5, center = False).mean()
        x['resid'] = x['init'] - x['ma']
        x = x.as_matrix()
        
        # Return all but initial signal
        return x[:,1:]
    

        
    
    