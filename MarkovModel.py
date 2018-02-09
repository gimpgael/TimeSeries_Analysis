# -*- coding: utf-8 -*-
"""
Markov Model for forecasting, designed after the idea of the following paper:
    https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=SILC2016&paper_id=38
    
The idea is basically to estimate a model based on the full historical sample 
and with several runs, to avoid being stucked in a local optima (as advised),  
and then try to look a similar period backward in order to forecast the next 
one.  

"""

from hmmlearn.hmm import GaussianHMM

import numpy as np
import pandas as pd

import math
import warnings

class MarkovModel():
    """Algorith model for variable forecasting based on Markov Models.
    Note that this algorithm works with numpy arrays as inputs.
    
    Attributes
    -----------------  
    - n_run:         Number of run to find the best Markov Model fitting the data
    - n_compo:       Number of components
    - n_iter:        Number of iterations
    - var_diff:      Boolean variable to determine if we use the input directly
                     or if we consider its first order difference
    - calib_period:  Period to consider when looking for matching period for 
                     forecasting next step
                     
    Methods
    -------------
    - check_var_diff
    - fit
    - select_training_variable
    - select_model                    
    - forecast                 
    - backtest                   
        
    """

    def __init__(self, n_run = 20, n_compo = 4, n_iter = 1000, var_diff = True, calib_period = 25, interval_period = 100):
        """Initialiser"""
        self.n_run_ = n_run
        self.n_compo_ = n_compo
        self.n_iter_ = n_iter
        self.check_var_diff(var_diff)           # Check if the var_diff variable has been correctly inputed
        self.var_diff_ = var_diff
        self.calib_period = calib_period
        self.interval_period = interval_period
        self.model = []
        self.backtest_res = []

    def check_var_diff(self, var_diff):
        """Function checking if the variable var_diff has been correctly 
        inputed"""
        
        if type(var_diff) != bool:
            print('The variable var_diff needs to be a boolean')
            return -1

    def fit(self, X):
        """Function fitting the algorithm on the raw data X"""
        
        # Select training variables
        x_train = self.select_training_variable(X)
                
        # Select the model to use over several run
        self.model = self.select_model(x_train)
        
    def select_training_variable(self, X):
        """Function selecting the variable to be trained, depending if we work
        on the original input or on the differences"""
        
        # Depending on the case, consider the variable itself or its first order
        # difference
        if self.var_diff_ == True:
            x_train = np.diff(X, n = 1, axis = 0)
        else:
            x_train = X
            
        return x_train
            
    def select_model(self, X):
        """Function selecting the model to use, based on several run on the
        input variable"""
        
        # Initial the best element
        test, startprob, transmat, means, covars = -math.inf, 0, 0, 0, 0
        
        # Ignore depreciation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
        # Test over n_runs models
        for _ in range(self.n_run_):
        
            # Initialise the model
            model = GaussianHMM(n_components = self.n_compo_, covariance_type = "diag", n_iter = self.n_iter_).fit(X)
    
            # Test the likelihood versus previous results, if good keeps in memory optimisation parameters
            if model.score(X) > test:
                test = model.score(X)
                startprob = model.startprob_
                transmat = model.transmat_
                means = model.means_
                covars = model._covars_
    
        # Initialise model to be used
        model = GaussianHMM(n_components = self.n_compo_, covariance_type = "diag", n_iter = self.n_iter_)
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model._covars_ = covars
        
        return model
    
    def forecast(self, X):
        """Forecast the next move of the input X"""
        
        # Select training variables
        x_train = self.select_training_variable(X)
        
        # If model hasn't previously been calibrated, then do it
        if type(self.model) == list:
            self.fit(x_train)
            
        # Likelihood of current data
        likelihood1 = self.model.score(x_train[-self.calib_period:])
        
        # Test and index of closest period
        test, index = math.inf, 0
            
        # Find the optimal period
        for i in range(x_train.shape[0] - self.calib_period):
            
            # Likelihood of tested data
            likelihood2 = self.model.score(x_train[i:i+self.calib_period])
            
            # Keep the best
            if np.abs(likelihood1 - likelihood2) < test:
                test = np.abs(likelihood1 - likelihood2)
                index = i
            
        if self.var_diff_ == True:
            return X[-1] + x_train[index+self.calib_period]
        else:
            return X[-1] + x_train[index+self.calib_period] - x_train[index+self.calib_period-1]
        
    def backtest(self, X):
        """Do the full time series backtest"""
        
        # Initialise output
        res = np.zeros((1, X.shape[1]))
        
        # Loop to do the backtest
        for i in range(self.interval_period, X.shape[0]+1):
            
            # Need to initialise each time the algo to null
            self.model = []
            
            # Select the time series
            x_int = X[i-self.interval_period:i]
            
            # Add to the forecast
            res = np.append(res, self.forecast(x_int).reshape(1,-1), axis = 0)
            
            self.backtest_res = res
            
        # Do not return the first row
        return res[1:]
    