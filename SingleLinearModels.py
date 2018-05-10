# -*- coding: utf-8 -*-
"""
This algorithm fits the variable to be explained versus any single variable
of a DataFrame, and predicts on the same way. It gives an idea on how every
variable is forecasting the y.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

class SingleLinearModels():
    """Algorithm computing a single linear regression per explanatory variable,
    and then adding everything together, whatever the type of
    Note that there is no initial transformation of variable.
    """
    
    def __init__(self, var_name = 'y'):
        """Initialise the algorithm"""
        
        # Name of the variable to be explained
        self.var_name = var_name
        
        self.models_dict = {}
        
    def create_variables(self, X):
        """List all variables to be used, under the assumption X is a DataFrame
        """
        
        return [x for x in X.columns if x != self.var_name]
    
    def fit(self, X):
        """Fit one single regression per variable. Note that X shoud be a 
        DataFrame"""
        
        # Variables to study
        var_list = self.create_variables(X)
        
        # For any single variable, create a linear model
        for var in var_list:
            
            # Remove NaNs and add a constant
            df_int = X[[self.var_name, var]].dropna()
            df_int['cst'] = 1
            
            # Fit and store
            model = LinearRegression().fit(df_int[var, 'cst'].as_matrix(), 
                                     df_int[self.var_name].as_matrix())
            self.models_dict[var] = model
            
    def predict(self, X):
        """Forecast function"""
        
        # Variables to be used for forecasting
        var_list = self.create_variables(X) 
        
        # Initialize output
        res = pd.DataFrame(data = None, index = X.index, columns = var_list)
        
        # Loop through all variables
        for var in var_list:
            df_int = pd.DataFrame(X[var])
            df_int['cst'] = 1
            
            res[var] = self.models_dict[var].predict(df_int.as_matrix())
            
        # Return DataFrame of forecasts, as well as averages
        return res, pd.DataFrame(res.mean(axis = 1), columns = ['mean'])
            
            
            
        
        
        
        
        