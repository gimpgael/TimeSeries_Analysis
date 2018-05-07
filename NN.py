# -*- coding: utf-8 -*-
"""
Neural Network class with a single hidden layer, but with the loss function 
tweaked to fit better the price models requirements.
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

np.random.seed(43)

class NN_TF():
    """Neural Network class coded in TensorFlow, but with the loss function
    made to handle the price variation in the classification process.
    
    Attributes
    -----------------    
    - layers: Number of nodes in the hidden layer.
    - batch_size: Size of batches during calibration of the Neural Network. By
    default 200.
    - epochs: Total number of times the model is going to see the input. By 
    default 1000.
    - learning_rate: Learning rate. By default 0.0001.
    """
    
    def __init__(self, layers = 30, batch_size = 200, epochs = 1000, learning_rate = 0.0001):
        """Initialize the network"""
        
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def randomize_data(self, X, y_delt, y_delt_weight):
        """Randomize variables"""
        
        perm = np.random.permutation(X.shape[0])
        
        return X[perm, :], y_delt[perm, :], y_delt_weight[perm, :]
    
    def extract_brent_data(self, df_all):
        """Extract the data from the Dataframe df_all, specialized for the Brent
        model I worked about"""
        
        # Extract the variables from the df_all DataFrame, containing evertyhing
        y = df_all['nearby'].as_matrix().reshape(-1,1)
        y_delt = df_all['px_delta'].as_matrix().reshape(-1,1)
        X = df_all[df_all.columns[2:]]
        
        # Standardize
        s = StandardScaler()
        X = s.fit_transform(X)
        
        # 2 dimensions y delta output. Note that we get rid off the case where
        # the price variation is 0
        y_3d = pd.DataFrame(data = 0, index = df_all.index, columns = [0,1,2])
        for dts in df_all.index:
            if df_all.loc[dts, 'px_delta'] < 0:
                y_3d.loc[dts,0] = df_all.loc[dts, 'px_delta']
            elif df_all.loc[dts, 'px_delta'] == 0:
                y_3d.loc[dts,1] = df_all.loc[dts, 'px_delta']
            elif df_all.loc[dts, 'px_delta'] > 0:
                y_3d.loc[dts,2] = df_all.loc[dts, 'px_delta']
                
        y_3d = y_3d.as_matrix()
        y_3d = y_3d[:, [0,2]]
        
        # Return variables
        return X, y, y_delt, y_3d    

    def construct_y_weight(self, y_3):
        """Build matrix with the absolute value of the price movement per row"""
        
        y_int = np.max(np.abs(y_3), axis = 1).reshape(-1,1)
        
        return np.repeat(y_int, y_3.shape[1], axis = 1)
        
    def fit(self, X, y_3, y_delt):
        """Fit the algorithm to the X inputs, taking into account the price 
        variation in the loss function"""
        
        # Create placeholders
        x_ = tf.placeholder(tf.float32, shape = [None, X.shape[1]])
        y_delt_ = tf.placeholder(tf.float32, shape = [None, y_3.shape[1]])
        y_delt_weight_ = tf.placeholder(tf.float32, shape = [None, y_3.shape[1]])
        
        # Create weight variable
        y_3_weight = self.construct_y_weight(y_3)
        
        # Store layers weight & bias
        weights = {'hidden_layer': tf.Variable(tf.random_normal([X.shape[1], self.layers])),
            'out': tf.Variable(tf.random_normal([self.layers, y_3.shape[1]]))}
        
        biases = {'hidden_layer': tf.Variable(tf.random_normal([self.layers])),
            'out': tf.Variable(tf.random_normal([y_3.shape[1]]))}
        
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x_, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)
        
        # Output layer with linear activation
        model_output = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

        loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, 
                                    labels = tf.abs(tf.sign(y_delt_))), y_delt_weight_))

        optmz = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        # Positions
        pos = tf.equal(tf.argmax(tf.sigmoid(model_output), axis = 1),0)
        
        # Run the graph
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            acc, pnl = [], []
            
            # Train for the number of epochs
            for epoch in range(self.epochs):
                
                # Initialize the total loss                
                total_loss = 0
                
                # Randomize inputs
                x_train, y_delt_train, y_delt_weight = self.randomize_data(X, y_3, y_3_weight)
                
                # Loop through batches
                for batch in range(X.shape[0] // self.batch_size + 1):
                    
                    _, l = sess.run([optmz, loss], 
                                    feed_dict = {x_ : x_train[batch * self.batch_size : (batch + 1) * self.batch_size, :],
                                                 y_delt_ : y_delt_train[batch * self.batch_size : (batch + 1) * self.batch_size, :],
                                                 y_delt_weight_ : y_delt_weight[batch * self.batch_size : (batch + 1) * self.batch_size, :]})
        
                    total_loss += l
                    
                # Keep track of loss
                acc.append(total_loss)
                
                # PnL computation, keep track of
                pos_int = sess.run(pos, feed_dict = {x_ : X})
                pos_int = np.where(pos_int,-1,1).reshape(-1,1)
                pnl.append(np.sum(np.multiply(pos_int, y_delt)))
                

            # Run weights and biases, and keep them
            self.W, self.b = sess.run([weights, biases])
   
        # Return the metrics kept at each epoch iteration                      
        return pnl, acc
    
    def predict(self, X):
        """Predict the output for a set of inputs"""
        
        # Initialize placeholders
        x_ = tf.placeholder(tf.float32, shape = [None, X.shape[1]])
        
        w1 = tf.Variable(tf.random_normal(self.W['hidden_layer'].shape))
        w2 = tf.Variable(tf.random_normal(self.W['out'].shape))
        
        b1 = tf.Variable(tf.random_normal(self.b['hidden_layer'].shape))
        b2 = tf.Variable(tf.random_normal(self.b['out'].shape))
        
        w1 = tf.assign(w1, self.W['hidden_layer'])
        w2 = tf.assign(w2, self.W['out'])
        
        b1 = tf.assign(b1, self.b['hidden_layer'])
        b2 = tf.assign(b2, self.b['out'])
        
        # Build the same network as during fitting
        layer_1 = tf.add(tf.matmul(x_, w1), b1)
        layer_1 = tf.nn.relu(layer_1)
        
        model_output = tf.equal(tf.argmax(tf.sigmoid(tf.add(tf.matmul(layer_1, w2), b2)), axis = 1), 0)
                        
        # Run the session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res = sess.run(model_output, feed_dict = {x_ : X})
            
        # Return positions
        return np.where(res,-1,1).reshape(-1,1)
        
        




