### Template imports
import logging
from random import randint, random
import traceback
from typing import cast, Dict, List, Set, Collection

### ML imports
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

### Genius Web 
# from geniusweb.party.DefaultParty import DefaultParty

### File Manager
import os
import importlib

### Maths
import random
import numpy as np


"""
    ML class
    
    Will help by giving UCB a headstart, using data from negotiations on known domains
    For each domain, we will get scores for each agent based on the characteristics of the domain,
    which will be converted into UCB "starting" confidence bounds, passing the batton to UCB which will continue playing different algorithms
"""

class magicianNN:
    r_state = 69 # nice
    weights_dict = {
            "num_of_issues": 0.1,  
            "num_of_bids": 0.001,
            "avg_vals_per_issue": 0.2,  
            "avg_bid_util": 1,  
        }
    def __init__(self, num_agents, input_features, hidden_layer_size=16) :
        """ Initialization method 
        Args: num_agents, input_features
        """
        self._num_agents = num_agents           # Number of agents determines the output layer size
        self._input_features = input_features   # Number of input features
        self._model = None
        # Choosing an ML model
        self._setupNeuralNetwork(hidden_layer_size)

    def _setupNeuralNetwork(self, hidden_layer_size):
        """
        Neural Network Setup method using keras 
        The NN will have as many neurons as number of features and number of neurons as  number of outputs
        """
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(self._input_features,)),
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self._num_agents, activation='softmax')  # Output layer with sigmoid activation to map to value between 0 and 1, one node per agent
        ])
        # Use some default optimizer (adam) and mean_squared_error loss function
        # Also tested SGD
        self._model.compile(optimizer='adam',
                      loss='mean_squared_error')
                      #metrics=['accuracy'])
  
    def loadNN(self):
        """ Load parameters into the model from stored file."""
        script_dir = os.path.dirname(os.path.realpath(__file__)) 
        model_dir = os.path.join(script_dir, 'model')  
        file_path = os.path.join(model_dir, "magicianNN")
        self._model = tf.keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}.")


    def predict_scores(self, domain_data):
        """ Gets predicted scores for each agent to initialize the UCB algorithm with meaningful data, learned from the utility gained on specific negotiation domains. """
        # Tensorflow needs numpy array to work
        # Selected keys/features from the dictionary
        keys = ["num_of_issues", "num_of_bids",  "avg_vals_per_issue", "avg_bid_util"]


        domain_features = [domain_data[key]*self.weights_dict[key] for key in keys]
        domain_features = np.array([domain_features])
        scores_prediction = self._model.predict(domain_features)
        return scores_prediction

    def saveNN(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
        model_dir = os.path.join(script_dir, 'model')  
        os.makedirs(model_dir, exist_ok=True) 
        file_path = os.path.join(model_dir, "magicianNN")
        self._model.save(file_path)
  
    # On new data, do a forward pass (and backpropagation)
    # X and y must be converted to numpy for it to work
    # y is the target variable "agent_id"
    def train_model(self, agent_scores, domain_data):
        # Selected keys/features from the dictionary
        keys = ["num_of_issues", "num_of_bids",  "avg_vals_per_issue", "avg_bid_util"]
        domain_features = [domain_data[key]*self.weights_dict[key] for key in keys]

        # feature_values = np.array(list(domain_data.values()))

        #domain_features = feature_values.reshape(1, -1)
        X = np.array([domain_features])
        y = np.array([agent_scores])

        print("X:", X.shape)
        print("y:", y.shape)
        # Model training step
        # TODO change epochs
        #self.model.fit(X, y, epochs=10)
        self._model.fit(X, y)

# Usage
# print("Current Working Directory:", os.getcwd())
# base_directory = 'c:/Users/jchri/Documents/GitHub/the_negotiator/strategies'

# Neural Network parameters setup and model initialization
numOfAgents = 5
hiddenLayerSize = 5
domainFeatureNum = 4
agent_nn = magicianNN(numOfAgents, domainFeatureNum, hiddenLayerSize)

"""
 A testing suite
"""

data_samples = []
#agent_nn.loadNN()
# Generate 100 random data samples
for i in range(10):
    # Generate random key-value pairs for feat_dict
    feat_dict = {
        "num_of_issues": np.random.randint(1, 11),  # Random number of issues (1-10)
        "num_of_bids": np.random.randint(100, 1000),  # Random number of possible bids (100-1000)
        "avg_vals_per_issue": np.random.uniform(0.5, 5.0),  # Random average values per issue (0.5-5.0)
        #"weight_std_dev": np.random.uniform(0.1, 1.0),  # Random weight standard deviation (0.1-1.0)
        "avg_bid_util": np.random.uniform(0.2, 0.8),  # Random average bid utility (0.2-0.8)
        #"bid_util_std_dev": np.random.uniform(0.05, 0.2)  # Random bid utility standard deviation (0.05-0.2)
    }

    # Generate agent_scores with Agent 3 having the best score (e.g., 0.9) and Agent 4 having the worst score (e.g., 0.1)
    num_agents = 5
    agent_scores = np.random.uniform(0.1, 0.9, size=num_agents)
    agent_scores = np.zeros(5)
    agent_scores[0] = 0.9  # Set Agent 3's score to the highest value
    agent_scores[3] = 0.3  # Set Agent 4's score to the lowest value
    agent_nn.train_model(agent_scores,feat_dict)
    # Append the generated data as a tuple (feat_dict, agent_scores)
    #data_samples.append((feat_dict, agent_scores))
    print(agent_nn.predict_scores(feat_dict))
#agent_nn.saveNN()