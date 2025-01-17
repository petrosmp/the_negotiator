### ML imports
import pandas as pd
import tensorflow as tf

### File Manager
import os
import importlib
import glob
import importlib.util

### Maths
import numpy as np
from random import randint, random

### Time
import datetime

"""
    ML class
    
    Will help by giving UCB a headstart, using data from negotiations on known domains
    For each domain, we will get scores for each agent based on the characteristics of the domain,
    which will be converted into UCB "starting" confidence bounds, passing the batton to UCB which will continue playing different algorithms
"""
class magicianNN:
    r_state = 69 # nice
    # scaling the value in decimal steps
    weights_dict = {
        "avg_bid_util": 1,
        "avg_vals_per_issue": 0.01,
        "bid_util_std_dev": 1,
        "num_of_bids": 0.0001,
        "num_of_issues": 0.1,  
        "weight_std_dev": 1,
    }
    # scaling the value with numbers with respect to the thresholds provided by the statistics
    numerical_normalization_dict = {
        "avg_bid_util": 1.9,
        "avg_vals_per_issue": 7.5,
        "bid_util_std_dev": 3,
        "num_of_bids": 1,
        "num_of_issues": 1.4,
        "weight_std_dev": 3,
    }
    agent_keys = ["Agent007", "DreamTeam109Agent", "TemplateAgent", "GEAAgent", "Agent33"]
    keys = ["avg_bid_util", "avg_vals_per_issue", "bid_util_std_dev", "num_of_bids", "num_of_issues", "weight_std_dev"]

    def __init__(self, num_agents, input_features, hidden_layer_size=16) :
        """ 
        Initialization method 
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
        domain_features = [domain_data[key]*self.weights_dict[key]*self.numerical_normalization_dict[key] for key in self.keys]

        domain_features = np.array([domain_features])
        scores_prediction = self._model.predict(domain_features)
        return scores_prediction

    def saveNN(self,filename="magicianNN"):
        """ Save the NNs parameters in the given folder. """
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
        model_dir = os.path.join(script_dir, 'model')  
        os.makedirs(model_dir, exist_ok=True) 
        file_path = os.path.join(model_dir, filename)
        self._model.save(file_path)
  

    def train_model(self, agent_dict, domain_data, num_epochs=8):
        """ 
        Training method using dictionary data 
        On new data, do a forward pass (and backpropagation)
        X and y must be converted to numpy for it to work
        y is the target variable "agent_id"
        """
        # Selected keys/features from the dictionary
        agent_scores = [agent_dict[agent_name] for agent_name in self.agent_keys]
        domain_features = [domain_data[key]*self.weights_dict[key]*self.numerical_normalization_dict[key] for key in self.keys]

        X = np.array([domain_features])
        y = np.array([agent_scores])

        # Model training step
        # TODO change epochs
        self._model.fit(X, y, epochs=num_epochs)

# Usage
# print("Current Working Directory:", os.getcwd())
# base_directory = 'c:/Users/jchri/Documents/GitHub/the_negotiator/strategies'

# Neural Network parameters setup and model initialization
numOfAgents = 5
hiddenLayerSize = 12
domainFeatureNum = 6
agent_nn = magicianNN(numOfAgents, domainFeatureNum, hiddenLayerSize)


"""
Training Data Parser
"""

def processDataFiles(directory_path):
    """ Get data files from the given folder """
    # List all .py files in the directory
    data_files = glob.glob(os.path.join(directory_path, '*.py'))
    for file_path in data_files:
        module_name = os.path.basename(file_path).split('.')[0] #remove .py
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for results varible
        if hasattr(module, 'results'):
            extractTrain(module.results)
            # Debugging/Testing:
            print(extractPredict(module.results))
        else:
            print(f"No 'results' variable in {module_name}")

def extractTrain(results):
    """ Extract agent results and domain features dictionary. """
    for item in results:
        if 'features' in item and 'results' in item:
            agent_nn.train_model(item['results'], item['features'], 2)

def extractPredict(features_dict):
    """ Extract domain features dictionary and predict. """
    for item in features_dict:
        if 'features' in item and 'results' in item:
            print(agent_nn.predict_scores(item['features']))

# Path to running script directory 
script_directory = os.path.dirname(os.path.abspath(__file__))
# Subfolder training_data
directory_path = os.path.join(script_directory, 'training_data')

# Processs dataset
processDataFiles(directory_path)

# Save neural network parameters with timestamp
current_time = datetime.datetime.now()
date_time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
agent_nn.saveNN("magicNN_"+str(date_time_string))

"""
 An initial random testing suite
"""

# data_samples = []
# #agent_nn.loadNN()
# # Generate 100 random data samples
# for i in range(1000):
#     # Generate random key-value pairs for feat_dict
#     feat_dict = {
#         "num_of_issues": np.random.randint(1, 11),  # Random number of issues (1-10)
#         "num_of_bids": np.random.randint(100, 1000),  # Random number of possible bids (100-1000)
#         "avg_vals_per_issue": np.random.uniform(0.5, 5.0),  # Random average values per issue (0.5-5.0)
#         "weight_std_dev": np.random.uniform(0.1, 1.0),  # Random weight standard deviation (0.1-1.0)
#         "avg_bid_util": np.random.uniform(0.2, 0.8),  # Random average bid utility (0.2-0.8)
#         "bid_util_std_dev": np.random.uniform(0.05, 0.2)  # Random bid utility standard deviation (0.05-0.2)
#     }

#     # Generate agent_scores with Agent 3 having the best score (e.g., 0.9) and Agent 4 having the worst score (e.g., 0.1)
#     num_agents = 5
#     agent_scores = np.random.uniform(0.1, 0.9, size=num_agents)
#     agent_scores = np.zeros(5)
#     agent_scores[0] = 0.9  # Set Agent 3's score to the highest value
#     agent_scores[3] = 0.3  # Set Agent 4's score to the lowest value
#     agent_nn.train_model(agent_scores,feat_dict)
#     # Append the generated data as a tuple (feat_dict, agent_scores)
#     #data_samples.append((feat_dict, agent_scores))
#     print(agent_nn.predict_scores(feat_dict))
# #agent_nn.saveNN()
