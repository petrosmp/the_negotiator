### Template imports
import logging
from random import randint, random
import traceback
from typing import cast, Dict, List, Set, Collection

### ML imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

### Genius Web 
# from geniusweb.party.DefaultParty import DefaultParty

### File Manager
import os
import importlib

### Maths
import random
import numpy as np

"""
    Helper Methods
"""

"""
Get the strategies from the subfolder and assign an id to them
"""
def find_available_agents(basedir):
    agents = []
      # Adjusted to full path
    strategy_folders = [f.path for f in os.scandir(basedir) if f.is_dir()]
    id = 0
    for folder in strategy_folders:
        strategy_name = os.path.basename(folder)
        # Assuming there's only one Python file per strategy folder
        
        for python_file in os.listdir(folder):
            if python_file.endswith('.py'):
                module_name = os.path.splitext(python_file)[0]
                class_path = f"strategies.{strategy_name}.{module_name}.{module_name}"
                # TODO import class
                agents.append({"class": class_path, "agent_id": id})
                id += 1
    return agents

def playStrategy(strategy_id):
    return random.randint(0, 10)
"""
    Meta Agent Class

    A meta-agent that will learn to select the best agent for the occasion
"""
class MetaAgent():

    # Initial function
    def __init__(self): #, reporter: Reporter = None):
        self.agent_classes = find_available_agents(os.path.join(os.path.dirname(os.path.abspath(__file__)),"strategies"))
        self.number_of_agents = len(self.agent_classes)

        """ UCB inits"""
        k = self.number_of_agents
        self.play_count = np.zeros((k,))             # arm pulls
        self.total_plays = 0                         # total arm pulls
        self.average_performance = np.zeros((k,))    # average performance for negotiation round
        self.ucb = np.zeros((k,))                    # ucb estimate for each arm
    

    # pick the arm/server with the biggest UCB
    def pick_strategy(self):
        picked_strategy = np.argmax(self.ucb)
        return picked_strategy

    # Given agent_id, update its UCB score
    def UCB_round(self, pick_strategy, reward):

        self.play_count[picked_strategy] += 1
        self.total_plays += 1

        # Update average reward
        old_avg = self.average_performance[picked_strategy]
        new_avg = (old_avg * (self.play_count[picked_strategy] - 1) + reward)/self.play_count[picked_strategy]
        self.average_performance[picked_strategy] = new_avg
        
        # Update UCB value
        self.ucb[picked_strategy] = new_avg + np.sqrt(2 * np.log(self.total_plays)/self.play_count[picked_strategy])
        return picked_strategy

"""
    ML class
    
    Will help by giving UCB a headstart, using data from negotiations on known domains
    For each domain, we will get scores for each agent based on the characteristics of the domain,
    which will be converted into UCB "starting" confidence bounds, passing the batton to UCB which will continue playing different algorithms
"""

class AgentLearning:
    r_state = 69 # nice
    def __init__(self) :
        # Choosing an ML model
        self._model = RandomForestClassifier(random_state = self.r_state)
        # TODO
        self.data = pd.DataFrame()

    def add_data(self, agent_id, domain_data, score):
        """ Add new data to the dataset, replicating entries based on score. """
        # TODO implement domain feature extraction
        new_entry = {'agent_id': agent_id, 'score': score, 'domain_data': domain_data}
        new_entry_df = pd.DataFrame([new_entry])

        # Replicate the entry based on the score
        replication_factor = self._calculate_replication_factor(score)
        replicated_data = pd.concat([new_entry_df] * replication_factor, ignore_index=True)
        self.data = pd.concat([self.data, replicated_data], ignore_index=True)
    
    # Give more weight to good scores by replicating the data entered in the dataset
    @staticmethod
    def _calculate_replication_factor(score):
        #return 5 if score > 0.9 else 4 if score > 0.85 else 2 if score > 0.75 else 1
        return 2 if score > 0.95 else 1

    # When the dataset is "complete" perform training
    def train_model(self):
        # all data except agent_id
        X = self.data.drop('agent_id', axis=1)
        # agent_id is the target, will try to predict it
        y = self.data['agent_id']
        # Model training step
        self._model.fit(X, y)

    # Gets "probabilities" for each agent
    # This is useful to initialize the UCB algorithm with meaningful data, learned from the utility gained on specific negotiation domains 
    def predict_probabilities(self, domain_data):
        features = {'domain_data': domain_data}
        features_df = pd.DataFrame([features])
        # The random forest algorithm can give probabilities for each agent
        probabilities = self._model.predict_proba(features_df)
        return probabilities
    
    # Unused, for testing purposes
    def predict_agent(self, domain_data):
        #TODO test features
        features = {'domain_data': domain_data}
        features_df = pd.DataFrame([features])
        return self._model.predict(features_df)

"""
Alternative NN approach
"""
import tensorflow as tf

# TODO prepare domain_data to enter the NN
class AgentLearningNN:
    r_state = 69 # nice
    def __init__(self, num_agents, input_features, hidden_layer_size=64) :
        self._num_agents = num_agents  # Number of agents determines the output layer size
        self._input_features = input_features  # Number of input features
        self._model = None
        # Choosing an ML model
        self._setupNeuralNetwork(hidden_layer_size)
        # TODO check
        #self.data = pd.DataFrame()

    def _setupNeuralNetwork(self, hidden_layer_size):
        # The NN will have as many neurons as number of features and number of neurons as  number of outputs
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

    
    def saveNN(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
        model_dir = os.path.join(script_dir, 'model')  
        os.makedirs(model_dir, exist_ok=True) 
        file_path = os.path.join(model_dir, "AgentNN")
        self._model.save(file_path)
  
    def loadNN(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
        model_dir = os.path.join(script_dir, 'model')  
        file_path = os.path.join(model_dir, "AgentNN")
        self._model = tf.keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}")

    # On new data, do a forward pass (and backpropagation)
    # X and y must be converted to numpy for it to work
    # y is the target variable "agent_id"
    def train_model(self, agent_scores, domain_data):
        # new_entry = {'agent_scores': agent_scores, 'domain_data': domain_data}
        # new_entry_df = pd.DataFrame([new_entry])
        # self.data = pd.concat([self.data, new_entry_df], ignore_index=True)
        # domain_features = np.array([domain_data])

        feature_values = np.array(list(domain_data.values()))

        domain_features = feature_values.reshape(1, -1)
        X = domain_features
        y = np.array([agent_scores])

        print("X:", X.shape)
        print("y:", y.shape)
        # Model training step
        # TODO change epochs
        #self.model.fit(X, y, epochs=10)
        self._model.fit(X, y)

    # Gets predicted scores for each agent
    # This is useful to initialize the UCB algorithm with meaningful data, learned from the utility gained on specific negotiation domains 
    def predict_scores(self, domain_data):
        # Tensorflow needs numpy array to work
        feature_values = np.array(list(domain_data.values()))
        feature_values = feature_values.reshape(1, -1)
        # The neural network can give probabilities for each agent
        scores_prediction = self._model.predict(feature_values)
        return scores_prediction
    
    # Unused, for testing purposes
    def predict_agent(self, domain_data):
        # Tensorflow needs numpy array to work
        features = np.array([domain_data])
        #features = {'domain_data': domain_data}
        probabilities = self.model.predict(features)
        return np.argmax(probabilities)

# Usage
# print("Current Working Directory:", os.getcwd())
# base_directory = 'c:/Users/jchri/Documents/GitHub/the_negotiator/strategies'

meta = MetaAgent()

picked_strategy = meta.pick_strategy()
reward = playStrategy(picked_strategy)
meta.UCB_round(picked_strategy, reward)

# Neural Network parameters setup and model initialization
numOfAgents = 5
hiddenLayerSize = 5
domainFeatureNum = 4
agent_nn = AgentLearningNN(numOfAgents, domainFeatureNum, hiddenLayerSize)

"""
 A testing suite
"""

data_samples = []
agent_nn.loadNN()
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