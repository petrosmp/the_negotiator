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
        # Choosing an ML model
        self.model = self._setupNeuralNetwork(hidden_layer_size)
        # TODO check
        #self.data = pd.DataFrame()

    def _setupNeuralNetwork(self, hidden_layer_size):
        # The NN will have as many neurons as number of features and number of neurons as  number of outputs
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(self._input_features,)),
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(self._num_agents, activation='softmax')  # Output layer with one node per agent
        ])
        # Use some default optimizer (adam) and sparse_categorical_crossentropy loss function with accuracy as a metric
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # On new data, do a forward pass (and backpropagation)
    # X and y must be converted to numpy for it to work
    # y is the target variable "agent_id"
    def train_model(self, agent_id, score, domain_data):
        new_entry = {'agent_id': agent_id, 'score': score, 'domain_data': domain_data}
        new_entry_df = pd.DataFrame([new_entry])
        self.data = pd.concat([self.data, new_entry_df], ignore_index=True)
        # all data except agent_id
        X = (self.data.drop('agent_id', axis=1)).to_numpy
        # agent_id is the target, will try to predict it
        y = (self.data['agent_id']).to_numpy
        # Model training step
        # TODO change epochs
        self.model.fit(X, y, epochs=10)

    # Gets "probabilities" for each agent
    # This is useful to initialize the UCB algorithm with meaningful data, learned from the utility gained on specific negotiation domains 
    def predict_probabilities(self, domain_data):
        # Tensorflow needs numpy array to work
        features = np.array([domain_data])
        # The neural network can give probabilities for each agent
        probabilities = self.model.predict_probabilities(features)
        return probabilities
    
    # Unused, for testing purposes
    def predict_agent(self, domain_data):
        # Tensorflow needs numpy array to work
        features = np.array([domain_data])
        #features = {'domain_data': domain_data}
        probabilities = self.predict_probabilities(features)
        return np.argmax(probabilities)

# Usage
# print("Current Working Directory:", os.getcwd())
# base_directory = 'c:/Users/jchri/Documents/GitHub/the_negotiator/strategies'

meta = MetaAgent()

picked_strategy = meta.pick_strategy()
reward = playStrategy(picked_strategy)
meta.UCB_round(picked_strategy, reward)

ml = AgentLearningNN(10, 5)

