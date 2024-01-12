### Template imports
import logging
from random import randint, random
import traceback
from typing import cast, Dict, List, Set, Collection

### ML imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    A meta-agent that will learn to select the best agent for the occasion
    """ 

    """
    From template/ time dependent agent
    """

    # def __init__(self, reporter: Reporter = None):
    #     super().__init__(reporter)
    #     self._profileint: ProfileInterface = None  # type:ignore
    #     self._utilspace: LinearAdditive = None  # type:ignore
    #     self._me: PartyId = None  # type:ignore
    #     self._progress: Progress = None  # type:ignore
    #     self._lastReceivedBid: Bid = None  # type:ignore
    #     self._extendedspace: ExtendedUtilSpace = None  # type:ignore
    #     self._e: float = 1.2
    #     self._lastvotes: Votes = None  # type:ignore
    #     self._settings: Settings = None  # type:ignore
    #     self.getReporter().log(logging.INFO, "party is initialized")

    # # Override
    # def getCapabilities(self) -> Capabilities:
    #     return Capabilities(
    #         set(["SAOP", "Learn", "MOPAC"]),
    #         set(["geniusweb.profile.utilityspace.LinearAdditive"]),
    #     )

    # #TODO learning
    # def predict_best_agent(settings):
    #     # Extract features from settings
    #     features = extractFeatures(settings)
        
    #     # Reshape data for prediction (assuming features is a dictionary)
    #     # features_df = pd.DataFrame([features])
        
    #     # Predict the best agent
    #     # best_agent_id = model.predict(features_df)[0]
    #     # return best_agent_id

    # # Parse the settings dictionary
    # def extractFeatures(settings):
    #     return {
    #         '': settings[""],
    #         '': settings[""],
    #     }

    # # Override
    # def getDescription(self) -> str:
    #     return (
    #         ""
    #     )

    # def notifyChange(self, data: Inform):
    #         """MUST BE IMPLEMENTED
    #         This is the entry point of all interaction with your agent after is has been initialised.
    #         How to handle the received data is based on its class type.

    #         Args:
    #             info (Inform): Contains either a request for action or information.
    #         """

    #         ############ a Settings message is the first message that will be send to your #############
    #         # agent containing all the information about the negotiation session.
    #         if isinstance(data, Settings):
    #             ################
    #             agents = settings["agents"]
    #             profiles = settings["profiles"]
    #             deadline_time_ms = settings["deadline_time_ms"]
    #             ################
    #             self.settings = cast(Settings, data)
    #             self.me = self.settings.getID()

    #             # progress towards the deadline has to be tracked manually through the use of the Progress object
    #             self.progress = self.settings.getProgress()

    #             self.parameters = self.settings.getParameters()
    #             self.storage_dir = self.parameters.get("storage_dir")

    #             # the profile contains the preferences of the agent over the domain
    #             profile_connection = ProfileConnectionFactory.create(
    #                 data.getProfile().getURI(), self.getReporter()
    #             )
    #             self.profile = profile_connection.getProfile()
    #             self.domain = self.profile.getDomain()
    #             profile_connection.close()

    #         # ActionDone informs you of an action (an offer or an accept)
    #         # that is performed by one of the agents (including yourself).
    #         elif isinstance(data, ActionDone):
    #             action = cast(ActionDone, data).getAction()
    #             actor = action.getActor()

    #             # ignore action if it is our action
    #             if actor != self.me:
    #                 # obtain the name of the opponent, cutting of the position ID.
    #                 self.other = str(actor).rsplit("_", 1)[0]

    #                 # process action done by opponent
    #                 self.opponent_action(action)
    #         # YourTurn notifies you that it is your turn to act
    #         elif isinstance(data, YourTurn):
    #             # execute a turn
    #             self.my_turn()

    #         # Finished will be send if the negotiation has ended (through agreement or deadline)
    #         elif isinstance(data, Finished):
    #             self.save_data()
    #             # terminate the agent MUST BE CALLED
    #             self.logger.log(logging.INFO, "party is terminating:")
    #             super().terminate()
    #         else:
    #             self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))
    # """
    # Testing
    # """
    # #TODO gpt sample data
    # data = {
    #     'agent_id': ['agent1', 'agent2', 'agent1', 'agent3'],
    #     # TODO performance metric check
    #     'performance_metric': [0.8, 0.75, 0.85, 0.78],
    #     # 'domain_feature1': [1, 2, 1, 2],
    #     # 'domain_feature2': [3, 4, 3, 4],
    # }
    # df = pd.DataFrame(data)
    # # TODO read and store data
    # df.to_csv('agent_performance_data.csv', index=False)
    # loaded_df = pd.read_csv('agent_performance_data.csv')

# Usage
# print("Current Working Directory:", os.getcwd())
# base_directory = 'c:/Users/jchri/Documents/GitHub/the_negotiator/strategies'

meta = MetaAgent()

picked_strategy = meta.pick_strategy()
reward = playStrategy(picked_strategy)
meta.UCB_round(picked_strategy, reward)
