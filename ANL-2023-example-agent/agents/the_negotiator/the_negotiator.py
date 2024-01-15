import logging
from random import randint, random, choice as random_choice
import traceback
from typing import cast, Dict, List, Set, Collection

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.ActionWithBid import ActionWithBid
from geniusweb.actions.LearningDone import LearningDone
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.actions.Vote import Vote
from geniusweb.actions.Votes import Votes
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.OptIn import OptIn
from geniusweb.inform.Settings import Settings
from geniusweb.inform.Voting import Voting
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds
from geniusweb.utils import val
from geniusweb.profileconnection.ProfileInterface import ProfileInterface
from geniusweb.profile.utilityspace.LinearAdditive import LinearAdditive
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from geniusweb.profile.utilityspace.DiscreteValueSetUtilities import DiscreteValueSetUtilities
from geniusweb.progress.Progress import Progress
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from time import sleep, time as clock
from decimal import Decimal
import sys
from agents.time_dependent_agent.extended_util_space import ExtendedUtilSpace
from tudelft_utilities_logging.Reporter import Reporter
from geniusweb.references.Parameters import Parameters
from pathlib import Path
import numpy as np

# ML
import tensorflow as tf
from .arsenal import arsenal


UCB_DIR_PREFIX = "UCB_data"

"""
Neural Network used for performance prediction
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
    # Keys to access dictionary data for agents and domain data
    agent_keys = ["Agent007", "DreamTeam109Agent", "TemplateAgent", "GEAAgent", "Agent33"]
    keys = ["avg_bid_util", "avg_vals_per_issue", "bid_util_std_dev", "num_of_bids", "num_of_issues", "weight_std_dev"]

    def __init__(self, num_agents, input_features, hidden_layer_size=16) :
        """ 
        Initialization method 
        Args: num_agents, input_features
        Neural Network Setup method using keras 
        The NN will have as many neurons as number of features and number of neurons as  number of outputs
        """
        self._num_agents = num_agents           # Number of agents determines the output layer size
        self._input_features = input_features   # Number of input features
        
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(self._input_features,)),
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self._num_agents, activation='softmax')  # Output layer with sigmoid activation to map to value between 0 and 1, one node per agent
        ])
        # Use some default optimizer (adam) and mean_squared_error loss function
        self._model.compile(optimizer='adam',
                      loss='mean_squared_error')

    def loadNN(self):
        """ Load parameters into the model from storage directory."""
        modelname = "magicNN_beta"
        model_dir = Path(__file__).parent / 'model' / modelname
        try:
            self._model = tf.keras.models.load_model(model_dir)
            print(f"Model loaded successfully from {model_dir}.")
        except Exception as e:
            print(e)
            

    def predict_scores(self, domain_data):
        """ Gets predicted scores for each agent to initialize the UCB algorithm with meaningful data, learned from the utility gained on specific negotiation domains. """
        # Tensorflow needs numpy array to work
        # Selected keys/features from the dictionary
        domain_features = [domain_data[key]*self.weights_dict[key]*self.numerical_normalization_dict[key] for key in self.keys]

        domain_features = np.array([domain_features])
        scores_prediction = self._model.predict(domain_features)
        return scores_prediction
    
class TheNegotiator(DefaultParty):
    """Class implementing the negotiator agent."""

    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)
        self._profileint: ProfileInterface = None  # type:ignore
        self._utilspace: LinearAdditive = None  # type:ignore
        self._me: PartyId = None  # type:ignore
        self._progress: Progress = None  # type:ignore
        self._lastReceivedBid: Bid = None  # type:ignore
        self._extendedspace: ExtendedUtilSpace = None  # type:ignore
        self._e: float = 1.2
        self._lastvotes: Votes = None  # type:ignore
        self._settings: Settings = None  # type:ignore
        self._connection_data: ActionWithBid = None
        self.getReporter().log(logging.INFO, "party is initialized")
        #TODO init model, load model

    # Competition type (do not change this function)
    def getCapabilities(self) -> Capabilities:
        """
        Parties are given turns in a round-robin order. When a party has the turn, 
        it can accept, offer, or end negotiation. A deal is reached if all parties,
        accept an offer. 
        
        More info at https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb#SessionProtocol
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def notifyChange(self, info: Inform):
        """
        This is the entry point of all interaction with your agent after is has been initialized.

        Args:
            info (Inform): Contains either a request for action or information.
            See more details at: https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb#Inform
        """
        
        # parse Inform object based on its type
        try:
            if isinstance(info, Settings):
                # Enter setting information. This is called once at the beginning of a session.
                # This is where we first see the domain, and thus where we extract the features
                # and select the agent that we will play as.

                self._settings = info

                # unpack the settings and store them as class variables for future reference
                self._me = self._settings.getID()
                self._progress = self._settings.getProgress()               # progress towards the deadline has to be tracked manually through the use of the Progress object
                self._parameters = self._settings.getParameters()            # what are parameters? https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb/wiki/WikiStart#PartyParameters
                self._storage_dir = self._parameters.get("storage_dir")

                # check that the directory in which the UCB data will be saved exists
                self._ucb_data_dir = Path(self._storage_dir) / UCB_DIR_PREFIX

                if not self._ucb_data_dir.exists():
                    self._ucb_data_dir.mkdir()

                # in order to get the profile (preferences) and the domain info we need a connection
                profile_connection = ProfileConnectionFactory.create(
                    info.getProfile().getURI(), self.getReporter()
                )
                self._profile: LinearAdditiveUtilitySpace = cast(LinearAdditiveUtilitySpace, profile_connection.getProfile())
                self._domain = self._profile.getDomain()
                self._all_bids = AllBidsList(self._domain)                  # compose a list of all possible bids
                profile_connection.close()            

                # we now have enough data to calculate anything we want. its time to pick a strat.
                
                # if we have played in this domain as this profile before, we already have UCB set up
                self._ucb_data_file = self._ucb_data_dir / f"{self._domain.getName()}_{self._profile.getName()[-1]}.ucb"
                if self._ucb_data_file.exists():
                    success, previous_ucb_data, play_counts, total_plays, estimates = UCB_parse(self._ucb_data_file, arsenal)
                    
                    # if for some reason the existing UCB data is corrupted, just call the magician
                    if not success:
                        self._reporter.log(logging.WARNING, f"there was an error parsing the data at {self._ucb_data_file}: {previous_ucb_data}. Calling the magician...")
                        features = self._extract_features()
                        previous_ucb_data = self._magician(features)

                    self._init_UCB(arsenal, previous_ucb_data, play_counts, total_plays, estimates)
                else:       # if this is a profile we haven't seen before, we need the magician
                    self._reporter.log(logging.INFO, f"ucb data path '{self._ucb_data_file}' does not exist, using magician")
                    # extract features from the domain and get some estimates about agent fitness
                    features = self._extract_features()
                    initial_values = self._magician(features)

                    # initialize the UCB machinery with the predictions
                    self._init_UCB(arsenal, initial_values, [10 for _ in arsenal], 50, None, first_time=True)   # initialize with total plays >0 to give some weight to the NN's predictions

                # use UCB to pick the strategy
                self._strat: DefaultParty = self._UCB_pick_strategy()

                # set ourselves as a proxy for the strategy agent
                self._strat.set_proxy(self)
                self._strat_str = f"{self._strat.__class__.__module__}.{self._strat.__class__.__name__}"
                
                # create a new directory for the strategy agent
                own_storage_dir = Path(self._parameters.get("storage_dir"))
                strategy_storage_dir = own_storage_dir / self._strat.__class__.__name__
                
                # if it already exists, no need to create it anew
                if not strategy_storage_dir.exists():
                    strategy_storage_dir.mkdir()

                # create a new set of settings (only the parameters differ) to pass to the strategy agent
                newinfo = Settings(
                    self._me,        # id
                    info.getProfile(),  # profile
                    info.getProtocol(), # protocol
                    info.getProgress(), # progress
                    Parameters({"storage_dir": strategy_storage_dir}) # parameters
                )

                # pass the info to the strat
                self._strat.notifyChange(newinfo)

            elif isinstance(info, ActionDone):
                self._strat.notifyChange(info)
            elif isinstance(info, YourTurn):
                
                self._strat.notifyChange(info)

                self.getConnection().send(self._connection_data)

            elif isinstance(info, Finished):
                # The negotiation session has now ended. Get the utility of the deal, store it somewhere and go next.
                try:
                    deal: Bid = next(iter(info.getAgreements().getMap().values()))
                    utility = float(self._profile.getUtility(deal))
                    self.getReporter().log(logging.INFO, f"Final outcome: bid={deal} giving us a utility of: {utility} (special thanks to {self._strat.__class__.__name__})")
                except StopIteration:
                    utility = 0.0     # no reservation values in our profiles
                    self.getReporter().log(logging.INFO, "no agreement reached!")
                
                # update the UCB estimates based on this pull right here
                self._UCB_round(self._strat, utility)

                self._reporter.log(logging.INFO, f"got reward of {round(utility, 2)}, updated UCB to {[round(x, 2) for x in self._ucb]}")

                # save the updated UCB data for future use
                UCB_write(self._ucb_data_file, arsenal, self._ucb, self._play_count, self._total_plays, self.estimates)

                # pass the info to the strategy agent so it can terminate gracefully
                self._strat.notifyChange(info)

                # stop this party and free resources.
                self.terminate()
        except Exception as ex:
            self.getReporter().log(logging.CRITICAL, "Failed to handle info", ex)
        self._updateRound(info)

    # Override
    def getDescription(self) -> str:
        return (
            "Time-dependent conceder. Aims at utility u(t) = scale * t^(1/e) "
            + "where t is the time (0=start, 1=end), e is the concession speed parameter (default 1.1), and scale such that u(0)=minimum and "
            + "u(1) = maximum possible utility. Parameters minPower (default 1) and maxPower (default infinity) are used "
            + "when voting"
        )

    # Override
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profileint != None:
            self._profileint.close()
            self._profileint = None

    ##################### private support funcs #########################

    def _updateRound(self, info: Inform):
        """
        Update {@link #progress}, depending on the last received {@link Inform}

        @param info the received info.
        """
        if self._settings == None:  # not yet initialized
            return

        if not isinstance(info, OptIn):
            return

        # if we get here, round must be increased.
        if isinstance(self._progress, ProgressRounds):
            self._progress = self._progress.advance()

    def _init_UCB(self, arms, initial_values, play_counts, total_plays, estimates, first_time=False):
        """
        Initialize the UCB machinery.

        Sets self._arms and self._ucb to the given parameters and initializes counters
        and other bookkeeping stuff.

        Initial values is assumed to be an array of numbers with length equal to the
        number of arms.
        """

        """ UCB inits"""
        k = len(arsenal)
        self._play_count = play_counts                  # arm pulls
        self._total_plays = total_plays                 # total arm pulls
        self.estimates = initial_values if first_time else estimates               # average performance for negotiation round
        self._ucb = [float("inf") for _ in range(k)] if first_time else initial_values   # ucb estimate for each arm

    def _UCB_round(self, picked_strategy, reward):
        """
        Update the UCB for the given strategy agent.

        To be used after the end of a negotiation session, when the reward has been
        received.
        """

        picked_index = arsenal.index(picked_strategy.__class__)

        self._play_count[picked_index] += 1
        self._total_plays += 1

        # Update average reward
        old_estimate = self.estimates[picked_index]
        new_estimate = (old_estimate * (max(1, self._play_count[picked_index] - 1)) + reward)/max(2, self._play_count[picked_index])
        self.estimates[picked_index] = new_estimate

        self._reporter.log(logging.INFO, f"estimate for {picked_strategy.__class__.__name__} is {new_estimate}")
        self._reporter.log(logging.INFO, f"the sqrt factor is {np.sqrt(2 * np.log(self._total_plays)/self._play_count[picked_index])}")

        # Update UCB value
        self._ucb[picked_index] = new_estimate + np.sqrt(2 * np.log(self._total_plays)/self._play_count[picked_index])

        self._reporter.log(logging.INFO, f"updated UCB to {self._ucb}, since we played arm #{picked_index} (aka {picked_strategy.__class__}) and we got reward {reward}")

    def _UCB_pick_strategy(self):
        """Pick a strategy that fits the current domain and profile according to our UCB's"""


        clas = arsenal[np.argmax(self._ucb)]

        instance = clas()

        self._reporter.log(logging.INFO, f"picked strategy {clas.__name__} because UCB is {[round(x, 2) for x in self._ucb]}")
        return instance

    def _extract_features(self):
        """Extract the features that are useful in picking a strategy from the current domain"""
        
        feat_dict = {}

        weights: Dict[str, Decimal] = self._profile.getWeights()

        # 1) number of issues in domain
        feat_dict.update({"num_of_issues": len(weights)})

        # 2) average number of values per issue (sponsored by Misko)
        feat_dict.update({
            "avg_vals_per_issue": np.average([len(x.getUtilities()) for x in [cast(DiscreteValueSetUtilities, y) for y in self._profile.getUtilities().values()]])
        })

        # 3) number of possible bids
        feat_dict.update({"num_of_bids": self._all_bids.size()})

        # 4) standard deviation of weights
        feat_dict.update({"weight_std_dev": np.std(list(weights.values()))})

        # 5) average bid utility
        bid_utils = []
        for i in range(self._all_bids.size()): 
            bid = self._all_bids.get(i)
            bid_utils.append(np.float64(self._profile.getUtility(bid)))

        feat_dict.update({"avg_bid_util": np.average(bid_utils)})

        # "6) standard deviation of bid utility
        feat_dict.update({"bid_util_std_dev": np.std(bid_utils)})

        return feat_dict

    def _magician(self, features):
        """
        Will help by giving UCB a headstart, using data from negotiations on known domains
        For each domain, we will get scores for each agent based on the characteristics of the domain,
        which will be converted into UCB "starting" confidence bounds, passing the batton to UCB which will continue playing different algorithms

        Predict the performance of each agent in the arsenal based
        on the given set of features.
        """
        features = {k: np.float64(v) for k,v in features.items()}
        # TODO use magicianNN
        # Neural Network parameters setup and model initialization
        numOfAgents = 5
        hiddenLayerSize = 12
        domainFeatureNum = 6

        self._nn = magicianNN(numOfAgents, domainFeatureNum, hiddenLayerSize)
        self._nn.loadNN()
        
        _estimates = self._nn.predict_scores(features)

        try:
            estimates = next(iter(_estimates))
        except StopIteration:
            self._reporter.log(logging.CRITICAL, f"Could not convert NN output '{_estimates}' (of type {type(_estimates)}) to array!")
            print(f"estimates {_estimates[0]}")
            return None


        print(f"estimates {estimates} (type {estimates.__class__.__name__})")
        for i,j in enumerate(estimates):
            print(f"estimate {i}: {j} (type: {j.__class__.__name__})")
        
        return estimates

    def set_connection_data(self, data):
        """Setter for the custom connection to use in order to send us the data"""
        self._connection_data = data

def UCB_parse(data_path: Path, arsenal: list[DefaultParty]):
    """
    Parse the data in the given path and return it in an array,
    formatted in the same way as the given arsenal.

    For example, if the contents of the file are:

        Agent2: 0.47362867532420694
        Agent5: 0.3106295100984454
        Agent1: 0.6513009282512279
        Agent3: 0.6643482559837177
        Agent4: 0.2796158075868863

    and the "arsenal" argument is:
    
        [Agent1, Agent2, Agent3, Agent4, Agent5],
        
    then the following array will be returned:

        [0.6513009282512279, 0.47362867532420694, 0.6643482559837177, 0.2796158075868863, 0.3106295100984454].    
    """

    # keep the names of the agents in the arsenal as strings instead of classes
    _arsenal = [x.__name__ for x in arsenal]

    # create an array the size of the arsenal
    ret = [None for _ in arsenal]

    play_counts = None
    total_plays = None
    estimates = None

    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line:
                
                if "play_counts" in line:
                    play_counts = line.split(": ")[-1]
                    play_counts = play_counts[1:][:-2]
                    play_counts = [int(x) for x in play_counts.split(", ")]
                elif "total_plays" in line:
                    total_plays = int(line.split(": ")[-1])
                elif "estimates" in line:
                    estimates = line.split(": ")[-1]
                    estimates = estimates[1:][:-1]
                    estimates = [float(x) for x in estimates.split(", ") if x and x!=" "]
                else:
                    try:
                        agent = line.split(': ')[0]
                        score = float(line.split(': ')[1])
                    except Exception as e:
                        error = f"error parsing UCB data: {e}"
                        return False, error
                    
                    # set the element corresponding to the agent to its score
                    try:
                        ret[_arsenal.index(agent)] = score
                    except Exception as e:
                        return False, f"error parsing UCB data: {e}"

    # check that we dont return data that will cause problems (there is a check for "data is None" in the caller, but not a check for "None in data")
    if None in ret:
        error = "error parsing the data: did not get a score for the following agents: {[x for i, x in enumerate(_arsenal) if ret[i] is None]}"
        return False, error

    if play_counts is None or total_plays is None or estimates is None:
        error = "error parsing the data: did not get a score for the following agents: {[x for i, x in enumerate(_arsenal) if ret[i] is None]}"
        return False, error

    print(f"read UCB from file: {ret}")

    return True, ret, play_counts, total_plays, estimates

def UCB_write(data_path: Path, arsenal: list[DefaultParty], ucb: list[int], play_counts: list[int], total_plays: int, estimates):
    """
    Write the given data in the file at the given path.
    
    Overwrites the file if it already exists.
    Assumes arsenal and ucb have the same length.
    """

    with open(data_path, 'w') as f:
        for agent, estimate in zip(arsenal, ucb):
            f.write(f"{agent.__name__}: {estimate}\n")  # elements of arsenal are classes, so just .__name__ is enough

        f.write(f"play_counts: {[int(x) for x in play_counts]}\n")
        f.write(f"total_plays: {total_plays}\n")
        f.write(f"estimates: {[float(x) for x in estimates]}")
