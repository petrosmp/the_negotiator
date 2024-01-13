import logging
from random import randint, random
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
from geniusweb.progress.Progress import Progress
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from time import sleep, time as clock
from decimal import Decimal
import sys
from agents.time_dependent_agent.extended_util_space import ExtendedUtilSpace
from tudelft_utilities_logging.Reporter import Reporter
from geniusweb.references.Parameters import Parameters


from .arsenal import arsenal
from .arsenal.tuc_students_time_dependent_agent import TUCStudentsTimeDependentAgent
from .arsenal.linear_agent import LinearAgent
from .arsenal.boulware_agent import BoulwareAgent
from .arsenal.conceder_agent import ConcederAgent
from .arsenal.hardliner_agent import HardlinerAgent

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

                print(f"\n\n\n{self.__class__.__name__} got settings!")
                self._settings = info
                
                # unpack the settings and store them as class variables for future reference
                self._me = self._settings.getID()
                self._progress = self._settings.getProgress()               # progress towards the deadline has to be tracked manually through the use of the Progress object
                self._parameters = self._settings.getParameters()            # what are parameters? https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb/wiki/WikiStart#PartyParameters
                self._storage_dir = self._parameters.get("storage_dir")

                # in order to get the profile (preferences) and the domain info we need a connection
                profile_connection = ProfileConnectionFactory.create(
                    info.getProfile().getURI(), self.getReporter()
                )
                self._profile: LinearAdditiveUtilitySpace = profile_connection.getProfile()
                self._domain = self._profile.getDomain()
                self._all_bids = AllBidsList(self._domain)                  # compose a list of all possible bids
                profile_connection.close()            

                # we now have enough data to calculate anything we want. its time to pick a strat.
                self._strat = self._pick_strategy()

                # set ourselves as a proxy for the strategy agent
                self._strat.set_proxy(self)
                self._strat_str = f"{self._strat.__class__.__module__}.{self._strat.__class__.__name__}"
                
                # create a new param set (so that the strategy agent has its own directory)
                storage_dir = self._parameters.get("storage_dir").replace(self.__class__.__name__, self._strat.__class__.__name__)
                parameters = Parameters({
                    "storage_dir": storage_dir
                })

                # we need to create other settings to pass to the real agent
                newinfo = Settings(
                    self._me,        # id
                    info.getProfile(),  # profile
                    info.getProtocol(), # protocol
                    info.getProgress(), # progress
                    parameters          # parameters
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
                    utility = self._profile.getUtility(deal)
                    self.getReporter().log(logging.INFO, f"Final outcome: bid={deal} giving us a utility of: {utility} (special thanks to {self._strat.__class__.__name__})")
                except StopIteration:
                    self.getReporter().log(logging.INFO, "no agreement reached!")
                    print("no agreement reached!\n\n")
                

                self._strat.notifyChange(info)
                self.terminate()
                # stop this party and free resources.
        except Exception as ex:
            self.getReporter().log(logging.CRITICAL, "Failed to handle info", ex)
        self._updateRound(info)

    def getE(self) -> float:
        """
        @return the E value that controls the party's behaviour. Depending on the
                value of e, extreme sets show clearly different patterns of
               behaviour [1]:

               1. Boulware: For this strategy e &lt; 1 and the initial offer is
                maintained till time is almost exhausted, when the agent concedes
                up to its reservation value.

                2. Conceder: For this strategy e &gt; 1 and the agent goes to its
                reservation value very quickly.

                3. When e = 1, the price is increased linearly.

                4. When e = 0, the agent plays hardball.
        """
        return self._e

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

    def _myTurn(self):
        self._updateUtilSpace()
        bid = self._makeBid()

        myAction: Action
        if bid == None or (
            self._lastReceivedBid != None
            and self._utilspace.getUtility(self._lastReceivedBid)
            >= self._utilspace.getUtility(bid)
        ):
            # if bid==null we failed to suggest next bid.
            myAction = Accept(self._me, self._lastReceivedBid)
        else:
            myAction = Offer(self._me, bid)
        self.getConnection().send(myAction)

    def _updateUtilSpace(self) -> LinearAdditive:  # throws IOException
        newutilspace = self._profileint.getProfile()
        if not newutilspace == self._utilspace:
            self._utilspace = cast(LinearAdditive, newutilspace)
            self._extendedspace = ExtendedUtilSpace(self._utilspace)
        return self._utilspace

    def _makeBid(self) -> Bid:
        """
        @return next possible bid with current target utility, or null if no such
                bid.
        """
        time = self._progress.get(round(clock() * 1000))

        utilityGoal = self._getUtilityGoal(
            time,
            self.getE(),
            self._extendedspace.getMin(),
            self._extendedspace.getMax(),
        )
        options: ImmutableList[Bid] = self._extendedspace.getBids(utilityGoal)
        if options.size() == 0:
            # if we can't find good bid, get max util bid....
            options = self._extendedspace.getBids(self._extendedspace.getMax())
        # pick a random one.
        return options.get(randint(0, options.size() - 1))

    def _getUtilityGoal(
        self, t: float, e: float, minUtil: Decimal, maxUtil: Decimal
    ) -> Decimal:
        """
        @param t       the time in [0,1] where 0 means start of nego and 1 the
                       end of nego (absolute time/round limit)
        @param e       the e value that determinses how fast the party makes
                       concessions with time. Typically around 1. 0 means no
                       concession, 1 linear concession, &gt;1 faster than linear
                       concession.
        @param minUtil the minimum utility possible in our profile
        @param maxUtil the maximum utility possible in our profile
        @return the utility goal for this time and e value
        """

                
        ft1 = round(Decimal(1 - pow(t, 1 / e)), 6)  # defaults ROUND_HALF_UP

        # ft1 is a normalized value [0,1], that decreases with time t

        # we translate ft1 according to the range of utilities in the current domain, using minUtil and maxUtil values
        # such that: ft1=0 -> util = minUtil, ft1=1 -> util = maxUtil, and using linear interpolation for ft1 in between (0,1)  
        util = minUtil + (maxUtil - minUtil) * ft1

        # bound the value according to [minUtil, maxUtil] in case ft1 is out of the desired range [0,1]
        if util < minUtil:
            util = minUtil
        elif util > maxUtil:
            util = maxUtil

        return util

    def _delayResponse(self):  # throws InterruptedException
        """
        Do random delay of provided delay in seconds, randomized by factor in
        [0.5, 1.5]. Does not delay if set to 0.

        @throws InterruptedException
        """
        delay = self._settings.getParameters().getDouble("delay", 0, 0, 10000000)
        if delay > 0:
            sleep(delay * (0.5 + random()))

    def _pick_strategy(self):
        """Pick a strategy that fits the current domain and profile"""

        features = self._extract_features()

        instance = HardlinerAgent()

        return instance

    def _extract_features(self):
        """Extract the features that are useful in picking a strategy from the current domain"""
        pass

    def set_connection_data(self, data):
        """Setter for the custom connection to use in order to send us the data"""
        self._connection_data = data
