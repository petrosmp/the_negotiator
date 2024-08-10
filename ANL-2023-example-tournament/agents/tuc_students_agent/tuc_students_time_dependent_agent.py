import logging
from random import randint, random
import traceback
from typing import cast, Dict, List, Set, Collection

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
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
from geniusweb.progress.Progress import Progress
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from time import sleep, time as clock
from decimal import Decimal
import sys
from agents.time_dependent_agent.extended_util_space import ExtendedUtilSpace
from tudelft_utilities_logging.Reporter import Reporter


class TUCStudentsTimeDependentAgent(DefaultParty):
    """
    General time dependent party.
    <p>
    Supports parameters as follows
    <table>
    <caption>parameters</caption>
    <tr>
    <td>e</td>
    <td>e determines how fast the party makes concessions with time. Typically
    around 1. 0 means no concession, 1 linear concession, &gt;1 faster than
    linear concession.</td>
    </tr>

    <tr>
    <td>minPower</td>
    <td>This value is used as minPower for placed {@link Vote}s. Default value is
    1.</td>
    </tr>

    <tr>
    <td>maxPower</td>
    <td>This value is used as maxPower for placed {@link Vote}s. Default value is
    infinity.</td>
    </tr>

    <tr>
    <td>delay</td>
    <td>The average time in seconds to wait before responding to a YourTurn. The
    actual waiting time will be random in [0.5*time, 1.5*time]. This can be used
    to simulate human users that take thinking time.</td>
    </tr>

    </table>
    <p>
    TimeDependentParty requires a {@link UtilitySpace}
    """

    # value initializations
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
        try:
            # parse Inform object, based on its specified type
            if isinstance(info, Settings):
                # Enter setting information. This is called once at the beginning of a session.
                # You can add your own code based on your own implementation.
                self._settings = info
                self._me = self._settings.getID()
                self._progress = self._settings.getProgress()
                newe = self._settings.getParameters().get("e")
                if newe != None:
                    if isinstance(newe, float):
                        self._e = newe
                    else:
                        self.getReporter().log(
                            logging.WARNING,
                            "parameter e should be Double but found " + str(newe),
                        )
                self._profileint = ProfileConnectionFactory.create(self._settings.getProfile().getURI(), self.getReporter()) #TODO
            elif isinstance(info, ActionDone):
                otheract: Action = info.getAction()
                if isinstance(otheract, Offer):
                    self._lastReceivedBid = otheract.getBid()
            elif isinstance(info, YourTurn):
                self._delayResponse()
                self._myTurn()
            elif isinstance(info, Finished):
                self.getReporter().log(logging.INFO, "Final outcome:" + str(info))
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
