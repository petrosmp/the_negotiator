from agents.time_dependent_agent.time_dependent_agent import TimeDependentAgent
from tudelft_utilities_logging.Reporter import Reporter
from .connection_intercept_mixin import ConnectionInterceptMixin


class BoulwareAgent(ConnectionInterceptMixin, TimeDependentAgent):
    """
    A simple party that places random bids and accepts when it receives an offer
    with sufficient utility.
    """

    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)

    # Override
    def getDescription(self) -> str:
        return (
            "Boulware: reluctant to concede. Parameters minPower (default 1) "
            + "and maxPower (default infinity) are used when voting"
        )

    # Override
    def getE(self) -> float:
        return 0.2
