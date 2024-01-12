"""
Package containing all the agents that the_negotiator can use.

If you want to add a new agent, bring it into this directory, import it
here and add it to the __all__ list.

Agents are assumed to:
    - extend geniusweb.party.DefaultParty.DefaultParty (or any of its
      subclasses)
    - take no constructor arguments

Those assumptions seem to make sense (they need to have some kind of
similar structure since they all have to be runnable in the same way),
but is not *necessarily* true for any agent you might want to use. If
you want to use an agent that does not satisf the criteria above, make
a new agent that satisfies them and use it as a proxy.
"""

__all__ = [
    "BoulwareAgent", "ConcederAgent", "HardlinerAgent", "LinearAgent",
    "TUCStudentsTimeDependentAgent",
]

from .boulware_agent import BoulwareAgent
from .conceder_agent import ConcederAgent
from .hardliner_agent import HardlinerAgent
from .linear_agent import LinearAgent
from .tuc_students_time_dependent_agent import TUCStudentsTimeDependentAgent
