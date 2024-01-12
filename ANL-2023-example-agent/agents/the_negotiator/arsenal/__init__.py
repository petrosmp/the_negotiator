"""
Package containing all the agents that the_negotiator can use.

If you want to add a new agent, bring it into this directory, import it
in __arsenal__ and add it to the list.

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

__all__ = [arsenal]

from .__arsenal__ import arsenal
