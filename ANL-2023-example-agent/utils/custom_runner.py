"""
The runner used to run the project but with a verbosity switch to make it shut up.
"""


import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import List, Optional

from pyson.ObjectMapper import ObjectMapper
from tudelft.utilities.listener.Listener import Listener
from tudelft_utilities_logging.Reporter import Reporter

from geniusweb.events.ProtocolEvent import ProtocolEvent
from geniusweb.protocol.CurrentNegoState import CurrentNegoState
from geniusweb.protocol.NegoProtocol import NegoProtocol
from geniusweb.protocol.NegoSettings import NegoSettings
from geniusweb.protocol.NegoState import NegoState
from geniusweb.simplerunner.ClassPathConnectionFactory import ClassPathConnectionFactory

from geniusweb.protocol.session.SessionResult import SessionResult
from pyson.ObjectMapper import ObjectMapper
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace

class Runner:
	'''
	A simple tool to run a negotiation stand-alone, without starting the servers.
	All referred files and classes need to be stored locally (or be in the
	dependency list if you use maven).
	<p>
	<em>IMPORTANT</em> SimpleRunner has a number of restrictions, compared to a
	run using a runserver and partyserver
	<ul>
	<li>With stand-alone runner, your parties are run together in a single
	classloader. The main implication is that there may arise version conflicts
	between parties.
	<li>Stand-alone runner does NOT enforce the time deadline. Parties may
	continue running indefinitely and thus bog down the JVM and stalling
	tournaments.
	</ul>
	'''

	_properlyStopped:bool = False
	_LOOPTIME = 200  # ms
	_FINALWAITTIME = 5000  # ms

	def __init__(self, settings:NegoSettings ,
			connectionfactory:ClassPathConnectionFactory , logger:Reporter ,
			maxruntime:int, verbose:bool=True, care_about:str=None):
		'''
		@param settings          the {@link NegoSettings}
		@param connectionfactory the {@link ProtocolToPartyConnFactory}
		@param logger            the {@link Reporter} to log problems
		@param maxruntime        limit in millisecs. Ignored if 0
		'''
		if settings == None  or connectionfactory == None:
			raise ValueError("Arguments must be not null");
		self._settings = settings;
		self._log = logger;
		self._protocol = settings.getProtocol(self._log);
		self._connectionfactory = connectionfactory;
		self._maxruntime = maxruntime;
		self._jackson = ObjectMapper()
		self._verbose = verbose
		self._care_about = care_about
		self._result = None

	def isProperlyStopped(self) -> bool:
		'''
		@return true if the runner has finished
		'''
		return self._properlyStopped

	def run(self):
		this = self

		class protocolListener(Listener[ProtocolEvent]):

			def notifyChange(self, evt: ProtocolEvent):
				this._handle(evt)

		self._protocol.addListener(protocolListener())
		self._protocol.start(self._connectionfactory)
		remainingtime = self._maxruntime;
		while not self._properlyStopped and  (self._maxruntime == 0 or remainingtime > 0):
			time.sleep(self._LOOPTIME / 1000.)
			remainingtime -= self._LOOPTIME
		self._log.log(logging.INFO, "Waiting for connection closure")

		remainingtime = self._FINALWAITTIME;
		while remainingtime > 0 and\
			len(self._connectionfactory.getOpenConnections()) != 0:
				time.sleep(self._LOOPTIME / 1000.)
				remainingtime -= self._LOOPTIME
			
		openconn = self._connectionfactory.getOpenConnections()
		if len(openconn) != 0:
			self._log.log(logging.WARNING, "Connections " + str(openconn)\
					+" did not close properly at end of run")
		self._log.log(logging.INFO, "end run")
		return self._result

	def _handle(self, evt:ProtocolEvent):
		if isinstance(evt , CurrentNegoState) and \
			evt.getState().isFinal(1000 * time.time()):
				self._stop()

	def _stop(self):
		self._logFinal(logging.INFO, self._protocol.getState())

		# get the result of the negotiation session
		result: SessionResult = next(iter(self._protocol.getState().getResults()))

		# get the agents that participated in the negotiation and the profile each participated as
		participants = []
		care_about = None
		for agent, profile in result.getParticipants().items():
			participants.append({
				"agent": agent.getName()[:-2].split('_')[-1],		# trim the _x at the end (that indicated the profile) and keep the part after the last '_'
				"profile": profile.getProfile().getURI().getPath()
			})

			# keep a reference to the agent that we care about (we dont really need the participants list, its here for debugging pruposes, it will be gone next commit)
			if participants[-1]["agent"] == self._care_about:
				care_about = participants[-1]

		# get the agreement at which the agents arived
		agreement = None
		utility = 0
		try:
			agreement = next(iter(result.getAgreements().getMap().values()))
		except StopIteration:
			pass
			# no agreement reached, 0 util for all

		# if there was an agreement find out the utility gained by the agent we care about
		if agreement:
		
			# check that we found the agent we care about
			if not care_about:
				print("error! could not find the agent we care about in the participants")
			else:
				profile_file = care_about["profile"]
				with open(profile_file) as f:
					profile_data = json.load(f)
				profile: LinearAdditiveUtilitySpace = ObjectMapper().parse(profile_data, LinearAdditiveUtilitySpace)
				utility = profile.getUtility(agreement)
		
		# set the result class variable so we can return it
		self._result = {
			"agent": self._care_about,
			"utility": float(utility)
		}

		self._properlyStopped = True

	def _logFinal(self, level:int , state: NegoState):
		'''
		Separate so that we can intercept this when mocking, as this will crash
		on mocks because {@link #jackson} can not handle mocks.
		
		@param level the log {@link Level}, eg logging.WARNING
		@param state the {@link NegoState} to log
		'''
		try:
			if self._verbose:	# this is the change
				self._log.log(level, "protocol ended normally: "
						+json.dumps(self._jackson.toJson(self._protocol.getState())))
		except Exception as e:  # catch json issues
			traceback.print_exc()

	def getProtocol(self) -> NegoProtocol:
		'''
		@return protocol that runs/ran the session.
		'''
		return self._protocol
