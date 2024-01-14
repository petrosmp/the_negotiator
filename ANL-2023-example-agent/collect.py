import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.runners import run_session

RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement
settings = {
    "agents": [
        {
            "class": "agents.the_negotiator.the_negotiator.TheNegotiator",
            "parameters": {"storage_dir": "agent_storage/TheNegotiator"},
        },
        {
            "class": "agents.template_agent.template_agent.TemplateAgent",
            "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
        },
    ],
    "profiles": ["domains/domain00/profileA.json", "domains/domain00/profileB.json"],
    "deadline_time_ms": 10000,
}

# run a session and obtain results in dictionaries
session_results_trace, session_results_summary, util_result = run_session(settings, verbose=False, care_about="TheNegotiator")

# now use the util result to write to the file
filename = f"dataset_{util_result['agent']}_{time.strftime('%Y%m%d-%H_%M_%S')}.py"
with open(filename, 'w') as f:
    f.write(
        f"""\
timestamp = "{time.strftime('%Y-%m-%d %H:%M:%S')}",
running_time = 25,   # minutes
results = [
    {{
        "features": {util_result['features']},
        "results": {{
            "{util_result['agent']}": {util_result['utility']},
        }}
    }},
]
        """
    )

