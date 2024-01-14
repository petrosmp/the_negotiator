import json
import os
from pathlib import Path
import time

from utils.custom_runners import run_tournament
from utils.aggregate_results import aggregate_results
import numpy as np

RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement.
tournament_settings = {
    "agents": [
        {
            "class": "agents.template_agent.template_agent.TemplateAgent",
            "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
        },
        {
            "class": "agents.the_negotiator.the_negotiator.TheNegotiator",
            "parameters": {"storage_dir": "agent_storage/TheNegotiator"},
        },
    ],
    "opponents": [

        {
            "class": "agents.boulware_agent.boulware_agent.BoulwareAgent",
        },
        {
            "class": "agents.conceder_agent.conceder_agent.ConcederAgent",
        },
    ],
    "profile_sets": [
        ["domains/domain00/profileA.json", "domains/domain00/profileB.json"],
        ["domains/domain01/profileA.json", "domains/domain01/profileB.json"],
    ],
    "deadline_time_ms": 10000,
}

# run a session and obtain results in dictionaries
tournament_steps, tournament_results, tournament_results_summary, util_results = run_tournament(tournament_settings, False)


filename = f"dataset_tournament_{time.strftime('%Y%m%d-%H_%M_%S')}.py"
with open(filename, 'w') as f:

    f.write(f'results = [\n')

    for key, group in aggregate_results(util_results).items():
        print(f"Features: {dict(key)}")

        f.write(f'\t{{\n\t\t"features": {dict(key)},\n\t\t"results":')

        res = {}
        
        for entry in group:
            
            agent = entry['agent']
            utility = entry['utility']

            if agent in res.keys():
                res[agent].append(utility)
            else:
                res.update({agent: [utility, ]})

        res = {k: np.average(v) for k, v in res.items()}

        f.write(f"{res},\n\t}},\n")


        print()
    f.write(f"]\n")


# save the tournament settings for reference
with open(RESULTS_DIR.joinpath("tournament_steps.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(tournament_steps, indent=2))
# save the tournament results
with open(RESULTS_DIR.joinpath("tournament_results.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(tournament_results, indent=2))
# save the tournament results summary
tournament_results_summary.to_csv(RESULTS_DIR.joinpath("tournament_results_summary.csv"))
