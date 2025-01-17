import json
import os
from pathlib import Path
import time

from utils.custom_runners import run_tournament
from utils.aggregate_results import aggregate_results
import numpy as np
from datetime import datetime as dt

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
            "class": "agents.ANL2022.agent007.agent007.Agent007",
            "parameters": {"storage_dir": "agent_storage/Agent007"},
        },
        {
            "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
            "parameters": {"storage_dir": "agent_storage/DreamTeam109Agent"},
        },
        {
            "class": "agents.template_agent.template_agent.TemplateAgent",
            "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
        },
        {
            "class": "agents.ANL2022.gea_agent.gea_agent.GEAAgent",
            "parameters": {"storage_dir": "agent_storage/GEAAgent"},
        },
        {
            "class": "agents.CSE3210.agent33.agent33.Agent33",
            "parameters": {"storage_dir": "agent_storage/Agent33"},
        },
    ],
    "opponents": [
        {
            "class": "agents.ANL2022.agent007.agent007.Agent007",
            "parameters": {"storage_dir": "agent_storage/opponents/Agent007"},
        },
        {
            "class": "agents.ANL2022.agent4410.agent_4410.Agent4410",
            "parameters": {"storage_dir": "agent_storage/opponents/Agent4410"},
        },
        {
            "class": "agents.ANL2022.agentfish.agentfish.AgentFish",
            "parameters": {"storage_dir": "agent_storage/opponents/AgentFish"},
        },
        {
            "class": "agents.ANL2022.charging_boul.charging_boul.ChargingBoul",
            "parameters": {"storage_dir": "agent_storage/opponents/ChargingBoul"},
        },
        {
            "class": "agents.ANL2022.learning_agent.learning_agent.LearningAgent",
            "parameters": {"storage_dir": "agent_storage/opponents/LearningAgent"},
        },
        {
            "class": "agents.ANL2022.compromising_agent.compromising_agent.CompromisingAgent",
            "parameters": {"storage_dir": "agent_storage/opponents/CompromisingAgent"},
        },
        {
            "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
            "parameters": {"storage_dir": "agent_storage/opponents/DreamTeam109Agent"},
        },
        {
            "class": "agents.ANL2022.gea_agent.gea_agent.GEAAgent",
            "parameters": {"storage_dir": "agent_storage/opponents/GEAAgent"},
        },
        {
            "class": "agents.ANL2022.LuckyAgent2022.LuckyAgent2022.LuckyAgent2022",
            "parameters": {"storage_dir": "agent_storage/opponents/LuckyAgent2022"},
        },
        {
            "class": "agents.ANL2022.Pinar_Agent.Pinar_Agent.Pinar_Agent",
            "parameters": {"storage_dir": "agent_storage/opponents/Pinar_Agent"},
        }
    ],
    "profile_sets": [
        ["domains/domain46/profileA.json", "domains/domain46/profileB.json"],
        ["domains/domain47/profileA.json", "domains/domain47/profileB.json"],
        ["domains/domain48/profileA.json", "domains/domain48/profileB.json"],
        # ["domains/domain34/profileA.json", "domains/domain34/profileB.json"],
        # ["domains/domain35/profileA.json", "domains/domain35/profileB.json"],
        # ["domains/domain36/profileA.json", "domains/domain36/profileB.json"],
        # ["domains/domain37/profileA.json", "domains/domain37/profileB.json"],
        # ["domains/domain38/profileA.json", "domains/domain37/profileB.json"],
        # ["domains/domain40/profileA.json", "domains/domain37/profileB.json"],
        # ["domains/domain41/profileA.json", "domains/domain37/profileB.json"],
        # ["domains/domain43/profileA.json", "domains/domain37/profileB.json"],
        # ["domains/domain44/profileA.json", "domains/domain37/profileB.json"],
        # ["domains/domain45/profileA.json", "domains/domain37/profileB.json"],
    ],
    "deadline_time_ms": 10000,
}

start_time = dt.now()

# run a session and obtain results in dictionaries
tournament_steps, tournament_results, tournament_results_summary, util_results = run_tournament(tournament_settings, False)

end_time = dt.now()
filename = f"dataset_tournament_kalo_{time.strftime('%Y%m%d-%H_%M_%S')}.py"
print(f"the creation of {filename} took {str(end_time-start_time)}")

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
