from itertools import permutations
from math import factorial
from typing import Tuple


from utils.ask_proceed import ask_proceed
from .runners import run_session, process_tournament_results


def run_tournament(tournament_settings: dict, verbose:bool) -> Tuple[list, list]:
    # create agent permutations, ensures that every agent plays against every other agent on both sides of a profile set.
    agents = tournament_settings["agents"]
    opponents = tournament_settings["opponents"]
    profile_sets = tournament_settings["profile_sets"]
    deadline_time_ms = tournament_settings["deadline_time_ms"]

    num_sessions = (len(agents)) * (len(opponents)) * len(profile_sets) * 2
    if num_sessions > 100:
        message = (
            f"WARNING: this would run {num_sessions} negotiation sessions. Proceed?"
        )
        if not ask_proceed(message):
            print("Exiting script")
            exit()

    tournament_results = []
    tournament_steps = []
    util_results = []
    print(f"profile sets are {profile_sets}\n\n\n")
    session_no = 1
    for agent in agents:
        for raw_profiles in profile_sets:
            for profiles in permutations(raw_profiles, 2):
                # quick an dirty check
                
                profiles = list(profiles)
                print(f"profiles: {profiles}")
                
                assert isinstance(profiles, list) and len(profiles) == 2
                for opponent in opponents:
                    # create session settings dict
                    settings = {
                        "agents": [agent, opponent],
                        "profiles": profiles,
                        "deadline_time_ms": deadline_time_ms,
                    }

                    # run a single negotiation session
                    print(f"running session {session_no}/{num_sessions}...")
                    _, session_results_summary, util_result = run_session(settings, verbose, agent["class"].split('.')[-1])

                    # assemble results
                    tournament_steps.append(settings)
                    tournament_results.append(session_results_summary)
                    util_results.append(util_result)

                    session_no += 1

    tournament_results_summary = process_tournament_results(tournament_results)

    return tournament_steps, tournament_results, tournament_results_summary, util_results
