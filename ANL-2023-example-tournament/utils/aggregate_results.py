from itertools import groupby

def aggregate_results(util_dict):
    # Define a function to extract a hashable representation of the features
    def features_key(entry):
        return tuple(sorted(entry["features"].items()))

    # Sort the list based on the features
    util_dict.sort(key=features_key)

    # Use groupby to group elements with the same features
    grouped_dict = {key: list(group) for key, group in groupby(util_dict, key=features_key)}

    return grouped_dict
