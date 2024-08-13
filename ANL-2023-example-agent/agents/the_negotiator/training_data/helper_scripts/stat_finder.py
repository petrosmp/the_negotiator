from dataset_tournament_kalo_20240115__02_56_18 import results as results_0_7
from dataset_tournament_kalo_20240115__08_57_37 import results as results_7_15
from dataset_tournament_kalo_20240115__11_19_44 import results as results_15_18
from dataset_tournament_kalo_20240115__12_09_28 import results as results_19_21
from dataset_tournament_kalo_20240115__12_42_03 import results as results_30_37
from dataset_tournament_kalo_20240115__12_48_45 import results as results_22_25
from dataset_tournament_kalo_20240115__13_21_24 import results as results_37_45
from dataset_tournament_kalo_20240115__13_40_43 import results as results_26_28
from dataset_tournament_kalo_20240115__13_53_28 import results as results_29_30

all_results =  results_0_7 + results_7_15 + results_15_18 + results_19_21 + results_22_25 + results_26_28 + results_29_30 + results_30_37 + results_37_45


min_avg_bid_util = float("inf")
min_avg_vals_per_issue = float("inf")
min_bid_util_std_dev = float("inf")
min_num_of_bids = float("inf")
min_num_of_issues = float("inf")
min_weight_std_dev = float("inf")

max_avg_bid_util = float("-inf")
max_avg_vals_per_issue = float("-inf")
max_bid_util_std_dev = float("-inf")
max_num_of_bids = float("-inf")
max_num_of_issues = float("-inf")
max_weight_std_dev = float("-inf")


for result in all_results:
    
    features = result["features"]
    
    avg_bid_util        = features['avg_bid_util']
    avg_vals_per_issue  = features['avg_vals_per_issue']
    bid_util_std_dev    = features['bid_util_std_dev']
    num_of_bids         = features['num_of_bids']
    num_of_issues       = features['num_of_issues']
    weight_std_dev      = features['weight_std_dev']

    # mins
    if avg_bid_util < min_avg_bid_util:
        min_avg_bid_util = avg_bid_util
    if avg_vals_per_issue < min_avg_vals_per_issue:
        min_avg_vals_per_issue = avg_vals_per_issue
    if bid_util_std_dev < min_bid_util_std_dev:
        min_bid_util_std_dev = bid_util_std_dev
    if num_of_bids < min_num_of_bids:
        min_num_of_bids = num_of_bids
    if num_of_issues < min_num_of_issues:
        min_num_of_issues = num_of_issues
    if weight_std_dev < min_weight_std_dev:
        min_weight_std_dev = weight_std_dev

    # maxs
    if avg_bid_util > max_avg_bid_util:
        max_avg_bid_util = avg_bid_util
    if avg_vals_per_issue > max_avg_vals_per_issue:
        max_avg_vals_per_issue = avg_vals_per_issue
    if bid_util_std_dev > max_bid_util_std_dev:
        max_bid_util_std_dev = bid_util_std_dev
    if num_of_bids > max_num_of_bids:
        max_num_of_bids = num_of_bids
    if num_of_issues > max_num_of_issues:
        max_num_of_issues = num_of_issues
    if weight_std_dev > max_weight_std_dev:
        max_weight_std_dev = weight_std_dev


print(f"avg_bid_util ranges from        {round(min_avg_bid_util, 5)}  to {round(max_avg_bid_util, 5)}")
print(f"avg_vals_per_issue ranges from  {round(min_avg_vals_per_issue, 5)} to {round(max_avg_vals_per_issue, 5)}")
print(f"bid_util_std_dev ranges from    {round(min_bid_util_std_dev, 5)} to {round(max_bid_util_std_dev, 5)}")
print(f"num_of_bids ranges from         {round(min_num_of_bids, 5)} to {round(max_num_of_bids, 5)}")
print(f"num_of_issues ranges from       {round(min_num_of_issues, 5)} to {round(max_num_of_issues, 5)}")
print(f"weight_std_dev ranges from      {round(min_weight_std_dev, 5)} to {round(max_weight_std_dev, 5)}")
