from typing import cast, Dict
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from geniusweb.profile.utilityspace.DiscreteValueSetUtilities import DiscreteValueSetUtilities
from decimal import Decimal
import numpy as np

def extract_features(profile:LinearAdditiveUtilitySpace):
        """Extract the features from the current domain"""

        # shortcuts for domain stuff
        domain = profile.getDomain()
        all_bids = AllBidsList(domain)
        weights: Dict[str, Decimal] = profile.getWeights()

        # initialize and fill the dict that will be returned
        feat_dict = {}

        # 1) number of issues in domain
        feat_dict.update({"num_of_issues": len(weights)})

        # 2) average number of values per issue (sponsored by Misko)
        feat_dict.update({
            "avg_vals_per_issue": np.average([len(x.getUtilities()) for x in [cast(DiscreteValueSetUtilities, y) for y in profile.getUtilities().values()]])
        })

        # 3) number of possible bids
        feat_dict.update({"num_of_bids": all_bids.size()})

        # 4) standard deviation of weights
        feat_dict.update({"weight_std_dev": float(np.std(list(weights.values())))})

        # 5) average bid utility
        bid_utils = []
        for i in range(all_bids.size()): 
            bid = all_bids.get(i)
            bid_utils.append(np.float64(profile.getUtility(bid)))

        feat_dict.update({"avg_bid_util": np.average(bid_utils)})

        # "6) standard deviation of bid utility
        feat_dict.update({"bid_util_std_dev": np.std(bid_utils)})

        return feat_dict

