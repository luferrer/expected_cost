import numpy as np
from scipy.special import logsumexp 
from expected_cost import ec, utils
from data import get_scaled_llks_for_multi_classif_task
import sys

adjusted_cost = True

# Template for analysis of multi-class classification scores. To change the input
# data, you can add your own loading process in the get_scaled_llks_for_multi_classif_task
# in data.py. 
C = 10
p0 = 0.8
data_priors =  np.ones(C)/C
targets, raw_llks, cal_llks = get_scaled_llks_for_multi_classif_task('gaussian_sim', data_priors, 100000)

# For this exercise, we want to compare metrics computed with scaled lks as scores vs
# those computed with posteriors. Hence, we generate posteriors using the priors
# found in the data. These are well-calibrated posteriors for this data, but if the
# priors change, these posteriors are no longer well-calibrated.
print("Computing log-posteriors assuming uniform priors")
raw_logpost = utils.llks_to_logpost(raw_llks, data_priors)
cal_logpost = utils.llks_to_logpost(cal_llks, data_priors)

num_targets = np.max(targets)+1

# Compute the cost for maximum a posteriori decisions
map_decisions = np.argmax(raw_logpost, axis=-1)
cost_01 = ec.cost_matrix.zero_one_costs(num_targets)
ac_01 = ec.average_cost(targets, map_decisions, cost_01, adjusted=adjusted_cost)

# Now compute the cost for Bayes decisions based on the posteriors
# before and after calibration.
ac_01_bayes,           _ = ec.average_cost_for_bayes_decisions(targets, raw_logpost, cost_01, adjusted=adjusted_cost)
ac_01_bayes_after_cal, _ = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost_01, adjusted=adjusted_cost)

print("*** Average cost for 0-1 cost matrix")
print("    Using MAP decisions:                        %.4f"%ac_01)
print("    Using Bayes decisions:                      %.4f"%ac_01_bayes)
print("    Using Bayes decisions on calibrated scores: %.4f"%ac_01_bayes_after_cal)
print("""Note that:
* The first two costs are the same because Bayes decisions for 0-1 cost coincide with MAP decisions.
* The Bayes decisions are better after calibration indicating that the raw scores were not well-calibrated.""")

# Try a cost with an abstention decision and evaluate the MAP decisions.
ab_cost = 0.4
cost_ab = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=ab_cost)
ac_ab = ec.average_cost(targets, map_decisions, cost_ab, adjusted=adjusted_cost)

# Now evaluate the cost for Bayes decisions for that cost function
ac_ab_bayes, bayes_decisions_ab = ec.average_cost_for_bayes_decisions(targets, raw_logpost, cost_ab, adjusted=adjusted_cost)
ac_ab_bayes_after_cal,       _  = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost_ab, adjusted=adjusted_cost)

print("*** Average cost for cost matrix with abstention cost of %.1f"%ab_cost)
print("    Using MAP decisions:                        %.4f"%ac_ab)
print("    Using Bayes decisions:                      %.4f"%ac_ab_bayes)
print("    Using Bayes decisions on calibrated scores: %.4f"%ac_ab_bayes_after_cal)
print("""Note that:
* The first two costs are not the same because optimal (Bayes) decisions include %d abstentions on the hardest samples. 
* Again, Bayes decisions are better after calibration."""%len(np.where(bayes_decisions_ab!= map_decisions)[0]))

# Now let's assume that the actual data the system will be used on has 
# different priors from those used to compute the posteriors and those in
# the available test data. Hence, knowing this, we use those priors (the 
# ones we expect to see in practice) for computing the cost.
p0 = 0.8
nu_priors = np.array([(1-p0)/(C-1)]*(C-1) + [p0] )

# Let's find the average cost for the 0-1 cost matrix when making Bayes
# decisions with the posteriors given by the system. The Bayes decisions will
# not change since the posteriors do not change, but the cost will because
# the priors changed (they are no longer those in the test data). 
ac_01_nup_bayes, _ = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost_01, priors=nu_priors, adjusted=adjusted_cost, silent=True)

# Now, instead of using the posteriors to make decisions, use the llks.
# In this case, the method computes the posteriors using the provided priors.
# Hence, the cost in this case is smaller because there is no mismatch in the priors.
ac_01_nup_bayes_with_matched_priors, _ = ec.average_cost_for_bayes_decisions(targets, cal_llks, cost_01, priors=nu_priors, adjusted=adjusted_cost, score_type='log_likelihoods')

print("*** Average cost for 0-1 cost matrix and new priors %s"%" ".join(['{:.2f}'.format(i) for i in nu_priors]))
print("    Using Bayes decisions on calibrated scores, computing the posteriors with data priors:    %.4f"%ac_01_nup_bayes)
print("    Using Bayes decisions on calibrated scores, computing the posteriors with cost priors:    %.4f"%ac_01_nup_bayes_with_matched_priors)
print("""Note that:
* The second average cost is much lower because the posteriors computed with the original priors are quite bad 
  for making decisions for the priors in the cost function.""")



