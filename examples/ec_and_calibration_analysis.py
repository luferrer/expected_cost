# Script used to generate some of the results in 
# "Analysis and Comparison of Classification Metrics", arXiv:2209.05355

import numpy as np
from scipy.special import logsumexp 
from expected_cost import ec, utils
from data import get_llks_for_multi_classif_task
import matplotlib.pyplot as plt

import sys
try:
    import torch    
    from expected_cost.calibration import affine_calibration_with_crossval
    has_psr = True
except:
    has_psr = False
    print("PSR code not available, skipping calibration experiments.")

adjusted_cost = True

outdir = "outputs/ec_and_calibration_analysis"
utils.mkdir_p(outdir)

#########################################################################################################
""" Generate a bunch of different scores for a C class problem (C can be changed to whatever you like):

* raw_llks: raw log scaled likelihoods with Gaussian class distributions

* cal_llks: calibrated log scaled likelihoods obtained by evaluating those Gaussian distribution on the raw_llks

* raw/cal_logpost_datap: log-posteriors obtained from the llks applying the true data priors

* raw/cal_logpost_mismp: log-posteriors obtained from the llks applying the mismatched data priors 
  to simulate a system that was trained with the wrong priors

* raw/cal_logpost_calto_llks: calibrated version of raw/cal_logpost where calibration is done 
  using uniform target priors which should produce an estimate of the log scaled lks. Note that
  it does not matter which of the two logpost (datap or mismp) are used here as input since the
  calibration compensates for any shift in the scores produced by the priors.

* raw/cal_logpost_calto_logpost_with_data_priors: calibrated version of raw/cal_logpost where 
  calibration is done using the data priors to directly produce logposteriors that are matched
  to the test data.
"""

C = 4
p0 = 0.9
data_priors = np.array([p0] + [(1-p0)/(C-1)]*(C-1))
mism_priors = np.array([(1-p0)/(C-1)]*(C-1) + [p0])
unif_priors = np.ones(C)/C

targets, raw_llks, cal_llks = get_llks_for_multi_classif_task('gaussian_sim', priors=data_priors, std=0.1, mean0=0)
num_targets = np.max(targets)+1

raw_logpost_datap = utils.llks_to_logpost(raw_llks, data_priors)
cal_logpost_datap = utils.llks_to_logpost(cal_llks, data_priors)

raw_logpost_mismp = utils.llks_to_logpost(raw_llks, mism_priors)
cal_logpost_mismp = utils.llks_to_logpost(cal_llks, mism_priors)

if has_psr:
  raw_logpost_calto_llks                     = affine_calibration_with_crossval(raw_logpost_mismp, targets, priors=unif_priors)
  raw_logpost_calto_logpost_with_data_priors = affine_calibration_with_crossval(raw_logpost_mismp, targets, priors=data_priors)
  cal_logpost_calto_llks                     = affine_calibration_with_crossval(cal_logpost_mismp, targets, priors=unif_priors)
  cal_logpost_calto_logpost_with_data_priors = affine_calibration_with_crossval(cal_logpost_mismp, targets, priors=data_priors)

# Make a dictionary of the available scores:
score_dict = {'raw_llks': raw_llks,
              'raw_logpost_datap': raw_logpost_datap,
              'raw_logpost_mismp': raw_logpost_mismp,
              'raw_logpost_calto_llks': raw_logpost_calto_llks,
              'raw_logpost_calto_logpost_with_data_priors': raw_logpost_calto_logpost_with_data_priors,
              'cal_llks': cal_llks,
              'cal_logpost_datap': cal_logpost_datap,
              'cal_logpost_mismp': cal_logpost_mismp,
              'cal_logpost_calto_llks': cal_logpost_calto_llks,
              'cal_logpost_calto_logpost_with_data_priors': cal_logpost_calto_logpost_with_data_priors}


# Plot the resulting score distributions 
for score_name, scores in score_dict.items():
  utils.plot_hists(targets, scores, "%s/dists_%s_C=%d.pdf"%(outdir,score_name,num_targets))

#########################################################################################################
# First, using the logposteriors computed with matched priors, compute a
# family of cost matrices varying the costs for the last row (in the log
# domain) and leaving the other one fixed at 1. For each of these matrices
# compute the cost for maximum-a-posterior decisions and the cost for Bayes
# decisions. 

print("***************************************************************************************************************")
print("Average cost for cost matrix with c_ii = 0, c_ij = 1 for i!=j and i!=C, and c_ij = alpha, for i!=j and i=C.")
print("Using raw and calibrated log posteriors computed with the data priors\n")

if num_targets == 2:
  print("Alpha        MAP         Bayes  Bayes_after_cal Optimal")
else:
  print("Alpha        MAP         Bayes  Bayes_after_cal")

map_decisions = np.argmax(raw_logpost_datap, axis=-1)

for logalpha in np.arange(-4, 4, 0.5):

    alpha = np.exp(logalpha)
    costm = 1-np.eye(num_targets)
    costm[-1,0:-1] = alpha
    cost = ec.cost_matrix(costm)  

    ec_map                = ec.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ec_bayes,           _ = ec.average_cost_for_bayes_decisions(targets, raw_logpost_datap, cost, adjusted=adjusted_cost)
    ec_bayes_after_cal, _ = ec.average_cost_for_bayes_decisions(targets, cal_logpost_datap, cost, adjusted=adjusted_cost)

    if num_targets == 2:
      ec_min                = ec.average_cost_for_optimal_decisions(targets, raw_logpost_datap, cost, adjusted=adjusted_cost)
      print("%6.3f      %6.3f      %6.3f      %6.3f      %6.3f"%(alpha, ec_map, ec_bayes, ec_bayes_after_cal, ec_min))
    else:
      print("%6.3f      %6.3f      %6.3f      %6.3f "%(alpha, ec_map, ec_bayes, ec_bayes_after_cal,))

print("""\nNote that:
* Columns 2 through 5 are the average cost when decisions are made with different algorithms.
* The name of the column indicates how the decisions were made: MAP decisions, Bayes decisions on raw scores,
  Bayes decisions on calibrated scores, optimal decisions (ie, selecting the best threshold for the test data).
* The difference between MAP and Bayes shows the suboptimality of MAP decisions when the costs are not equal.
* The difference between the Bayes and the Bayes_after_cal shows how well or badly calibrated the system scores are. 
  Larger differences indicate a more severe calibration problem at that specific operating point. 
* (Only for binary classification) The difference between Bayes_after_cal and Optimal indicates that either the 
  calibration did not work perfectly for that operating point or that the optimal cost is, in fact, too optimistic 
  since the threshold is selected on the evaluation data itself.\n\n""")

print("*********************************************************************************************************************************")
print("Average cost for cost matrix with c_ii=0, c_ij=1 for i!=j, and with a last column for an abstention decision with cost alpha.")
print("Using raw and calibrated log posteriors computed with the data priors\n")
print("Alpha      MAP      Bayes    Perc_abstentions_with_Bayes   Bayes_after_cal  Perc_abstentions_with_Bayes_after_cal")


for alpha in [0.01, 0.1, 0.2, 0.4, 0.6, 1.0]:
    cost = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=alpha)

    ec_map                                  = ec.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ec_bayes,           decisions,          = ec.average_cost_for_bayes_decisions(targets, raw_logpost_datap, cost, adjusted=adjusted_cost)
    ec_bayes_after_cal, decisions_after_cal = ec.average_cost_for_bayes_decisions(targets, cal_logpost_datap, cost, adjusted=adjusted_cost)
    perc_abs = np.sum(decisions==num_targets)/len(decisions)*100
    perc_abs_after_cal = np.sum(decisions_after_cal==num_targets)/len(decisions)*100
    print("%6.3f   %6.3f    %6.3f        %6.1f                 %6.3f         %6.1f"%(alpha, ec_map, ec_bayes, perc_abs, ec_bayes_after_cal, perc_abs_after_cal))
      

print("""\nNote that
* The lower the cost of abstention, the more samples get this label and the worse the MAP decisions are (since 
  they do not take advantage of the abstention option). 
* If the cost of abstention is too high, the system never chooses to abstain.
* The difference between the Bayes column and the Bayes_after_cal column show the extent of the misscalibration
  in the raw log-posteriors.
* Note that, even for the binary case, when the cost function has an abstention decision, optimal decisions cannot be made 
  by sweeping a threshold as for the square cost function above because there are three possible decisions.\n""")

#########################################################################################################
# Finally, analyze the issue of mismatched priors

if has_psr is False:
  print("*** Calibration analysis skipped since the psr package is not available")
  sys.exit(0)

# Now we can evaluate the cross-entropy and a couple of expected costs (using Bayes decisions)
# for each of these scores on the test data. In all cases we take the target priors to be the
# ones in the test data. 

cost_01 = ec.cost_matrix.zero_one_costs(num_targets)
cost_ab3 = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=0.3)
cost_ab1 = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=0.1)

costm = 1-np.eye(num_targets)
costm[-1,0:-1] = 2
cost01a = ec.cost_matrix(costm)  

cost_dict = {'cost_01': cost_01, 'cost_01_abs=0.1': cost_ab1, 'cost_01_abs=0.3': cost_ab3, 'cost_01_lastrow=2': cost01a}

print("*********************************************************************************************************************************")
print("""Four different ECs for different logposteriors computed using raw and calibrated likelihoods:

* logpost_datap: log-posteriors obtained from the raw/cal llks applying the true data priors

* logpost_mismp: log-posteriors obtained from the raw/cal llks applying mismatched data priors 
  to simulate a system that was trained with the wrong priors

* logpost_calto_llks: calibrated version of logpost_mismp where calibration is done using
  uniform target priors which produces an estimate of the log scaled lks.

* logpost_calto_logpost_with_data_priors: calibrated version of logpost_mismp where calibration
  is done using the data priors to directly produce logposteriors that are matched to the test data.\n""")

print("%-50s | "%"", end='')
for cost_name, cost in cost_dict.items():
    print("  %-17s |"%cost_name, end='')
print("")
print("%-50s | "%"Score_type", end='')
for cost_name, cost in cost_dict.items():
    print("    raw      cal    |", end='')
print("")

for score_name in ['logpost_datap', 'logpost_mismp', 'logpost_calto_llks', 'logpost_calto_logpost_with_data_priors']: 
    
    print("%-50s | "%score_name, end='')
    
    for cost_name, cost in cost_dict.items():

        for raworcal in ['raw', 'cal']:

          scores = score_dict["%s_%s"%(raworcal, score_name)]
          score_type = 'log_likelihoods' if 'llk' in score_name else 'log_posteriors'
          ecval, _ = ec.average_cost_for_bayes_decisions(targets, scores, cost, priors=data_priors, adjusted=adjusted_cost, score_type=score_type, silent=True)
          print("  %5.3f   "%ecval, end='')

        print("|", end='')

    print('')


print("""\nNote that:
* The difference between the raw and cal column for each cost in the logpost_datap row shows the effect of miscalibration 
  in the likelihoods used to compute the log-posteriors.
* The difference between the logpost_datap and logpost_mismp results in the same column indicate the effect of using 
  mismatched priors when computing the log-posteriors.
* The last two rows show the effect of calibration using an affine calibration transformation. 
* Note that the columns called cal correspond to perfectly calibrated likelihoods obtained from the distributions used
  for simulation. Hence, they should always be no worse than the scores calibrated with the affine model.\n""")


