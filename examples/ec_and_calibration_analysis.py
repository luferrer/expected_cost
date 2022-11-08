# Script used to generate some of the results in 
# "Analysis and Comparison of Classification Metrics", arXiv:2209.05355

import numpy as np
from scipy.special import logsumexp 
from expected_cost import ec, utils
from data import get_llks_for_multi_classif_task
import matplotlib.pyplot as plt
import re
from IPython import embed

import sys
try:
    import torch    
    from expected_cost.calibration import affine_calibration_with_crossval
    from psrcal.losses import LogLoss, ECE, Brier, CalLossBrier, CalLossLogLoss, plot_reliability_diagram
    has_psr = True
except:
    has_psr = False
    print("PSR code not available, skipping calibration experiments.")


adjusted_cost = True
latex_tables = False

sep1 = "&" if latex_tables else " "
sep2 = " " if latex_tables else "|"
sep3 = "\\\\" if latex_tables else ""

def print_header(metric_dict, score_dict):

    print("%-30s  %s"%("",sep2), end='')
    for metric_name, metric in metric_dict.items():
        print("     %-24s %s"%(metric_name,sep2), end='')
    print("")
    print("%-30s  %s"%("Score_type",sep2), end='')
    for metric_name, metric in metric_dict.items():
        for rc in score_dict.keys():
            print("%s   %s   "%(sep1,rc), end='')
        print("%s"%sep2, end='')
    print("%s"%sep3)

outdir = "outputs/ec_and_calibration_analysis"
utils.mkdir_p(outdir)

#########################################################################################################
""" Generate a bunch of different scores for a C class problem (C can be changed to whatever you like):

* llks: log scaled likelihoods with Gaussian class distributions

* logpost Datap: log-posteriors obtained from the llks applying the true data priors

* logpost Mismp: log-posteriors obtained from the llks applying the mismatched data priors 
  to simulate a system that was trained with the wrong priors

* for each of those posteriors, two calibrated versions, using an affine transformation and
  temp scaling

Further, two llk versions are used (cal, mc1): miscalibrated and calibrated ones, resulting in two versions 
of each of the above posteriors. Finally, another miscalibrated version of the posteriors (mc2) is 
created by scaling the log-posteriors directly.
"""

C = 10
p0 = 0.9 # 0.9 for the paper
data_priors = np.array([p0] + [(1-p0)/(C-1)]*(C-1))
mism_priors = np.array([(1-p0)/(C-1)]*(C-1) + [p0])
unif_priors = np.ones(C)/C

score_dict = {'cal':{}, 'mc1':{}, 'mc2':{}}

# Parameters with which to misscalibrate the scores
shift_for_raw_llks = np.zeros(C) 
shift_for_raw_llks[0] = 0.5 # 0.5 for paper
score_scale1 = 0.5 # 0.5 for paper
score_scale2 = 0.2
targets, score_dict['mc1']['llks'], score_dict['cal']['llks'] = get_llks_for_multi_classif_task('gaussian_sim', priors=data_priors, 
          sim_params={'feat_std':0.15, 'score_scale':score_scale1, 'score_shift': shift_for_raw_llks}, K=10000)

num_targets = np.max(targets)+1

for rc in ['cal', 'mc1', 'mc2']:

    if rc != 'mc2':
        llks = score_dict[rc]['llks']
        score_dict[rc]['Datap'] = utils.llks_to_logpost(llks, data_priors)
        score_dict[rc]['Mismp'] = utils.llks_to_logpost(llks, mism_priors)
    else:
        # Miscalibrate the posteriors by scaling them and renormalizing
        for pr in ['Datap', 'Mismp']:
            score_dict[rc][pr] = score_scale2 * score_dict['cal'][pr]
            score_dict[rc][pr] -= logsumexp(score_dict[rc][pr], axis=1, keepdims=True)

    if has_psr:
        for pr in ['Datap', 'Mismp']:
            #score_dict[rc]['Post-%s-affcal_llks'%pr] = affine_calibration_with_crossval(score_dict[rc]['Post-%s'%pr], targets, priors=unif_priors)
            score_dict[rc]['%s-affcal'%pr]      = affine_calibration_with_crossval(score_dict[rc][pr], targets)
            score_dict[rc]['%s-temcal'%pr]      = affine_calibration_with_crossval(score_dict[rc][pr], targets, use_bias=False)
        
    # Plot the resulting score distributions 
    for score_name, scores in score_dict[rc].items():
        utils.plot_hists(targets, scores, "%s/dists_%s_%s_C=%d.pdf"%(outdir,rc,score_name,num_targets))

#########################################################################################################
# First, using the logposteriors computed with matched priors, compute a
# family of cost matrices varying the costs for the last row (in the log
# domain) and leaving the other one fixed at 1. For each of these matrices
# compute the cost for maximum-a-posterior decisions and the cost for Bayes
# decisions. 

print("***************************************************************************************************************")
print("Average cost for cost matrix with c_ii = 0, c_ij = 1 for i!=j and i!=C, and c_ij = alpha, for i!=j and i=C, ")
print("using calibrated (cal) and misscalibrated (mc1) log posteriors computed with the data priors\n")

if num_targets == 2:
  print("Alpha        MAP_mc1     NEC_Bayes_mc1  NEC_Bayes_cal Optimal")
else:
  print("Alpha        MAP_mc1     NEC_Bayes_mc1  NEC_Bayes_cal")

map_decisions = np.argmax(score_dict['mc1']['Datap'], axis=-1)

for logalpha in np.arange(-4, 4, 0.5):

    alpha = np.exp(logalpha)
    costm = 1-np.eye(num_targets)
    costm[-1,0:-1] = alpha
    cost = ec.cost_matrix(costm)  

    ec_map                = ec.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ec_bayes,           _ = ec.average_cost_for_bayes_decisions(targets, score_dict['mc1']['Datap'], cost, adjusted=adjusted_cost)
    ec_bayes_after_cal, _ = ec.average_cost_for_bayes_decisions(targets, score_dict['cal']['Datap'], cost, adjusted=adjusted_cost)

    if num_targets == 2:
      ec_min                = ec.average_cost_for_optimal_decisions(targets, score_dict['mc1']['Datap'], cost, adjusted=adjusted_cost)
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
print("Average cost for cost matrix with c_ii=0, c_ij=1 for i!=j, and with a last column for an abstention decision with cost alpha, ")
print("using calibrated (cal) and misscalibrated (mc1) log posteriors computed with the data priors\n")
print("        |               mc1 scores                 |            cal")
print("Alpha   |   MAP    |  EC_Bayes  NEC_Bayes   %Abs   |  EC_Bayes   NEC_Bayes   %Abs")


for alpha in [0.01, 0.1, 0.2, 0.4, 0.6, 1.0]:
    cost = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=alpha)

    ec_map                                  = ec.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ec_bayes,           decisions,          = ec.average_cost_for_bayes_decisions(targets, score_dict['mc1']['Datap'], cost, adjusted=adjusted_cost)
    ec_bayes_after_cal, decisions_after_cal = ec.average_cost_for_bayes_decisions(targets, score_dict['cal']['Datap'], cost, adjusted=adjusted_cost)

    ec_bayes_nonorm,           _ = ec.average_cost_for_bayes_decisions(targets, score_dict['mc1']['Datap'], cost, adjusted=False)
    ec_bayes_after_cal_nonorm, _ = ec.average_cost_for_bayes_decisions(targets, score_dict['cal']['Datap'], cost, adjusted=False)

    perc_abs = np.sum(decisions==num_targets)/len(decisions)*100
    perc_abs_after_cal = np.sum(decisions_after_cal==num_targets)/len(decisions)*100
    if latex_tables:
      print("%6.3f  & %6.3f & %6.3f &  %6.1f  & %6.3f  &   %6.3f  & %6.1f \\\\"%(alpha, ec_bayes_nonorm, ec_bayes, perc_abs, ec_bayes_after_cal_nonorm, ec_bayes_after_cal, perc_abs_after_cal))
    else:
      print("%6.3f  |  %6.3f  |   %6.3f    %6.3f   %6.1f   |   %6.3f    %6.3f    %6.1f"%(alpha, ec_map, ec_bayes_nonorm, ec_bayes, perc_abs, ec_bayes_after_cal_nonorm, ec_bayes_after_cal, perc_abs_after_cal))
      

print("""\nNote that
* The lower the cost of abstention, the more samples get this label and the worse the MAP decisions are (since 
  they do not take advantage of the abstention option). 
* If the cost of abstention is too high, the system never chooses to abstain.
* The difference between the Bayes column and the Bayes_after_cal column show the extent of the misscalibration
  in the raw log-posteriors.
* Note that, even for the binary case, when the cost function has an abstention decision, optimal decisions cannot be made 
  by sweeping a threshold as for the square cost function above because there are three possible decisions.\n""")

#########################################################################################################
# Finally, analyze the issue of mismatched priors and calibration with various metrics

if has_psr is False:
  print("*** Calibration analysis skipped since the psr package is not available")
  sys.exit(0)

# Now we can evaluate a couple of expected costs (using Bayes decisions) and other metrics
# for each of these scores on the test data. In all cases we take the target priors to be the
# ones in the test data. 

cost_01 = ec.cost_matrix.zero_one_costs(num_targets)
cost_ab1 = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=0.1)
metric_dict = {'cost_01': cost_01, 'cost_01_abs=0.1': cost_ab1, 
             'cross-entropy': LogLoss, 'brier-score': Brier}

print("*********************************************************************************************************************************")
print("PSRs for calibrated and misscalibrated scores (see script header for details):\n")

print_header(metric_dict, score_dict)

for score_name in np.sort(list(score_dict['mc1'].keys())): 

    if 'llks' in score_name: continue
    print("%-30s  %s"%(score_name,sep2), end='')
    
    for metric_name, metric in metric_dict.items():

        for rc in score_dict.keys():

          scores = score_dict[rc][score_name]
          if 'cost' in metric_name:
            metric_value, _ = ec.average_cost_for_bayes_decisions(targets, scores, metric, adjusted=adjusted_cost, silent=True)
          else:
            metric_value = metric(torch.tensor(scores), torch.tensor(targets))

          print("%s %6.2f  "%(sep1,metric_value), end='')
        print("%s"%sep2, end='')
    print('%s'%sep3)


print("""\nNote that:
* The difference between the mc1 or mac2 columns and the and cal column for each metric shows the effect of miscalibration.
* The difference between the Datap and Mismp results in the same column show the effect of using mismatched priors when 
  computing the log-posteriors.
* The affc and tempc two rows show the effect of calibration using an affine or temp-scale calibration transformation. 
* Note that the columns called cal correspond to perfectly calibrated likelihoods obtained from the distributions used
  for simulation. Hence, they should always be no worse than the scores calibrated with the affine model.\n""")

print("*********************************************************************************************************************************")
print("Calibration loss calibrated and misscalibrated scores (see script header for details):\n")

metric_dict = {'Cal-loss-cross-entropy': CalLossLogLoss, 'Cal-loss-brier-score': CalLossBrier, 'ECE': ECE}


print_header(metric_dict, score_dict)

for score_name in np.sort(list(score_dict['mc1'].keys())): 

    if 'llks' in score_name: continue
    print("%-30s  %s"%(score_name,sep2), end='')
    
    for metric_name, metric in metric_dict.items():

        for rc in score_dict.keys():

          scores = score_dict[rc][score_name]
          if metric_name == "ECE":
            nbins = 15
            metric_value = metric(torch.tensor(scores), torch.tensor(targets), M=nbins)
            plot_reliability_diagram(torch.tensor(scores), torch.tensor(targets), "%s/reliability_diagram_%s_%s_C=%d.pdf"%(outdir,rc,score_name,num_targets), nbins=nbins)

          else:
            score_name_before_cal = re.sub("-...cal","", score_name)
            cal_scores = score_dict[rc]["%s-affcal"%(score_name_before_cal)]
            metric_value = metric(torch.tensor(scores), torch.tensor(cal_scores), torch.tensor(targets))

          print("%s %6.f  "%(sep1,metric_value), end='')
        print("%s"%sep2, end='')
    print('%s'%sep3)

print("""\nNote that:
* The calibration loss computed with cross-entropy and with Brier score are very similar for every system.
* The ECE, on the other hand, vastly underestimated the calibration problem in some cases.\n""")

