import numpy as np
from scipy.special import logsumexp 
from expected_cost import ec, utils
from data import get_llrs_for_bin_classif_task

adjusted_cost = True

# Template for analysis of binary classification scores. To change the input
# data, you can add your own loading process in the get_llrs_for_bin_classif_task
# in the data.py file.

P1 = 0.1
priors = [(1-P1), P1]
targets, raw_llrs, cal_llrs = get_llrs_for_bin_classif_task('gaussian_sim', prior1=P1)
raw_logpost = utils.llrs_to_logpost(raw_llrs, priors)
cal_logpost = utils.llrs_to_logpost(cal_llrs, priors)


# Now, compute a family of cost matrices varying the cost for one of the classes
# (in the log domain) and leaving the other one fixed at 1.
# For each of these matrices compute the cost for maximum-a-posterior
# decisions and the cost for Bayes decisions.
# Also, compute the cost we would get by finding the decision threshold empirically 
# to optimize the specific cost function. 

print("*** Average cost for cost matrix = [[0 1] [alpha 0]]\n")

print("Alpha        MAP         Bayes  Bayes_after_cal Optimal")

map_decisions = np.argmax(raw_logpost, axis=-1)

for logalpha in np.arange(-4, 4, 0.5):

    alpha = np.exp(logalpha)
    cost = ec.cost_matrix([[0, 1],[alpha, 0]])  

    ac_map                = ec.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ac_bayes,           _ = ec.average_cost_for_bayes_decisions(targets, raw_logpost, cost, adjusted=adjusted_cost)
    ac_min                = ec.average_cost_for_optimal_decisions(targets, raw_logpost, cost, adjusted=adjusted_cost)
    ac_bayes_after_cal, _ = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost, adjusted=adjusted_cost)

    print("%6.3f      %6.3f      %6.3f      %6.3f      %6.3f"%(alpha, ac_map, ac_bayes, ac_bayes_after_cal, ac_min, ))

print("""\nNote that:
* Columns 2 through 5 are the average cost when decisions are made with different algorithms.
* The name of the column indicates how the decisions were made: MAP decisions, Bayes decisions on raw scores,
  Bayes decisions on calibrated scores, optimal decisions (ie, selecting the best threshold for the test data).
* The difference between MAP and Bayes shows the suboptimality of MAP decisions when the costs are not equal.
* The difference between the Bayes and the Bayes_after_cal shows how well or badly calibrated the system scores are. 
  Larger differences indicate a more severe calibration problem at that specific operating point. For this data, 
  calibration is fine around the equal-cost point, but degrades significantly as the costs become more imbalanced.
* The difference between Bayes_after_cal and Optimal indicates that either the calibration did not work perfectly for 
  that operating point or that the optimal cost is, in fact, too optimistic since the threshold is selected on the 
  evaluation data itself.\n\n""")


print("*** Average cost for cost matrix with an abstention option = [[0 1 alpha] [1 0 alpha]].\n")
print("Alpha      MAP      Bayes    Num_abstentions_with_Bayes   Bayes_after_cal  Num_abstentions_with_Bayes_after_cal")

for logalpha in np.arange(-4, 0.5, 0.5):

    alpha = np.exp(logalpha)
    cost = ec.cost_matrix([[0, 1, alpha],[1, 0, alpha]])  

    ac_map                                  = ec.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ac_bayes,           decisions,          = ec.average_cost_for_bayes_decisions(targets, raw_logpost, cost, adjusted=adjusted_cost)
    ac_bayes_after_cal, decisions_after_cal = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost, adjusted=adjusted_cost)
    print("%6.3f   %6.3f    %6.3f        %6d                  %6.3f         %6d"%(alpha, ac_map, ac_bayes, np.sum(decisions==2), ac_bayes_after_cal, np.sum(decisions_after_cal==2)))


print("""\nNote that
* The lower the cost of abstention, the more samples get this label and the worse the MAP decisions are (since 
  they do not take advantage of the abstention option). 
* If the cost of abstention is too high, the system never chooses to abstain.
* The cost is much lower after calibration, showing again that, for this data, the original scores were 
  not well-calibrated across all possible operating points. 
* Importantly, in this case where the cost function has an abstention decision, optimal decisions cannot be made 
  by sweeping a threshold as for the square cost function above because there are three possible decisions.\n""")



