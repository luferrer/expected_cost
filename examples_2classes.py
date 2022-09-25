import numpy as np
from scipy.special import logsumexp 
import avcost
from IPython import embed

adjusted_cost = True

# Load some scores from a resnet and targets for cifar10 data
print("\n**** Scoring in CIFAR10 data (first two classes) *****\n")
preacts = np.load("data/resnet-50_cifar10/predictions.npy")
targets = np.load("data/resnet-50_cifar10/targets.npy")

# Keep only two of the scores and the samples from those two classes
# to fake a 2-class problem.
# You can edit the indices below to see 
sel = targets <= 1
targets = targets[sel]
preacts = preacts[sel][:,:2]

num_targets = np.max(targets)+1

# Compute log-softmax to get log posteriors. 
logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)


# Now, compute a family of cost matrices varying the cost for one of the classes
# in the log domain and leaving the other one fixed at 1.
# For each of these matrices compute the cost for maximum-a-posterior
# decisions and the cost for Bayes decisions.
# Also, compute the cost we would get by finding the  decision threshold empirically to 
# optimize the specific cost function. 
map_decisions = np.argmax(logpost, axis=-1)

print("Results for cost matrix = [[0 1] [alpha 0]]\n")
print("Alpha      MAP      Bayes     Optimal")


for logalpha in np.arange(-4, 4, 0.5):

    alpha = np.exp(logalpha)
    cost = avcost.cost_matrix([[0, 1],[alpha, 0]])  

    ac_map         = avcost.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ac_bayes, _, _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost, adjusted=adjusted_cost)
    ac_min         = avcost.average_cost_for_optimal_decisions(targets, logpost, cost, adjusted=adjusted_cost)

    print("%6.3f   %6.3f    %6.3f    %6.3f"%(alpha, ac_map, ac_bayes, ac_min))


print("\nThe difference between MAP and Bayes shows the suboptimality of MAP decisions when the costs are not equal. "+
    "On the other hand, the difference between the Bayes and the Optimal decisions shows how well or badly calibrated the system scores are. "+
    "Larger differences indicate a more severe calibration problem at that specific operating point. "+
    "In this case, calibration is fine around the equal-cost point, but degrades significantly as the costs become more imbalanced.\n")


print("Results for cost matrix with an abstention option = [[0 1 alpha] [1 0 alpha]].\n")
print("Alpha      MAP      Bayes    NumAbstentionsWithBayes")


for logalpha in np.arange(-4, 0.5, 0.5):

    alpha = np.exp(logalpha)
    cost = avcost.cost_matrix([[0, 1, alpha],[1, 0, alpha]])  

    ac_map         = avcost.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ac_bayes, decisions, _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost, adjusted=adjusted_cost)
    print("%6.3f   %6.3f    %6.3f    %6d"%(alpha, ac_map, ac_bayes, np.sum(decisions==2)))


print("\nThe lower the cost of abstention, the more samples get this label and the worse the MAP decisions are (which do not have the abstention option). "+
    "Further, the adjusted Bayes cost increases as alpha gets lower because the naïve system used for normalization (which would always choose to abstain) "+
    "is harder to beat (i.e., it has a lower cost).")



