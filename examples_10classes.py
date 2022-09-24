import numpy as np
from scipy.special import logsumexp 
import avcost

adjusted_cost = True

# Load some scores from a resnet and targets for cifar10 data
print("\n**** Scoring in CIFAR10 data (10 classes) *****\n")
preacts = np.load("data/resnet-50_cifar10/predictions.npy")
targets = np.load("data/resnet-50_cifar10/targets.npy")

num_targets = np.max(targets)+1

# Compute log-softmax to get log posteriors. 
logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

# Print the priors from the data and the priors that are
# derived from the posteriors from the model. In this case,
# they are consistent, both are uniform.
priors_from_data = np.bincount(targets)/len(targets)
priors_from_post = np.mean(np.exp(logpost), axis=0)
print("Priors from data = %s"%" ".join(['{:.1f}'.format(i) for i in priors_from_data]))
print("Priors from post = %s"%" ".join(['{:.1f}'.format(i) for i in priors_from_post]))
print("Priors are consistent so these posteriors can be used to make Bayes decisions for this data "+
    "(assuming they are well calibrated)\n")

# Make maximum a posteriori decisions. These decisions are optimal 
# for the 0-1 cost function.
map_decisions = np.argmax(logpost, axis=-1)
cost_01 = avcost.cost_matrix.zero_one_costs(num_targets)
ac_01 = avcost.average_cost(targets, map_decisions, cost_01, adjusted=adjusted_cost)

# We can now check that we get the same cost if we make Bayes
# decisions based on the posteriors
ac_01_bayes, bayes_decisions_01, _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost_01, adjusted=adjusted_cost)

print("Average cost for 0-1 cost matrix using MAP decisions:   %.4f"%ac_01)
print("Average cost for 0-1 cost matrix using Bayes decisions: %.4f"%ac_01_bayes)
print("These two costs are the same because Bayes decisions for 0-1 cost coincide with MAP decisions.\n")

# Now try a cost with an abstention decision and evaluate the MAP decisions.
# Note that MAP will never choose the abstention decision.
ab_cost = 0.2
cost_ab = avcost.cost_matrix.zero_one_costs(num_targets, abstention_cost=ab_cost)
ac_ab = avcost.average_cost(targets, map_decisions, cost_ab, adjusted=adjusted_cost)

# Now, can we do any better than MAP decisions for this specific cost function?
# Let's see what Bayes decisions give us.
ac_ab_bayes, bayes_decisions_ab, _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost_ab, adjusted=adjusted_cost)

print("Average cost for cost matrix with abstention cost of %.1f using MAP decisions:   %.4f"%(ab_cost, ac_ab))
print("Average cost for cost matrix with abstention cost of %.1f using Bayes decisions: %.4f"%(ab_cost, ac_ab_bayes))
print("These two costs are not the same because optimal (Bayes) decisions include %d abstentions on the hardest samples.\n"%len(np.where(bayes_decisions_ab!= map_decisions)[0]))


# Non let's assume we expect that, on the actual data the system will be used for the priors
# are not uniform but rather one of the classes is much more frequent than the others:
nu_priors = np.array([0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# Let's find the average cost for the 0-1 cost matrix when making Bayes decisions
# with the posteriors given by the system. The Bayes decisions will not change, but the cost will.
# Note that the method outputs a warning in this case, since you are using a cost function
# with explicit priors, but feeding posteriors that may have a different implicit prior than 
# the provided one.
ac_01_nup_bayes, _, ac_01_nup_bayes_with_llks0 = avcost.average_cost_for_bayes_decisions(targets, logpost, cost_01, nu_priors, adjusted=adjusted_cost)

# Now, let's see if we can do better. Since the posteriors have the prior embedded in them,
# the right thing to do in this case is to convert them to scaled-likelihoods, getting rid
# of the prior:
log_scaled_likelihoods = logpost - np.log(priors_from_post)

# Now, with these new scores, we can make Bayes decisions again but this time the method
# will use the provided priors to turn the likelihoods into posteriors. Now the warning
# goes away because the posteriors are now computed internally using the right priors,
# so they are guaranteed to have the correct implicit priors for this cost function.
# Note that the average_cost_for_bayes_decisions, when computed using posteriors or log-posteriors
# internally does this same computation and outputs the cost as a third returned value.
# Here we did it again just to show the procedure, but you can check that:
# ac_01_nup_bayes_with_llks == ac_01_nup_bayes_with_llks0
ac_01_nup_bayes_with_llks, _ = avcost.average_cost_for_bayes_decisions(targets, log_scaled_likelihoods, cost_01, nu_priors, adjusted=adjusted_cost, score_type='log_likelihoods')

print("Average cost for 0-1 cost matrix and non-uniform priors using Bayes decisions with the original posteriors from the model:         %.4f"%ac_01_nup_bayes)
print("Average cost for 0-1 cost matrix and non-uniform priors using Bayes decisions with the scaled likelihoods and the target priors:   %.4f"%ac_01_nup_bayes_with_llks)
print("The second average cost is much lower because the original priors are quite bad for making decisions when priors are non-uniform.\n")






