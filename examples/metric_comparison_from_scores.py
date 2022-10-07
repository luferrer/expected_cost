""" Script used to generate the plots and results in the paper:
"Analysis and Comparison of Classification Metrics"

Note that here the indices for the classes are taken to be 0 and 1
(to be consistent with the fact that the average_cost repository
assumes that classes go from 0 to C-1), which correspond to 1 and 2 
in the paper, respectively.
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import expit, logit
from classmetrics import avcost
from classmetrics import utils
from IPython import embed


def plot_vertical_line(x, ylim, style):
    plt.plot([x,x], ylim, style)

def value_at_thr(values, thrs, sel_thr):
    # Find the value in thrs that is closest to sel_thr
    # and then return the value in values for that thr.
    i = np.argmin(np.abs(np.array(thrs)-sel_thr))
    return values[i]


#########################################################################################
# Ccreate scores using a Gaussian distribution for each class. These scores will not
# necessarily be well calibrated.

plt.figure()

N0 = 100000
N1 = 10000
K = N0 + N1

std0 = 1.0
std1 = 1.0
mean0 = -1.5
mean1 = 1.0

raw_scores0 = np.random.normal(mean0, std0, N0)
raw_scores1 = np.random.normal(mean1, std1, N1)
raw_scores = np.r_[raw_scores0, raw_scores1]
targets = np.r_[np.zeros(N0), np.ones(N1)]

# Now get the LLRs given the model we chose for the distributions
# These new scores are well-calibrated by definition
cal_scores = utils.get_llr_for_gaussian_model(raw_scores, mean0, mean1, std0, std1)

# Plot the resulting distributions
plt.figure()
c, hs = utils.make_hist(targets, raw_scores)
plt.plot(c, hs[1], 'r-', label='raw_scores')
plt.plot(c, hs[0], 'r:')

c, hs = utils.make_hist(targets, cal_scores)
plt.plot(c, hs[1], 'b-', label='cal_scores')
plt.plot(c, hs[0], 'b:')
plt.legend()

plt.savefig("metric_comparison_from_scores.dists.pdf")

# Now, we take raw and calibrated scores and choose a bunch of
# different decision thresholds and compute a few metrics
# for each case.
score_dict = {'raw_scores': raw_scores, 'cal_scores': cal_scores}

# We consider two priors, the ones in the data and the uniform one
priors_data = np.array([N0/K, N1/K])
priors_unif = np.array([0.5, 0.5])

# We consider to cost matrices, the usual 0-1 matrix
# and one with a higher weight for K10
costs_01 = avcost.cost_matrix([[0, 1], [1, 0]])
costs_0b = avcost.cost_matrix([[0, 1], [2, 0]])

colors = {'EC1': 'b', 'EC2': 'r', 'FS': 'g'} #, 'MCC': 'k'}
metrics = colors.keys()

for score_name, scores in score_dict.items():

    print("*** Metric values at different thresholds for %s"%score_name)

    # The dict below will accumulate a list of values for each threshold
    # for each metric
    metric_dict = dict([(m, []) for m in metrics])

    thrs = np.arange(-1,3,0.01)

    for thr in thrs:

        # Number of samples of class 0 with a score larger than the thr (ie, labelled as class 1)
        K01 = np.sum(scores[targets==0]>thr)
        # Number of samples of class 1 with a score smaller than the thr (ie, labelled as class 0)
        K10 = np.sum(scores[targets==1]<thr)

        R = utils.compute_R_matrix_from_counts_for_binary_classif(K01, K10, N0, N1)

        metric_dict['EC1'].append(avcost.average_cost_from_confusion_matrix(R, priors_unif, costs_01, adjusted=True))
        metric_dict['EC2'].append(avcost.average_cost_from_confusion_matrix(R, priors_data, costs_0b, adjusted=True))
        metric_dict['FS'].append(utils.Fscore(K10, K01, N0, N1))
        #metric_dict['MCC'].append(utils.MCCoeff(K10, K01, N0, N1))


    for metric_name, metric_list in metric_dict.items():
        metric_dict[metric_name] = np.array(metric_list)

    plt.figure(figsize=(6,4))
    plt.plot(thrs, metric_dict['EC1'], label=r'$\mathrm{NEC}_u$', color=colors['EC1'])
    plt.plot(thrs, metric_dict['EC2'], label=r'$\mathrm{NEC}_{\beta^2=2}$', color=colors['EC2'])
    plt.plot(thrs, 1-metric_dict['FS'], label=r'$1-\mathrm{FS}_{\beta=1}$', color=colors['FS'])
   # plt.plot(thrs, -metric_dict['MCC'], label="-1 * MCC", color=colors['MCC'])

    thr_dict = dict()
    thr_dict['bayes_thr_for_EC1'] = utils.bayes_thr_for_binary_classif(priors_unif, costs_01)
    thr_dict['best_thr_for_EC1']  = thrs[np.nanargmin(metric_dict['EC1'])]
    thr_dict['bayes_thr_for_EC2'] = utils.bayes_thr_for_binary_classif(priors_data, costs_0b)
    thr_dict['best_thr_for_EC2']  = thrs[np.nanargmin(metric_dict['EC2'])]
    thr_dict['best_thr_for_FS']   = thrs[np.nanargmin(1-metric_dict['FS'])]
    #thr_dict['best_thr_for_MCC']  = thrs[np.nanargmin(-metric_dict['MCC'])]

    ylim = plt.ylim()
    for metric in metrics:
        plot_vertical_line(thr_dict['best_thr_for_'+metric], ylim, colors[metric]+':')
        if 'EC' in metric:
            plot_vertical_line(thr_dict['bayes_thr_for_'+metric], ylim, colors[metric]+'--') 
    plt.xlabel("Threshold")
    plt.legend()
    plt.savefig("metric_comparison_from_%s.pdf"%score_name)

    # Now, print the value of every metric for every threshold above
    print("    %-20s ( %6s ) "%("Thr_type", "Thr_val"), end='')
    for metric in metrics:
        print("%5s  "%metric, end='')
    print("")

    for thr_name, thr in thr_dict.items():
        print("    %-20s ( %6.3f  ) "%(thr_name, thr), end='')
        for metric in metrics:
            print("%5.2f  "%value_at_thr(metric_dict[metric], thrs, thr), end='')
        print("")
    print("")


print("""Note that:
* The metrics are immune to calibration issues when computed for decisions made with thresholds optimized for any metric.
* With FS and MCC we have no way to tell if we have a calibration problem.
* With EC, we can compare the metric for the bayes and the best threshold. If the first is much worse than the latter, then we know we have a calibration problem.
* For this particular data, the decisions would be very similar if we optimize the threshold for EC2, FS or MCC.
* You can play with the priors, costs, or score distribution parameters to see how the metrics change.""")

