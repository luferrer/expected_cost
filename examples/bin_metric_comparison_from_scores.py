import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import expit, logit, logsumexp 
from expected_cost import ec, utils
from data import get_llrs_for_bin_classif_task
from IPython import embed
import re


def plot_vertical_line(x, ylim, style):
    plt.plot([x,x], ylim, style)

def value_at_thr(values, thrs, sel_thr):
    # Find the value in thrs that is closest to sel_thr
    # and then return the value in values for that thr.
    i = np.argmin(np.abs(np.array(thrs)-sel_thr))
    return values[i]


outdir = "outputs/metric_comparison_from_scores"
utils.mkdir_p(outdir)

targets, raw_scores, cal_scores = get_llrs_for_bin_classif_task('gaussian_sim', prior1=0.1)
N0 = sum(targets==0)
N1 = sum(targets==1)
K = N0 + N1

# Plot the resulting distributions
plt.figure()
c, hs = utils.make_hist(targets, raw_scores)
plt.plot(c, hs[1], 'r-', label='raw_scores')
plt.plot(c, hs[0], 'r:')

c, hs = utils.make_hist(targets, cal_scores)
plt.plot(c, hs[1], 'b-', label='cal_scores')
plt.plot(c, hs[0], 'b:')
plt.legend()

plt.savefig("%s/score_dists.pdf"%outdir)

# Now, we take raw and calibrated scores and choose a bunch of
# different decision thresholds and compute a few metrics
# for each case.
score_dict = {'raw_scores': raw_scores, 'cal_scores': cal_scores}

# We consider two priors, the ones in the data and the uniform one
priors_data = np.array([N0/K, N1/K])
priors_unif = np.array([0.5, 0.5])

# We consider to cost matrices, the usual 0-1 matrix
# and one with a higher weight for K10
costs_01 = ec.cost_matrix([[0, 1], [1, 0]])
costs_0b = ec.cost_matrix([[0, 1], [2, 0]])

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

        metric_dict['EC1'].append(ec.average_cost_from_confusion_matrix(R, priors_unif, costs_01, adjusted=True))
        metric_dict['EC2'].append(ec.average_cost_from_confusion_matrix(R, priors_data, costs_0b, adjusted=True))
        metric_dict['FS'].append(utils.Fscore(K10, K01, N0, N1))
        #metric_dict['MCC'].append(utils.MCCoeff(K10, K01, N0, N1))


    for metric_name, metric_list in metric_dict.items():
        metric_dict[metric_name] = np.array(metric_list)

    plt.figure(figsize=(4,3.5))
    plt.plot(thrs, metric_dict['EC1'], label=r'$\mathrm{NEC}_u$', color=colors['EC1'])
    plt.plot(thrs, metric_dict['EC2'], label=r'$\mathrm{NEC}_{\beta^2=2}$', color=colors['EC2'])
    plt.plot(thrs, 1-metric_dict['FS'], label=r'$1-\mathrm{FS}_{\beta=1}$', color=colors['FS'])
   # plt.plot(thrs, -metric_dict['MCC'], label="-1 * MCC", color=colors['MCC'])

    thr_dict = dict()
    thr_dict['bayes_thr_for_EC1'] = utils.bayes_thr_for_llrs(priors_unif, costs_01)
    thr_dict['best_thr_for_EC1']  = thrs[np.nanargmin(metric_dict['EC1'])]
    thr_dict['bayes_thr_for_EC2'] = utils.bayes_thr_for_llrs(priors_data, costs_0b)
    thr_dict['best_thr_for_EC2']  = thrs[np.nanargmin(metric_dict['EC2'])]
    thr_dict['best_thr_for_FS']   = thrs[np.nanargmin(1-metric_dict['FS'])]
    #thr_dict['best_thr_for_MCC']  = thrs[np.nanargmin(-metric_dict['MCC'])]

    ylim = plt.ylim()
    for metric in metrics:
        plot_vertical_line(thr_dict['best_thr_for_'+metric], ylim, colors[metric]+':')
        if 'EC' in metric:
            plot_vertical_line(thr_dict['bayes_thr_for_'+metric], ylim, colors[metric]+'--') 
    plt.xlabel("Threshold")
    plt.legend(loc='upper right')
    plt.title(re.sub("cal", "calibrated", re.sub("_"," ",score_name)))
    plt.tight_layout()
    plt.savefig("%s/%s.pdf"%(outdir, score_name))

    # Now, print the value of every metric for every threshold above
    print("    %-20s ( %6s ) "%("Thr_type", "Thr_val"), end='')
    for metric in metrics:
        print("%5s  "%metric, end='')
    print("")

    for thr_name, thr in thr_dict.items():
        print("    %-20s ( %6.3f  ) "%(thr_name, thr), end='')
        for metric in metrics:
            print("%5.3f  "%value_at_thr(metric_dict[metric], thrs, thr), end='')
        print("")
    print("")


print("""Note that:
* The metrics are immune to calibration issues when computed for decisions made with thresholds optimized for any metric.
* With FS we have no way to tell if we have a calibration problem.
* With EC, we can compare the metric for the bayes and the best threshold. If the first is much worse than the latter, then we know we have a calibration problem.
* The thresholds selected for EC1 are highly suboptimal for EC2, and conversely.
* For this particular data, the decisions would be the same if we optimize the threshold for EC2 or FS.
* You can play with the priors, costs, or score distribution parameters to see how the metrics change.
* Plots all three metrics as a function of the threshold can be found in the %s dir."""%outdir)

