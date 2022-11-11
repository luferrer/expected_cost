# Script used to generate some of the results (Figure 2, Tables 1 and 2) in 
# "Analysis and Comparison of Classification Metrics", arXiv:2209.05355

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import expit, logit
from expected_cost import ec, utils

outdir = "outputs/metric_comparison_with_conf_matrix"
utils.mkdir_p(outdir)

#########################################################################################
# Generate a family of confusion matrices, compute different metrics from it, 
# and plot their comparisons

K  = 1000
# In the paper, this is called R1*. For the balanced case, it should be set to 500, 
# for the imbalanced set, to 100.
for N1 in [100, 500]:


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,2.5))

    out_name = "%s/results_for_prior1_%.2f"%(outdir,N1/K)
    print("Printing results and dumping plots in %s*"%out_name)

    outf = open(out_name+".results", "w")
    outf.write("%4s & %4s & %9s & %9s & %9s & %9s & %9s &  %5s & %5s  &  %5s & %5s\\\\\n"%
        ("K10", "K01", "NEC_b", "NEC_beta^2=1", "NEC_beta^2=2", "FS", "MCC", "R21", "R12", "R*2", "R*1"))

    N0 = K-N1 # R0* in the paper

    # We consider two priors, the ones in the data and the uniform one
    priors_data = np.array([N0/K, N1/K])
    priors_unif = np.array([0.5, 0.5])

    # We consider to cost matrices, the usual 0-1 matrix
    # and one with a higher weight for K10
    costs_01 = ec.cost_matrix([[0, 1], [1, 0]])
    costs_0b = ec.cost_matrix([[0, 1], [2, 0]])

    # K10: number of class 1 samples labelled as 0
    for K10 in np.arange(0,N1,int(N1/20)):

        EC1s = []
        EC2s = []
        EC3s = []
        FSs  = []
        MCCs = []

        # K01: number of class 0 samples labelled as 1
        for K01 in np.arange(0,N0,int(N0/20)):

            R = utils.compute_R_matrix_from_counts_for_binary_classif(K01, K10, N0, N1)
            
            EC1s.append(ec.average_cost_from_confusion_matrix(R, priors_unif, costs_01, adjusted=True))
            EC2s.append(ec.average_cost_from_confusion_matrix(R, priors_data, costs_01, adjusted=True))
            EC3s.append(ec.average_cost_from_confusion_matrix(R, priors_data, costs_0b, adjusted=True))
            
            FSs.append(utils.Fscore(K10, K01, N0, N1))
            MCCs.append(utils.MCCoeff(K10, K01, N0, N1))
            
            if (K10==0 or K01==0 or K10==K01 or np.abs(K01-10*K10)<100) and np.around(EC1s[-1],1) in [0.1,0.5,0.9]:
                outf.write("%4d & %4d & %9.2f & %9.2f & %9.2f & %9.2f & %9.2f &  %5.2f & %5.2f &  %5.2f & %5.2f \\\\\n"%
                     (K10, K01, EC1s[-1], EC2s[-1], EC3s[-1], FSs[-1], MCCs[-1], K10/N1, K01/N0, (N1-K10+K01)/K, (N0-K01+K10)/K))

        ax1.plot(EC1s, MCCs, 'b.', label="K10=%d"%K10)
        ax2.plot(EC2s, FSs,  'b.', label="K10=%d"%K10)
        ax3.plot(EC3s, FSs,  'b.', label="K10=%d"%K10)

    ax1.set_xlabel(r'$\mathrm{NEC}_u$')
    ax1.set_ylabel("MCC")
    ax1.plot([1,1],[-1,1],'k:',linewidth=2)

    ax2.set_xlabel(r'$\mathrm{NEC}_{\beta^2=1}$')
    ax2.set_ylabel(r'$\mathrm{F}_{\beta=1}$')
    ax2.set_xlim(0,2)
    ax2.plot([1,1],[0,1],'k:',linewidth=2)

    ax3.set_xlabel(r'$\mathrm{NEC}_{\beta^2=2}$')
    ax3.set_ylabel(r'$\mathrm{F}_{\beta=1}$')
    ax3.set_xlim(0,2)
    ax3.plot([1,1],[0,1],'k:',linewidth=2)

    fig.tight_layout()
    fig.savefig(out_name+".pdf")
