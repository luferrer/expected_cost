# Expected cost

Methods for computing the expected cost (EC) on an evaluation dataset, as defined in statistical learning text books (e.g., Bishop's "Pattern recognition and machine learning", and Hastie's et al "The elements of statistical learning"). 
Given a matrix of user-defined costs with elements $c_{y\ d}$, where $y$ is the true class of a sample and $d$ is the decision made by the system for that sample, 
this metric is estimated as the average of the costs across all samples in the dataset. That is:

$\mathrm{EC} = \frac{1}{N} \sum_i c_{y_i,d_i}$

where the sum runs over the $N$ samples in the evaluation set and $c_{y_i,d_i}$ is the cost incurred at sample $i$.

The EC is a generalization of the total error (which, in turn, is 1 minus the accuracy) and the balanced total error (which is 1 minus the balanced accuracy). The generalization is in the following ways: (1) it allows for costs that are different for each type of error, and (2) it allows for decisions that do not correspond one to one to the classes (e.g., it allows for the introduction of an "abstain" decision). The EC comes with an elegant theory on how to make optimal decisions given a certain set of costs, and it enables analysis of calibration. For these reasons we believe it is superior to other commonly used classification metrics, like the F-beta score or the Mathews correlation coefficient. All these issues are discussed in detail in:

*L. Ferrer, ["Analysis and Comparison of Classification Metrics"](https://arxiv.org/abs/2209.05355), 	arXiv:2209.05355*

The results in the paper can be replicated with the code in the examples directory in this repository.

The code provides methods for computing the EC when decisions are given by:

* hard decisions obtained with some external method, 

* Bayes decisions made by optimizing the cost given the scores from a system assuming they can be used to obtain well-calibrated posteriors, or

* optimal decisions made by choosing the decision threshold that minimizes the cost. This last option is only applicable for the binary case and the standard square cost function.

The scripts in the examples directory can be used with any dataset of scores and targets. See the examples/data.py file for examples on how to load your own data in the format required by the examples.

## How to install

You can install this package as:

```pip install expected_cost```

which will also install all the dependencies. Some of the notebooks in this repository also require the psrcal package, which you can install as:

```pip install psrcal```

This is not included in the requirements of expected_cost because its installation requires pytorch, which takes a while. If you only need to compute expected cost or make Bayes decisions, and do not want to do or evaluate calibration, you do not need psrcal (or pytorch). 

If you want the latest stuff, along with all the notebooks, you can do the following:

1. Clone this repository:  

   ```git clone https://github.com/luferrer/expected_cost.git```

2. Install the requirements:  
   
   ```pip install -r requirements.txt```
   
   (You can delete the psrcal line if you do not need calibration capabilities).

3. Add the resulting top directory in your PYTHONPATH. In bash this would be:

   ```export PYTHONPATH=ROOT_DIR/expected_cost:$PYTHONPATH```

where ROOT_DIR is the absolute path (or the relative path from the directory where you have the scripts or notebooks you want to run) to the top directory from where you did the clone above.
 
4. You can now run any notebook in the notebooks directory.

