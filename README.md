# Expected cost

Methods for computing the expected cost on an evaluation dataset. 
Given a matrix of user-defined costs for each combination of true class and decision made by the system, 
this metric is estimated as the average of the costs across all samples in the dataset.

The decisions needed for evaluation of the cost can be:

* hard decisions obtained with some external method, 

* Bayes decisions made by optimizing the cost given the scores from a system and assuming they can be used to obtain well-calibrated posteriors, or

* (for the binary case and the standard square cost function) optimal decisions made by optimizing the cost by choosing the decision threshold that minimizes the cost.

