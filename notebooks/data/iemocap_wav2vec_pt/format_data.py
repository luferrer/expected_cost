
import joblib
import numpy as np

logposts = []
targets = []

for fold in np.arange(0,5):

    logposts.append(np.log(joblib.load(f'{fold}/IEMOCAPPredict/predictions')))
    targets.append(joblib.load(f'{fold}/IEMOCAPPredict/targets'))

targets = np.concatenate(targets)
logposts = np.concatenate(logposts)

assert np.all(np.sum(targets, axis=1)==1)
targets = np.where(targets)[1]

np.save("scores.npy", logposts)
np.save("targets.npy", targets)
