# Wrappers for psrcal methods which work with tensors, while this repo works with numpy arrays

from psrcal import losses
import torch
import numpy as np

def loss_from_psrcal(loss, log_probs, labels, norm=True, priors=None):

    if priors is not None:
        priors = torch.tensor(priors)
    return loss(torch.tensor(log_probs), torch.tensor(labels), norm=norm, priors=priors).detach().numpy()


def Brier(log_probs, labels, norm=True, priors=None):

    return loss_from_psrcal(losses.Brier, log_probs, labels, norm, priors)


def LogLoss(log_probs, labels, norm=True, priors=None):

    return loss_from_psrcal(losses.LogLoss, log_probs, labels, norm, priors)


def ECE(log_probs, labels, M=15, return_values=False):

    out = losses.ECE(torch.tensor(log_probs), torch.tensor(labels), M, return_values)
    return [v.detach().numpy() if torch.is_tensor(v) else v for v in out] if return_values else out.detach().numpy()


def ECEbin(log_probs, labels, M=15, return_values=False):

    out = losses.ECEbin(torch.tensor(log_probs), torch.tensor(labels), M, return_values)
    return [v.detach().numpy() if torch.is_tensor(v) else v for v in out] if return_values else out.detach().numpy()


def L2ECEbin(log_probs, labels, M=15, return_values=False):

    out = losses.ECEbin(torch.tensor(log_probs), torch.tensor(labels), M, return_values, l2norm=True)
    return [v.detach().numpy() if torch.is_tensor(v) else v for v in out] if return_values else out.detach().numpy()


def CalLoss(metric, raw_scores, cal_scores, targets, relative=True, **metric_kwargs):
    r = metric(raw_scores, targets, metric_kwargs)
    c = metric(cal_scores, targets, metric_kwargs)
    if r>0 and not np.isnan(r) and not np.isinf(r):
        if relative:
            return (r-c)/r*100 
        else:
            return (r-c)*100
    else:
        return np.nan
