import functools
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler

from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import AccuracyMetric, AverageMetric, EstimatedTimeArrival
from codebase.torchutils.common import GradientAccumulator
from codebase.torchutils.common import ThroughputTester, time_enumerate

_logger = logging.getLogger(__name__)

scaler = None


def _run_one_epoch(is_training: bool,
                   epoch: int,
                   model: nn.Module,
                   loader: data.DataLoader,
                   criterion: nn.modules.loss._Loss,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   use_amp: bool,
                   accmulated_steps: int,
                   device: str,
                   memory_format: str,
                   log_interval: int):
    phase = "train" if is_training else "eval"
    model.train(mode=is_training)

    global scaler
    if scaler is None:
        scaler = GradScaler(enabled=use_amp and is_training)

    gradident_accumulator = GradientAccumulator(steps=accmulated_steps, enabled=is_training)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    eta = EstimatedTimeArrival(len(loader))
    speed_tester = ThroughputTester()

    if is_training and scheduler is not None:
        scheduler.step(epoch)

    lr = optimizer.param_groups[0]['lr']
    _logger.info(f"{phase.upper()} start, epoch={epoch:04d}, lr={lr:.6f}")

    all_outputs = []
    all_targets = []

    for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
        inputs = inputs.to(device=device, non_blocking=True, memory_format=memory_format)
        targets = targets.to(device=device, non_blocking=True)

        with torch.set_grad_enabled(mode=is_training):
            with autocast(enabled=use_amp and is_training):
                outputs = model(inputs)
                loss: torch.Tensor = criterion(outputs, targets)

        all_outputs.append(outputs)
        all_targets.append(targets)

        gradident_accumulator.backward_step(model, loss, optimizer, scaler)

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        accuracy_metric.update(outputs, targets)
        eta.step()
        speed_tester.update(inputs)

        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                phase.upper(),
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{accuracy_metric}",
                f"{eta}",
            ]))
            time_cost_metric.reset()
            speed_tester.reset()

    _logger.info(", ".join([
        phase.upper(),
        f"epoch={epoch:04d} {phase} complete",
        f"{loss_metric}",
        f"{accuracy_metric}",
    ]))

    return {
        f"{phase}/lr": lr,
        f"{phase}/loss": loss_metric.compute(),
        f"{phase}/top1_acc": accuracy_metric.at(1).rate,
        f"{phase}/top5_acc": accuracy_metric.at(5).rate,
        f"{phase}/targets": torch.concat(all_targets),
        f"{phase}/outputs": torch.concat(all_outputs),
    }


train_one_epoch = functools.partial(_run_one_epoch, is_training=True)
evaluate_one_epoch = functools.partial(_run_one_epoch, is_training=False)
