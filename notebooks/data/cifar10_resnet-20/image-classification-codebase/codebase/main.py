import logging
import dataclasses
import pprint

import torch
from torch import optim
import torch.cuda
import torch.utils.data
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter
from pyhocon import ConfigTree

from codebase.config import Args
from codebase.data import DATA
from codebase.models import MODEL
from codebase.optimizer import OPTIMIZER
from codebase.scheduler import SCHEDULER
from codebase.criterion import CRITERION
from codebase.engine import train_one_epoch, evaluate_one_epoch

from codebase.torchutils.common import set_cudnn_auto_tune, set_reproducible, generate_random_seed, disable_debug_api
from codebase.torchutils.common import set_proper_device, get_device
from codebase.torchutils.common import unwarp_module
from codebase.torchutils.common import compute_nparam, compute_flops
from codebase.torchutils.common import StateCheckPoint
from codebase.torchutils.common import MetricsStore
from codebase.torchutils.common import patch_download_in_cn
from codebase.torchutils.common import only_master
from codebase.torchutils.distributed import distributed_init, is_dist_avail_and_init, is_master, world_size
from codebase.torchutils.metrics import EstimatedTimeArrival
from codebase.torchutils.logging import init_logger, create_code_snapshot
import numpy as np

_logger = logging.getLogger(__name__)


def excute_pipeline(
    only_evaluate: bool,
    start_epoch: int,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    state_ckpt: StateCheckPoint,
    states: dict,
    metric_store: MetricsStore,
    **kwargs
):
    if only_evaluate:
        metric_store += evaluate_one_epoch(
            epoch=0,
            loader=val_loader,
            **kwargs
        )
        np.save("scores.npy", metric_store["eval/outputs"][0].numpy())
        np.save("targets.npy", metric_store["eval/targets"][0].numpy())
        return

    eta = EstimatedTimeArrival(max_epochs)

    for epoch in range(start_epoch+1, max_epochs+1):
        if is_dist_avail_and_init():
            if hasattr(train_loader, "sampler"):
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)

        metric_store += train_one_epoch(
            epoch=epoch,
            loader=train_loader,
            **kwargs
        )

        metric_store += evaluate_one_epoch(
            epoch=epoch,
            loader=val_loader,
            **kwargs
        )

        for name, metric in metric_store.get_last_metrics().items():
            writer.add_scalar(name, metric, epoch)

        state_ckpt.save(metric_store=metric_store, states=states)

        eta.step()

        best_metrics = metric_store.get_best_metrics()
        _logger.info(f"Epoch={epoch:04d} complete, best val top1-acc={best_metrics['eval/top1_acc']*100:.2f}%, "
                     f"top5-acc={best_metrics['eval/top5_acc']*100:.2f}% (epoch={metric_store.best_epoch+1}), {eta}")


def prepare_for_training(conf: ConfigTree, output_dir: str, local_rank: int):
    model_config = conf.get("model")
    load_from = model_config.pop("load_from")
    model: nn.Module = MODEL.build_from(model_config)
    if load_from is not None:
        model.load_state_dict(torch.load(load_from, map_location="cpu"))

    if is_dist_avail_and_init() and conf.get_bool("sync_batchnorm"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    train_loader, val_loader = DATA.build_from(conf.get("data"), dict(local_rank=local_rank))

    criterion = CRITERION.build_from(conf.get("criterion"))

    optimizer_config: dict = conf.get("optimizer")
    basic_bs = optimizer_config.pop("basic_bs")
    optimizer_config["lr"] = optimizer_config["lr"] * (conf.get("data.batch_size") * world_size() / basic_bs)
    optimizer = OPTIMIZER.build_from(optimizer_config, dict(params=model.named_parameters()))
    _logger.info(f'Set lr={optimizer_config["lr"]:.4f} with batch size={conf.get("data.batch_size") * world_size()}')

    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    if torch.cuda.is_available():
        model = model.to(device=get_device(), memory_format=getattr(torch, conf.get("memory_format")))
        criterion = criterion.to(device=get_device())

    if conf.get_bool("use_compile"):
        if hasattr(torch, "compile"):
            _logger.info("Use torch.compile to optimize model, please wait for while.")
            model = torch.compile(
                model=model,
                **conf.get("compile")
            )
        else:
            _logger.info("PyTorch version is too old to support torch.compile, skip it.")

    if conf.get_bool("use_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # image_size = conf.get_int('data.image_size')
    # _logger.info(f"Model details: n_params={compute_nparam(model)/1e6:.2f}M, "
    #              f"flops={compute_flops(model,(1,3, image_size, image_size))/1e6:.2f}M.")

    writer = only_master(SummaryWriter(output_dir))

    metric_store = MetricsStore(dominant_metric_name="eval/top1_acc")
    states = dict(model=unwarp_module(model), optimizer=optimizer, scheduler=scheduler)
    state_ckpt = StateCheckPoint(output_dir)

    state_ckpt.restore(metric_store, states, device=get_device())

    if is_dist_avail_and_init():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    return model, train_loader, val_loader, criterion, optimizer, scheduler, \
        state_ckpt, writer, metric_store, states


def _init(local_rank: int, ngpus_per_node: int, args: Args):
    set_proper_device(local_rank)
    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")

    # patch_download_in_cn()

    if StateCheckPoint(args.output_dir).is_ckpt_exists():
        _logger.info("-"*30+"Resume from the last training checkpoints."+"-"*30)

    if set_reproducible:
        set_reproducible(generate_random_seed())
    else:
        set_cudnn_auto_tune()
        disable_debug_api()

    create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                         source_directory=".", store_directory=args.output_dir)

    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))

    distributed_init(dist_backend=args.dist_backend, init_method=args.dist_url,
                     world_size=args.world_size, rank=rank)


def main_worker(local_rank: int,
                ngpus_per_node: int,
                args: Args,
                conf: ConfigTree):

    _init(local_rank=local_rank, ngpus_per_node=ngpus_per_node, args=args)

    model, train_loader, val_loader, criterion, optimizer, \
        scheduler, saver, writer, metric_store, states = \
        prepare_for_training(conf, args.output_dir, local_rank)

    excute_pipeline(
        only_evaluate=conf.get_bool("only_evaluate"),
        start_epoch=metric_store.total_epoch,
        max_epochs=conf.get_int("max_epochs"),
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        state_ckpt=saver,
        states=states,
        metric_store=metric_store,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        use_amp=conf.get_bool("use_amp"),
        accmulated_steps=conf.get_int("accmulated_steps"),
        device=get_device(),
        memory_format=getattr(torch, conf.get("memory_format")),
        log_interval=conf.get_int("log_interval"),
    )


def main(args: Args):
    distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, args.conf))
    else:
        local_rank = 0
        main_worker(local_rank, ngpus_per_node, args, args.conf)
