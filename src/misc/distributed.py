"""Single-process DDP compatibility used by the original training pipeline."""

import os

import torch
import torch.distributed as distributed
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


def init_distributed() -> bool:
    """Initialize the torchrun process group; remain non-distributed under python."""
    if 'RANK' not in os.environ or 'LOCAL_RANK' not in os.environ:
        print('Not init distributed mode.')
        return False
    distributed.init_process_group(init_method='env://')
    distributed.barrier()
    rank = distributed.get_rank()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    print('Initialized distributed mode...')
    return True


def is_initialized() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def wrap_model(model: nn.Module, *, find_unused_parameters=False, sync_bn=False):
    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if not is_initialized():
        return model
    rank = distributed.get_rank()
    return DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_loader(loader, *, shuffle: bool, seed: int):
    # Match torchrun --nproc_per_node=1 even when launched with plain python.
    replicas = distributed.get_world_size() if is_initialized() else 1
    rank = distributed.get_rank() if is_initialized() else 0
    sampler = DistributedSampler(loader.dataset, num_replicas=replicas, rank=rank,
                                 shuffle=shuffle, seed=int(seed))
    num_workers = int(loader.num_workers or 0)
    kwargs = {
        'dataset': loader.dataset,
        'batch_size': loader.batch_size,
        'sampler': sampler,
        'drop_last': loader.drop_last,
        'collate_fn': loader.collate_fn,
        'pin_memory': loader.pin_memory,
        'num_workers': num_workers,
        'persistent_workers': bool(loader.persistent_workers) if num_workers > 0 else False,
    }
    if num_workers > 0:
        kwargs['prefetch_factor'] = loader.prefetch_factor or 2
    wrapped = DataLoader(**kwargs)
    wrapped.shuffle = shuffle
    return wrapped


def de_parallel(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def is_parallel(model: nn.Module) -> bool:
    return isinstance(model, DistributedDataParallel)


def get_world_size():
    return distributed.get_world_size() if is_initialized() else 1


def is_main_process():
    return not is_initialized() or distributed.get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
