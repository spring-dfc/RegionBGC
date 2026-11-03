"""RepVGG reparameterization shared by training saves and deployment tools."""

import copy

import torch
from torch import nn

from .encoder import RepVGGBlock


FORMAT = 'regionbgc-repvgg-deploy-v3'
DEPLOY_FORMATS = (FORMAT,)


def compact_structure(model, fuse=False):
    """Fuse or reconstruct RepVGG convolutions and remove training branches."""
    model.eval()
    count = 0
    for block in list(model.modules()):
        if not isinstance(block, RepVGGBlock):
            continue
        if not hasattr(block, 'conv1'):
            continue
        if fuse:
            block.convert_to_deploy()
        else:
            weight = block.conv1.conv.weight
            block.conv = nn.Conv1d(
                block.ch_in, block.ch_out, 3, padding=1
            ).to(device=weight.device, dtype=weight.dtype)
        del block.conv1
        del block.conv2
        count += 1
    model.eval()
    return count


@torch.no_grad()
def build_deploy_checkpoint(model, yaml_cfg, last_epoch, weights_source='model'):
    """Build CPU FP32 inference weights without modifying training state or RNG.

    ``model`` must be the unwrapped model/EMA branch used for evaluation.
    The original module, optimizer parameter identities, and BN statistics stay
    intact so training can continue immediately after any checkpoint save.
    """
    with torch.random.fork_rng(devices=[]):
        deployed = copy.deepcopy(model).cpu().float().eval()
        count = compact_structure(deployed, fuse=True)
    artifact = {
        'format': FORMAT,
        'model': deployed.state_dict(),
        'yaml_cfg': copy.deepcopy(yaml_cfg),
        'last_epoch': last_epoch,
        'feature_layout': 'BCL',
        'metadata': {
            'weights_source': weights_source,
            'fused_blocks': count,
            'device': 'cpu',
            'dtype': 'float32',
        },
    }
    representation = getattr(model, 'interval_representation', None)
    if representation is not None:
        artifact['interval_representation'] = representation
    return artifact
