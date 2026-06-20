# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# KPConvX project: gpu_init.py  (adapted stub)
# Original: apple/ml-kpconvx Standalone/KPConvX/utils/gpu_init.py
#
# Adaptation: the original init_gpu() picked a free CUDA device via pynvml and
# returned torch.device("cuda:N"). eb_jepa manages devices itself, so we stub it
# to return the current default CUDA device (or CPU when CUDA is unavailable).
import torch


def init_gpu(gpu_id="0"):
    """Return the active torch device (stub of the original GPU picker)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_MB(a):
    return round(a.element_size() * a.nelement() / 1024 / 1024, 2)
