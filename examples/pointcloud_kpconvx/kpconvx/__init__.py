# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Vendored from apple/ml-kpconvx (commit 54e644a9) — Standalone/KPConvX subtree.
# See ./LICENSE and ./SOURCE.sha. Only package-relative import rewrites and the
# `load_kernels` kernel-dir / `init_gpu` stub adaptations were made; the operator
# math is unchanged.
"""KPConvX operator package (vendored from apple/ml-kpconvx, Standalone flavor).

Provides the KPConv / KPConvBlock / KPConvResidualBlock operators and a
``KPConvXEncoder`` classification-style encoder wrapping them with a pure-torch
neighborhood pyramid (no pykeops / C++ extensions required).
"""
from .encoder import KPConvXEncoder

__all__ = ["KPConvXEncoder"]
