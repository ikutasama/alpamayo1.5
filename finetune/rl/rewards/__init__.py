# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward helpers for Alpamayo RL.

Submodules are intentionally not imported eagerly to keep reward workers and
smoke tests from loading optional heavy dependencies unless the specific helper
is used.
"""
