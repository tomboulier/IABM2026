# -------------------------------------------------------------------------
# Copyright 2025 Thomas Boulier
#
# This file contains code derived from the implementation in:
# “Generative Deep Learning, 2nd Edition” by David Foster (O’Reilly).
# Original source code (Apache License 2.0) available at:
# https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
#
# Modifications:
# - Adapted to fit within the specific project structure.
# - Changed function and variable names to match project conventions.
# - Added type hints for better code clarity.
# - Updated documentation to align with project standards.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# -------------------------------------------------------------------------

from __future__ import annotations

import tensorflow as tf


def offset_cosine_diffusion_schedule(diffusion_times: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute noise and signal rates using an offset cosine schedule.

    Parameters
    ----------
    diffusion_times : tf.Tensor
        Diffusion timesteps in range [0, 1], where 0 is the start (clean images)
        and 1 is the end (pure noise). Shape: (batch_size, 1, 1, 1)

    Returns
    -------
    noise_rates : tf.Tensor
        Noise level at each timestep (sine of diffusion angle).
    signal_rates : tf.Tensor
        Signal level at each timestep (cosine of diffusion angle).
    """
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates
