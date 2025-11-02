# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from typing import Tuple

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, normalize
from torch import Tensor


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


def get_scale_shift(range):
    scale = 2. / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.
    return scale, shift


class CubicSpline:
    def __init__(self, start, end):
        self.t0 = start['time']
        self.t1 = end['time']
        self.dt = end['time'] - start['time']

        dp = end['position'] - start['position']
        dv = end['velocity'] - start['velocity']

        self.dc0 = torch.tensor(0.0)
        self.dc1 = start['velocity']
        self.dc2 = -(3.0 * start['velocity'] + dv)
        self.dc3 = (2.0 * start['velocity'] + dv)

        self.c0 = self.dc0 * self.dt + start['position']
        self.c1 = self.dc1 * self.dt
        self.c2 = self.dc2 * self.dt + 3.0 * dp
        self.c3 = self.dc3 * self.dt - 2.0 * dp

    def position(self, time):
        tn = self.normalized_time(time)
        return self.c3 * tn ** 3 + self.c2 * tn ** 2 + self.c1 * tn + self.c0

    def velocity(self, time):
        tn = self.normalized_time(time)
        return (3.0 * self.c3 * tn ** 2 + 2.0 * self.c2 * tn + self.c1) / self.dt

    def acceleration(self, time):
        tn = self.normalized_time(time)
        return (6.0 * self.c3 * tn + 2.0 * self.c2) / (self.dt ** 2)

    def start_time_derivative(self, t):
        tn = self.normalized_time(t)
        dCoff = -(self.dc3 * tn ** 3 + self.dc2 * tn ** 2 + self.dc1 * tn + self.dc0)
        dTn = -(self.t1 - t) / (self.dt ** 2)
        return self.velocity(t) * self.dt * dTn + dCoff

    def final_time_derivative(self, t):
        tn = self.normalized_time(t)
        dCoff = (self.dc3 * tn ** 3 + self.dc2 * tn ** 2 + self.dc1 * tn + self.dc0)
        dTn = -(t - self.t0) / (self.dt ** 2)
        return self.velocity(t) * self.dt * dTn + dCoff

    def normalized_time(self, t):
        return (t - self.t0) / self.dt