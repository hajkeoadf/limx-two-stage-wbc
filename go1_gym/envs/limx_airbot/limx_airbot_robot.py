# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from loco_manipulation_gym import LOCO_MANI_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import sys
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from loco_manipulation_gym.envs import LeggedRobot
from loco_manipulation_gym import LOCO_MANI_GYM_ROOT_DIR


from loco_manipulation_gym.envs.base.base_task import BaseTask
from loco_manipulation_gym.utils.terrain import Terrain
from loco_manipulation_gym.utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift
from loco_manipulation_gym.utils.helpers import class_to_dict
from .limx_airbot_config import LimxAirbotRoughCfg

DEFAULT_BASE_HEIGHT = 0.5

class LimxAirbot(LeggedRobot):
    cfg:LimxAirbotRoughCfg

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques,self.arm_pos = self._compute_torques(self.actions)
            self._push_gripper(torch.arange(self.num_envs, device=self.device), self.cfg)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_pos))
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        self.update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer :
            self._draw_ee_goal_track()
            self._draw_debug_vis()

    def reset(self):
        """ Reset all robots"""
        self.update_curr_ee_goal()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        force_control_envs = env_ids[self.force_or_position_control[env_ids] == 1]
        self.forces[force_control_envs, self.gripper_stator_index, :3] = 0.
        self.selected_env_ids[force_control_envs] = 0
        self.push_end_time[force_control_envs] = 0.
        self.force_target[force_control_envs, :3] = 0.
        self.push_duration[force_control_envs] = 0.
        self.current_Fxyz_cmd[force_control_envs, :3] = 0.
        # 重置独立的力目标
        self.curr_force_goal[force_control_envs, :3] = 0.0

        # reset push gripper 
        if self.cfg.domain_rand.push_gripper_stators:
            self.forces[env_ids, self.gripper_stator_index, :3] = 0.
            self.selected_env_ids[env_ids] = 0
            self.push_end_time[env_ids] = 0.
            self.push_force[env_ids, :3] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _push_gripper(self, env_ids_all, cfg):
        """ Randomly pushes the gripper stators. Emulates an impulse by setting a randomized gripper stator velocity.
        """

        force_control_envs = self.force_or_position_control == 1
        position_control_envs = self.force_or_position_control == 0
        
        # FORCE CONTROLLED ENVS
        new_selected_env_ids = env_ids_all[(self.episode_length_buf % self.push_interval[:, 0]) == 0]
        
        # print('newselect', new_selected_env_ids)
        # self.push_force[force_control_envs] = 0.0
        
        # Define force and duration for the push 
        if new_selected_env_ids.nelement() > 0:
            self.freed_envs[new_selected_env_ids] = torch.rand(len(new_selected_env_ids), dtype=torch.float, device=self.device,
                                        requires_grad=False) > self.cfg.domain_rand.gripper_forced_prob
            min_force = cfg.domain_rand.max_push_force_xyz_gripper[0]
            max_force = cfg.domain_rand.max_push_force_xyz_gripper[1]

            self.force_target[new_selected_env_ids, 0] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            self.force_target[new_selected_env_ids, 1] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            self.force_target[new_selected_env_ids, 2] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            push_duration = torch_rand_float(cfg.domain_rand.push_duration_gripper_min, cfg.domain_rand.push_duration_gripper_max, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids)) # 4.0/self.dt
            self.push_end_time[new_selected_env_ids] = self.episode_length_buf[new_selected_env_ids] + push_duration
            self.push_duration[new_selected_env_ids] = push_duration

            self.selected_env_ids[new_selected_env_ids] = 1
            
        # Get ids of all envs to apply a force to 
        if self.episode_length_buf[self.selected_env_ids == 1].nelement() > 0:
            subset_env_ids_selected = env_ids_all[self.selected_env_ids == 1]

            # Step 1: apply force from 0 to force_target
            env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids == 1] < (self.push_end_time[self.selected_env_ids == 1]).type(torch.int32)]
            # print(env_ids_apply_push_step1)
            if env_ids_apply_push_step1.nelement() > 0:
                push_duration_reshaped = self.push_duration[env_ids_apply_push_step1].unsqueeze(-1)
                # world frame
                self.current_Fxyz_cmd[env_ids_apply_push_step1, :3] = (self.force_target[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                # 将力命令设置到独立的curr_force_goal中，而不是commands
                self.curr_force_goal[env_ids_apply_push_step1, 0] = self.current_Fxyz_cmd[env_ids_apply_push_step1, 0]
                self.curr_force_goal[env_ids_apply_push_step1, 1] = self.current_Fxyz_cmd[env_ids_apply_push_step1, 1]
                self.curr_force_goal[env_ids_apply_push_step1, 2] = self.current_Fxyz_cmd[env_ids_apply_push_step1, 2]
            
                # print(self.curr_force_goal[env_ids_apply_push_step1, :3])
                
            # Step 2: apply force from force_target back to 0
            env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids == 1] > (self.push_end_time[self.selected_env_ids == 1] + self.cfg.commands.settling_nb_it_force_gripper).type(torch.int32)]
            if env_ids_apply_push_step2.nelement() > 0:
                push_duration_reshaped = self.push_duration[env_ids_apply_push_step2].unsqueeze(-1)
                self.current_Fxyz_cmd[env_ids_apply_push_step2, :3] = self.force_target[env_ids_apply_push_step2, :3] - (self.force_target[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time[env_ids_apply_push_step2].unsqueeze(-1)+self.cfg.commands.settling_nb_it_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                # print("current target: ", self.current_Fxyz_cmd[env_ids_apply_push_step2, :3])
            
                # 将力命令设置到独立的curr_force_goal中，而不是commands
                self.curr_force_goal[env_ids_apply_push_step2, 0] = self.current_Fxyz_cmd[env_ids_apply_push_step2, 0]
                self.curr_force_goal[env_ids_apply_push_step2, 1] = self.current_Fxyz_cmd[env_ids_apply_push_step2, 1]
                self.curr_force_goal[env_ids_apply_push_step2, 2] = self.current_Fxyz_cmd[env_ids_apply_push_step2, 2]
                
                # print(self.curr_force_goal[env_ids_apply_push_step2, :3])
            
            # Reset the tensors
            env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids == 1] >= (2*self.push_end_time[self.selected_env_ids == 1] + self.cfg.commands.settling_nb_it_force_gripper).type(torch.int32)]                                        
            if env_ids_to_reset.nelement() > 0:
                self.selected_env_ids[env_ids_to_reset] = 0
                self.force_target[env_ids_to_reset, :3] = 0.
                self.current_Fxyz_cmd[env_ids_to_reset, :3] = 0.
                self.push_end_time[env_ids_to_reset] = 0.
                self.push_duration[env_ids_to_reset] = 0.
                # 重置独立的力目标
                self.curr_force_goal[env_ids_to_reset, :3] = 0.0

                self.push_interval[env_ids_to_reset, 0] = torch.randint(int(cfg.domain_rand.push_interval_gripper_min), int(cfg.domain_rand.push_interval_gripper_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                
        self.selected_env_ids[self.freed_envs] = 0
        self.force_target[self.freed_envs, :3] = 0.
        self.current_Fxyz_cmd[self.freed_envs, :3] = 0.
        self.push_end_time[self.freed_envs] = 0.
        self.push_duration[self.freed_envs] = 0. 
        # 重置独立的力目标
        self.curr_force_goal[self.freed_envs, :3] = 0.0

        # Cartesian coordinates in the base frame
        ee_position_cmd_base = self.curr_ee_goal_cart.view(self.num_envs, 3)

        # Commands in world frame
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.base_pos[:, 0].view(self.num_envs, 1) 
        y_base_pos_world = self.base_pos[:, 1].view(self.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.base_pos[:, 2].view(self.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        self.ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world

        # Get current ee position in world frame 
        ee_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0], "gripperStator")
        self.ee_pos_world = self.rigid_body_state.view(self.num_envs, -1, 13)[:,ee_idx,0:3].view(self.num_envs,3)

        # ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)
        # self.forces[:, self.gripper_stator_index, :3] = torch.clamp(((self.ee_position_cmd_world - self.ee_pos_world) * self.cfg.domain_rand.gripper_force_kp*3 +  (0 - self.gripper_velocity) * self.cfg.domain_rand.gripper_force_kd), self.cfg.domain_rand.max_push_force_xyz_gripper_freed[0], self.cfg.domain_rand.max_push_force_xyz_gripper_freed[1])
        self.forces[:, self.gripper_stator_index, :3] = torch.clamp(((self.ee_position_cmd_world - self.ee_pos_world) * self.gripper_force_kps +  (0 - self.gripper_velocity) * self.gripper_force_kds), 
                                                        self.cfg.domain_rand.max_push_force_xyz_gripper_freed[0], 
                                                        self.cfg.domain_rand.max_push_force_xyz_gripper_freed[1])
        self.forces[self.freed_envs, self.gripper_stator_index, :3] = 0
        
        # don't force the position envs
        self.forces[position_control_envs, self.gripper_stator_index, :3] = 0
        self.curr_force_goal[position_control_envs, :3] = 0.0
        
        # compute the error
        force_magn_meas = torch.norm(self.forces[:, self.gripper_stator_index, :2], dim=1).view(self.num_envs)
        force_magn_target = torch.norm(self.curr_force_goal[:, :2], dim=1).view(self.num_envs) 
        force_magn_error_xy = torch.abs(force_magn_meas - force_magn_target).view(self.num_envs)

        force_magn_meas = (self.forces[:, self.gripper_stator_index, 2]).view(self.num_envs, 1)
        force_magn_cmd = (self.curr_force_goal[:, 2]).view(self.num_envs, 1)
        force_magn_error_z = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        self.gripper_xy_force_tracking_error_buf = force_magn_error_xy
        self.gripper_z_force_tracking_error_buf = force_magn_error_z
        
        if self.forces_deactivated:
            self.forces[:, self.gripper_stator_index, 0] = self.Fx
            self.forces[:, self.gripper_stator_index, 1] = self.Fy
            self.forces[:, self.gripper_stator_index, 2] = self.Fz

        # print(self.commands[:, 12:15])
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations for biped robot with arm
        """
        self.dof_err = self.dof_pos - self.default_dof_pos
        # 双足机器人不需要将车轮位置置零，因为没有车轮
        
        self.obs_buf = torch.cat((  
            self.base_lin_vel * self.obs_scales.lin_vel,                     # 3维: 基座线速度
            self.base_ang_vel * self.obs_scales.ang_vel,                     # 3维: 基座角速度  
            self.projected_gravity,                                          # 3维: 重力投影
            self.commands[:, :3] * self.commands_scale,                      # 3维: 运动指令
            self.dof_err * self.obs_scales.dof_pos,                         # 14维: 关节位置误差(双足8+机械臂6)
            self.dof_vel * self.obs_scales.dof_vel,                         # 14维: 关节速度(双足8+机械臂6)
            self._local_gripper_pos * self.obs_scales.gripper_track,        # 3维: 本地末端执行器位置
            self.curr_ee_goal_cart * self.obs_scales.gripper_track,          # 3维: 当前目标位置
            (self._local_gripper_pos - self.curr_ee_goal_cart) * self.obs_scales.gripper_track,  # 3维: 位置误差
            self.actions                                                     # 14维: 上一步动作
        ), dim=-1)  # 总计: 3+3+3+3+14+14+3+3+3+14 = 63维 (无夹爪)

        # 计算特权观测 (privileged observations)
        if self.privileged_obs_buf is not None:
            # 获取地面摩擦系数 (1维)
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            friction_coeff = (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale
            
            # 获取末端执行器力反馈 (3维) 
            forces_global = self.forces[:, self.gripper_stator_index, :3]
            base_quat_world = self.base_quat.view(self.num_envs,4)
            base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
            base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.num_envs,4)
            forces_local = quat_rotate_inverse(base_quat_world_indep, forces_global)
            
            # 组合特权观测: 摩擦(1) + 末端力(3) = 4维
            privileged_obs = torch.cat([
                friction_coeff,           # 1维: 地面摩擦系数
                forces_local,             # 3维: 末端执行器力反馈
            ], dim=-1)  # 总计: 1+3 = 4维
            
            # 添加高度测量 (如果启用)
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                # 特权观测 = 基础观测 + 特权信息 + 高度测量
                self.privileged_obs_buf = torch.cat((self.obs_buf, privileged_obs, heights), dim=-1)
            else:
                # 特权观测 = 基础观测 + 特权信息
                self.privileged_obs_buf = torch.cat((self.obs_buf, privileged_obs), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if self.cfg.control.arm_control_type == "position":
            props["driveMode"][self.arm_indices.cpu()].fill(gymapi.DOF_MODE_POS)
            props["stiffness"][self.arm_indices.cpu()].fill(self.cfg.control.arm_stiffness)
            props["damping"][self.arm_indices.cpu()].fill(self.cfg.control.arm_stiffness)
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _resample_force_or_position_control(self, env_ids):
        if self.cfg.commands.hybrid_mode == "mixed":
            self.force_or_position_control[env_ids] = torch.rand(len(env_ids), device=self.device)
        elif self.cfg.commands.hybrid_mode == "binary":
            self.force_or_position_control[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device).float()
        elif self.cfg.commands.hybrid_mode == "force":
            self.force_or_position_control[env_ids] = torch.ones(len(env_ids), device=self.device)
        elif self.cfg.commands.hybrid_mode == "position":
            self.force_or_position_control[env_ids] = torch.zeros(len(env_ids), device=self.device)
        else:
            print("Error: hybrid_mode not recognized")
            exit(0)
        self.commands[env_ids, INDEX_FORCE_OR_POSITION_INDICATOR] = self.force_or_position_control[env_ids]

    def orientation_error(self,desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
       
    def control_ik(self,local_ee_pose,local_goal_pose,local_j_eef):
        pos_err = local_goal_pose[:,0:3] - local_ee_pose[:,0:3]
        orn = torch.tensor([0,0,0,1], device=self.device).repeat(self.num_envs, 1)
        orn_err = self.orientation_error(orn, orn)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        # solve damped least squares
        j_eef_T = torch.transpose(local_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        u = (j_eef_T @ torch.inverse(local_j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        # 双足机器人控制逻辑
        self.dof_err = self.dof_pos - self.default_dof_pos
        # 双足机器人不需要将车轮误差置零
        
        # 机械臂IK控制 
        ik_u = self.control_ik(self._local_gripper_pos, self.curr_ee_goal, self.j_eef)
        self.arm_u[:, self.arm_indices] = self.dof_pos[:, self.arm_indices] + actions[:, self.arm_indices] + ik_u

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        modify_dof_vel = self.dof_vel.clone().detach()
        modify_dof_vel[:, self.arm_indices] = 0  # 机械臂速度置零用于位置控制
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled - self.dof_err) - self.d_gains*modify_dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled - self.dof_err) - self.motor_strength[1] * self.d_gains*modify_dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - modify_dof_vel) - self.d_gains*(modify_dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits),self.arm_u

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.forces = torch.zeros(self.num_envs, self.total_rigid_body_num, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Push gripper 
        self.selected_env_ids = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval = torch.randint(int(self.cfg.domain_rand.push_interval_gripper_min), int(self.cfg.domain_rand.push_interval_gripper_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_forces_eval_time = torch.zeros(1, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_force = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 独立的力目标变量
        self.curr_force_goal = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.init_ee_goal_variale()

        self.force_or_position_control = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._resample_force_or_position_control(torch.arange(self.num_envs, device=self.device))

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # random motor lenth
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LOCO_MANI_GYM_ROOT_DIR=LOCO_MANI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 机械臂关节（仅包含J1-J6）
        arm_names = []
        for name in self.cfg.asset.arm_joint_name:
            # 只匹配以J开头的关节，且不包含"_Joint"后缀
            for dof_name in self.dof_names:
                if dof_name.startswith(name) and "_Joint" not in dof_name:
                    arm_names.append(dof_name)
        # 双足机器人不需要车轮，但需要识别踝关节作为足部
        ankle_names = []
        for name in ["ankle"]:
            ankle_names.extend([s for s in self.dof_names if name in s])

        # 双足关节（包含所有腿部关节）
        biped_names = []
        for name in self.cfg.asset.biped_joint_name:
            biped_names.extend([s for s in self.dof_names if name in s])

        print("###self.rigid_body names:", body_names)
        print("###self.dof names:", self.dof_names)
        print("###penalized_contact_names:", penalized_contact_names)
        print("###termination_contact_names:", termination_contact_names)
        print("###feet_names:", feet_names)
        print("###ankle_names:", ankle_names)
        print("###arm_names:", arm_names)
        print("###biped_names:", biped_names)
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for envs_idx in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[envs_idx].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, envs_idx)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, envs_idx, self.cfg.asset.self_collisions, 0)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if envs_idx==0:
                self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(feet_names)):
                    self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

                self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(penalized_contact_names)):
                    self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

                self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(termination_contact_names)):
                    self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

                # 踝关节索引（双足机器人的足部）
                self.ankle_indices = torch.zeros(len(ankle_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(ankle_names)):
                    self.ankle_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], ankle_names[i])
                
                # 机械臂关节索引
                self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(arm_names)):
                    self.arm_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], arm_names[i])

                # 双足关节索引（所有腿部关节）
                self.biped_indices = torch.zeros(len(biped_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(biped_names)):
                    self.biped_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], biped_names[i])

                
                print("###self.ankle_indices:", self.ankle_indices)
                print("###self.arm_indices:", self.arm_indices)
                print("###self.biped_indices:", self.biped_indices)

            dof_props = self._process_dof_props(dof_props_asset, envs_idx)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, envs_idx)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # def _draw_debug_vis(self):
    #     """ Draws visualizations for dubugging (slows down simulation a lot).
    #         Default behaviour: draws height measurement points
    #     """
    #     # draw height lines
    #     if not self.terrain.cfg.measure_heights:
    #         return
    #     self.gym.clear_lines(self.viewer)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
    #     for i in range(self.num_envs):
    #         base_pos = (self.root_states[i, :3]).cpu().numpy()
    #         heights = self.measured_heights[i].cpu().numpy()
    #         height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
    #         for j in range(heights.shape[0]):
    #             x = height_points[j, 0] + base_pos[0]
    #             y = height_points[j, 1] + base_pos[1]
    #             z = heights[j]
    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

# ee goal function
    #----------------------------------------
    def init_ee_goal_variale(self):

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.asset.arm_gripper_name)
        robot_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)

        self.hand_index = robot_link_dict[self.cfg.asset.arm_gripper_name]
        # create some wrapper tensors for different slices
        self.whole_body_jacobian = gymtorch.wrap_tensor(actor_jacobian)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        self.j_eef = self.whole_body_jacobian[:,self.hand_index, :6, self.arm_indices+6]

        #
        self.ee_pos = self.rigid_body_states[:, self.hand_index, :3]
        self.ee_orn = self.rigid_body_states[:, self.hand_index, 3:7]
        self.ee_vel = self.rigid_body_states[:, self.hand_index, 7:]
        self.ee_j_eef = self.whole_body_jacobian[:, self.hand_index, :6, self.arm_indices+6]

        # 机械臂关节（仅包含J1-J6）
        arm_names = []
        for name in self.cfg.asset.arm_joint_name:
            # 只匹配以J开头的关节，且不包含"_Joint"后缀
            for dof_name in self.dof_names:
                if dof_name.startswith(name) and "_Joint" not in dof_name:
                    arm_names.append(dof_name)

        self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(arm_names)):
            self.arm_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], arm_names[i])
        self.arm_u = torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device)

        print("###arm_names:",arm_names)
        print("###arm_indices:",self.arm_indices)
        print("###hand_index:",self.hand_index)
        print("###gripperMover_handles (末端执行器):",self.gripperMover_handles)

        #update variable
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_delta_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
        self.init_goal_ee_l_ranges = self.goal_ee_l_ranges = np.array(self.goal_ee_ranges['init_pos_l'])
        self.init_goal_ee_p_ranges = self.goal_ee_p_ranges = np.array(self.goal_ee_ranges['init_pos_p'])
        self.init_goal_ee_y_ranges = self.goal_ee_y_ranges = np.array(self.goal_ee_ranges['init_pos_y'])
        self.goal_ee_delta_orn_ranges = torch.tensor(self.goal_ee_ranges['final_delta_orn'])
        self._local_cube_object_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)
        self._cube_object_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)
        
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])

        if self.cfg.goal_ee.command_mode == 'cart':
            self.curr_ee_goal = self.curr_ee_goal_cart
        else:
            self.curr_ee_goal = self.curr_ee_goal_sphere

        local_axis_z_offset = self.cfg.goal_ee.local_axis_z_offset
        self.local_axis_z = torch.tensor(local_axis_z_offset, device=self.device).repeat(self.num_envs, 1)
        self.z_invariant_offset = torch.tensor(local_axis_z_offset,device=self.device).repeat(self.num_envs, 1)  
        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_yaw_eular = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)


        self.base_align_z_axis = torch.tensor([0.,0.,local_axis_z_offset],dtype=torch.float,device=self.device).repeat(self.num_envs,1)
        # 末端执行器状态 (原夹爪位置，现在是link6)
        self._gripper_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13] 
        self._gripper_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        self._gripper_quat = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]
        self._local_gripper_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)
        
    def refresh_ee_goal_variable(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_align_z_axis = torch.cat([self.root_states[:, :2], self.local_axis_z], dim=1)

        self.base_quat = self.root_states[:, 3:7]
        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_fixed = wrap_to_pi(base_yaw).view(self.num_envs,1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_yaw_eular = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        # 计算末端执行器的本地位置 (原来是夹爪位置)
        self._local_gripper_pos = quat_rotate_inverse(self.base_yaw_quat,self._gripper_pos - self.base_align_z_axis) 
        self.base_to_obj_dist = self._cube_object_pos[:,:2] - self.root_states[:,:2]
        self.base_align_z_axis[:,:2] = self.root_states[:, :2]

    def cart2sphere(self,cart):
        sphere = torch.zeros_like(cart)
        sphere[:, 0] = torch.norm(cart, dim=-1)
        sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])
        sphere[:, 2] = torch.asin(cart[:, 1] / sphere[:, 0])
        return sphere

    def sphere2cart(self,sphere):
        cart = torch.zeros_like(sphere)
        cart[:, 0] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.cos(sphere[:, 1])
        cart[:, 1] = sphere[:, 0] * torch.sin(sphere[:, 2])
        cart[:, 2] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.sin(sphere[:, 1])
        return cart

    def update_curr_ee_goal(self):
        self.refresh_ee_goal_variable()
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = self.sphere2cart(self.curr_ee_goal_sphere)
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)

        #update current ee goal global axis
        self._cube_object_pos = self.base_align_z_axis + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = self.sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def _get_init_start_ee_sphere(self):
        init_start_ee_cart = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        init_start_ee_cart[:, 0] = 0.15
        init_start_ee_cart[:, 2] = 0.15
        self.init_start_ee_sphere = self.cart2sphere(init_start_ee_cart)

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_l_ranges[0], self.goal_ee_l_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_p_ranges[0], self.goal_ee_p_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_y_ranges[0], self.goal_ee_y_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_delta_orn_ranges[0, 0], self.goal_ee_delta_orn_ranges[0, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_delta_orn_ranges[1, 0], self.goal_ee_delta_orn_ranges[1, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_delta_orn_ranges[2, 0], self.goal_ee_delta_orn_ranges[2, 1], (len(env_ids), 1), device=self.device)
        self.ee_goal_delta_orn_euler[env_ids] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)
        self.ee_goal_orn_euler[env_ids] = wrap_to_pi(self.ee_goal_delta_orn_euler[env_ids] + self.base_yaw_eular[env_ids])
        # self  
     
    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            self._resample_force_or_position_control(env_ids)
            self._resample_ee_goal_orn_once(env_ids)
            # if is_init:
            #     self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[env_ids].clone()
            #     self._resample_ee_goal_sphere_once(env_ids, self.cfg.goal_ee.init_ranges)
            # else:
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for i in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self.collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
            self.ee_goal_cart[init_env_ids, :] = self.sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0
            self.traj_timesteps[init_env_ids] = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (len(init_env_ids), 1), device=self.device).squeeze() / self.dt

    def _draw_ee_goal_track(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze()
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = self.sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        ee_target_all_cart_world += torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)[:, :, None]
        # curr_ee_goal_cart_world = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart) + self.root_states[:, :3]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        transformed_target_ee = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
        upper_arm_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        
        ee_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self._local_gripper_pos)
        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)
        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(transformed_target_ee[i, 0], transformed_target_ee[i, 1], transformed_target_ee[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_err = self.dof_pos - self.default_dof_pos
        # 双足机器人不需要将车轮误差置零
        dof_err[:, self.arm_indices] = 0  # 只将机械臂误差置零
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_joint_pos_rate(self):
        # Penalize joint position rate for biped robot
        dof_err = self.last_dof_pos - self.default_dof_pos
        # 双足机器人不需要将车轮误差置零
        return torch.sum(torch.square(dof_err), dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_object_distance(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""
        dis_err = torch.sum(torch.square(self._local_gripper_pos-self.curr_ee_goal), dim=1)
        dis_err = dis_err * (1 - self.force_or_position_control)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return torch.exp(-dis_err/0.1)
    
    def _reward_object_distance_l2(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""
        dis_err = torch.sum(torch.square(self._local_gripper_pos-self.curr_ee_goal), dim=1)
        dis_err = dis_err * (1 - self.force_or_position_control)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return dis_err

    ########################
    # force tracking rewards
    ########################
    
    def _reward_ee_force_x(self):
        
        xy_forces_global = self.forces[:, self.gripper_stator_index, 0:3]
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        xy_forces_local = quat_rotate_inverse(base_quat_world_indep, xy_forces_global)
        
        force_magn_meas = (xy_forces_local[:, 0]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_X]).view(self.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        force_magn_coeff = self.cfg.rewards.sigma_force_z
        return torch.exp(-force_magn_coeff*force_magn_error) * self.force_or_position_control
    
    def _reward_ee_force_y(self):
        
        xy_forces_global = self.forces[:, self.gripper_stator_index, 0:3]
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        xy_forces_local = quat_rotate_inverse(base_quat_world_indep, xy_forces_global)

        force_magn_meas = (xy_forces_local[:, 1]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_Y]).view(self.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        force_magn_coeff = self.cfg.rewards.sigma_force_z
        return torch.exp(-force_magn_coeff*force_magn_error) * self.force_or_position_control
    
    def _reward_ee_force_z(self):

        force_magn_meas = (self.forces[:, self.gripper_stator_index, 2]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_Z]).view(self.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        force_magn_coeff = self.cfg.rewards.sigma_force_z
        return torch.exp(-force_magn_coeff*force_magn_error) * self.force_or_position_control

    def _reward_ee_force_magnitude_x_pen(self):
        
        force_magn_meas = torch.abs(self.forces[:, self.gripper_stator_index, 0]).view(self.num_envs, 1)
        force_magn_cmd = 0.0 
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        return force_magn_error  * self.force_or_position_control
    
    def _reward_ee_force_magnitude_y_pen(self):
       
        force_magn_meas = torch.abs(self.forces[:, self.gripper_stator_index, 1]).view(self.num_envs, 1)
        force_magn_cmd = 0.0 
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)

        return force_magn_error  * self.force_or_position_control
        