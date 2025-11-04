import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from go1_gym.envs.automatic.legged_robot import LeggedRobot

class Rewards:
    def __init__(self, env):
        self.env: LeggedRobot = env

    def load_env(self, env):
        self.env = env

    # arm rewards
    def _reward_arm_vel_control(self):
        linv_vel_error = torch.abs(self.env.plan_actions[:, 2] - self.env.base_lin_vel[:, 0] * self.env.obs_scales.lin_vel)
        ang_vel_error = torch.abs(self.env.plan_actions[:, 3] - self.env.base_ang_vel[:, 2] * self.env.obs_scales.ang_vel)
    
        return linv_vel_error + ang_vel_error
    
    def _reward_arm_orientation_control(self):
        pitch_error = torch.abs(self.env.plan_actions[:, 0] + self.env.pitch)
        base_height_error = torch.abs(self.env.plan_actions[:, 1] + self.env.base_pos[:, 2])
    
        return pitch_error + base_height_error

    def _reward_arm_control_limits(self):
        out_of_limits = -(self.env.plan_actions[:, 0] - self.env.cfg.commands.limit_body_pitch[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.plan_actions[:, 0] - self.env.cfg.commands.limit_body_pitch[1]).clip(min=0.)
        out_of_limits += -(self.env.plan_actions[:, 1] - self.env.cfg.commands.limit_base_height[0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.plan_actions[:, 1] - self.env.cfg.commands.limit_base_height[1]).clip(min=0.)
        return out_of_limits
        
    def _reward_arm_control_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.plan_actions - self.env.last_plan_actions)
        diff = diff * (self.env.last_plan_actions != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_control_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
        
    def _reward_arm_energy(self):
        energy_sum = torch.sum(
            torch.square(self.env.torques[:, self.env.num_actions_loco:]*self.env.dof_vel[:, self.env.num_actions_loco:])
            , dim=1)
        return energy_sum

    def _reward_arm_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel[..., self.env.num_actions_loco:]), dim=1)

    def _reward_arm_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., self.env.num_actions_loco:] / self.env.dt), dim=1)

    def _reward_arm_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., self.env.num_actions_loco:], dim=1)

    def _reward_arm_manip_commands_tracking_combine(self):
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        lpy_error = torch.sum((torch.abs(lpy - self.env.commands_arm_obs[:, 0:3])) / self.env.commands_arm_lpy_range, dim=1)
        
        rpy = self.env.get_alpha_beta_gamma_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        rpy_error = torch.sum((torch.abs(rpy - self.env.target_abg)) / self.env.commands_arm_rpy_range, dim=1)
    
        return torch.exp(-(self.env.cfg.rewards.manip_weight_lpy*lpy_error + self.env.cfg.rewards.manip_weight_rpy*rpy_error))

    def _reward_arm_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:] - self.env.last_joint_pos_target[:, self.env.num_actions_loco:])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_arm_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:] - 2 * self.env.last_joint_pos_target[:, self.env.num_actions_loco:] + self.env.last_last_joint_pos_target[:, self.env.num_actions_loco:])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, self.env.num_actions_loco:] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    # dog rewards
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands_dog[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands_dog[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_loco_energy(self):
        # print("loco energy: ", -0.00005*torch.sum(torch.square(self.env.torques[:, :self.env.num_actions_loco]*self.env.dof_vel[:, :self.env.num_actions_loco]), dim=1)[:20])
        return torch.sum(
            torch.square(self.env.torques[:, :self.env.num_actions_loco]*self.env.dof_vel[:, :self.env.num_actions_loco])
            , dim=1)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel[..., :self.env.num_actions_loco]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., :self.env.num_actions_loco] / self.env.dt), dim=1)

    def _reward_keep_ankle_pitch_zero_in_air(self):
        ankle_pitch = torch.abs(self.env.dof_pos[:, 3]) * ~self.env.contact_filt[:, 0] + torch.abs(
            self.env.dof_pos[:, 7]) * ~self.env.contact_filt[:, 1]
        return torch.exp(-torch.abs(ankle_pitch) / 0.2)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., :self.env.num_actions_loco], dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        if self.env.cfg.reward_scales.tracking_contacts_shaped_force > 0:
            # 奖励模式：返回值越大越好
            for i in range(len(self.env.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                stand_phase = desired_contact[:, i]
                # swing_phase: 希望 foot_forces 小（不接触）→ exp 接近 1 → 奖励大
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma
                )
                # stand_phase: 希望 foot_forces 大（接触）→ exp(-大数) 接近 0，所以用 1 - exp 让大值得到高奖励
                reward += stand_phase * (1 - torch.exp(
                    -foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma
                ))
        else:
            # 惩罚模式：返回值越小越好（因为会被乘以负数，所以返回值越小惩罚越小）
            for i in range(len(self.env.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                stand_phase = desired_contact[:, i]
                # swing_phase: 希望 foot_forces 小 → 当 foot_forces 小时，(1-exp) 接近 0 → 惩罚小（奖励大）
                reward += swing_phase * (
                    1 - torch.exp(
                        -foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma
                    )
                )
                # stand_phase: 希望 foot_forces 大 → 当 foot_forces 大时，exp(-大数) 接近 0 → 惩罚小（奖励大）
                reward += stand_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma
                )
        return reward / len(self.env.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=-1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        if self.env.cfg.reward_scales.tracking_contacts_shaped_vel > 0:
            # 奖励模式：返回值越大越好
            for i in range(len(self.env.feet_indices)):
                stand_phase = desired_contact[:, i]
                swing_phase = 1 - desired_contact[:, i]
                # stand_phase: 希望 foot_velocities 小（脚不动）→ exp 接近 1 → 奖励大
                reward += stand_phase * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma
                )
                # swing_phase: 希望跟踪期望的垂直速度 → exp 接近 1 → 奖励大
                reward += swing_phase * torch.exp(
                    -((self.env.foot_velocities[:, i, 2] - self.env.des_foot_velocity_z) ** 2)
                    / self.env.cfg.rewards.gait_vel_sigma
                )
        else:
            # 惩罚模式：返回值越小越好（因为会被乘以负数）
            for i in range(len(self.env.feet_indices)):
                stand_phase = desired_contact[:, i]
                swing_phase = 1 - desired_contact[:, i]
                # stand_phase: 希望 foot_velocities 小 → 当速度小时，(1-exp) 接近 0 → 惩罚小（奖励大）
                reward += stand_phase * (
                    1 - torch.exp(
                        -foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma
                    )
                )
                # swing_phase: 希望跟踪期望的垂直速度 → 当跟踪好时，(1-exp) 接近 0 → 惩罚小（奖励大）
                reward += swing_phase * (1 - torch.exp(
                    -((self.env.foot_velocities[:, i, 2] - self.env.des_foot_velocity_z) ** 2)
                    / self.env.cfg.rewards.gait_vel_sigma)
                )
        return reward / len(self.env.feet_indices)

    def _reward_tracking_contacts_shaped_height(self):
        foot_heights = self.env.foot_heights
        desired_contact = self.env.desired_contact_states
        reward = 0
        if self.env.cfg.reward_scales.tracking_contacts_shaped_height > 0:
            # 奖励模式：返回值越大越好
            for i in range(len(self.env.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                stand_phase = desired_contact[:, i]
                # swing_phase: 希望跟踪期望高度 → exp 接近 1 → 奖励大
                reward += swing_phase * torch.exp(
                    -(foot_heights[:, i] - self.env.des_foot_height) ** 2 / self.env.cfg.rewards.gait_height_sigma
                )
                # stand_phase: 希望 foot_heights 接近 0（脚贴地）→ exp 接近 1 → 奖励大
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.env.cfg.rewards.gait_height_sigma)
        else:
            # 惩罚模式：返回值越小越好（因为会被乘以负数）
            for i in range(len(self.env.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                stand_phase = desired_contact[:, i]
                # swing_phase: 希望跟踪期望高度 → 当跟踪好时，(1-exp) 接近 0 → 惩罚小（奖励大）
                reward += swing_phase * (
                    1 - torch.exp(-(foot_heights[:, i] - self.env.des_foot_height) ** 2 / self.env.cfg.rewards.gait_height_sigma)
                )
                # stand_phase: 希望 foot_heights 接近 0 → 当接近 0 时，(1-exp) 接近 0 → 惩罚小（奖励大）
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.env.cfg.rewards.gait_height_sigma))
        return reward / len(self.env.feet_indices)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - self.env.last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actions_loco] + self.env.last_last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_actions_loco] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                self.env.contact_forces[:, self.env.feet_indices, 2]
                - self.env.base_mass.mean() * 9.8 / 2
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_feet_distance(self):
        # Penalize feet distance smaller than minimum required distance
        feet_distance = torch.norm(
            self.env.foot_positions[:, 0, :2] - self.env.foot_positions[:, 1, :2], dim=-1
        )
        return torch.clip(self.env.cfg.rewards.min_feet_distance - feet_distance, 0, 1)

    def _reward_feet_regulation(self):
        feet_height = self.env.cfg.rewards.base_height_target * 0.025
        reward = torch.sum(
            torch.exp(-self.env.foot_heights / feet_height)
            * torch.square(torch.norm(self.env.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        return reward

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_orientation_heuristic(self):
        guide = torch.zeros_like(self.env.pitch)
        # Adjusted thresholds for base height 0.8m (was 0.38m)
        # Base height increased ~2.1x, so thresholds are scaled accordingly
        down_flag = self.env.delta_z < -self.env.cfg.hybrid.rewards.headupdown_thres
        up_flag = self.env.delta_z > self.env.cfg.hybrid.rewards.headupdown_thres+0.1
        # Adjusted pitch targets: for taller base, may need larger pitch angles
        # Original: pitch_down=0.4, pitch_up=-0.3
        # For 0.8m base: keeping same angles or can adjust based on task requirements
        guide[down_flag] = torch.square(self.env.pitch - 0.3)[down_flag]  # pitch down to 0.3 rad (~23°)
        guide[up_flag] = torch.square(self.env.pitch + 0.1)[up_flag]  # pitch up to -0.1 rad (~17°)
        
        return guide

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        # roll command is 0, only pitch command exists
        pitch_command = self.env.commands_dog[:, 3]  # body pitch command
        # print(roll_pitch_commands)
        quat_roll = quat_from_angle_axis(torch.zeros_like(pitch_command),  # roll is always 0
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-pitch_command,
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_base_height_control(self):
        base_height_command = self.env.commands_dog[:, 4]
        return torch.square(self.env.base_pos[:, 2] - base_height_command)
    
    # vis
    def _reward_vis_manip_commands_tracking_lpy(self):
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        lpy_error = torch.sum((torch.abs(lpy - self.env.commands_arm_obs[:, 0:3])) / self.env.commands_arm_lpy_range, dim=1)
        return torch.exp(-lpy_error)

    def _reward_vis_manip_commands_tracking_rpy(self):
        rpy = self.env.get_alpha_beta_gamma_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        rpy_error = torch.sum((torch.abs(rpy - self.env.commands_arm_obs[:, 3:6])) / self.env.commands_arm_rpy_range, dim=1)
        return torch.exp(-rpy_error)