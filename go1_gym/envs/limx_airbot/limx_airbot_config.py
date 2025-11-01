
import sys
import os
from loco_manipulation_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
class LimxAirbotRoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 63  # 双足机器人总观测维度: 3+3+3+3+14+14+3+3+3+14 = 63 (删除夹爪后)
        symmetric = False  #true :  set num_privileged_obs = None;    false: num_privileged_obs = observations + 187 ,set "terrain.measure_heights" to true
        num_privileged_obs = num_observations + 14 + 187 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 14 # 双足关节(8) + 机械臂关节(6) (无夹爪)
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm 
        episode_length_s = 20 # episode length in seconds 

        # 注释：观测维度分解
        # base_lin_vel: 3维, base_ang_vel: 3维, projected_gravity: 3维
        # commands: 3维, dof_err: 16维, dof_vel: 16维
        # gripper_pos: 3维, goal_pos: 3维, pos_error: 3维, actions: 16维
        # 总计: 3+3+3+3+16+16+3+3+3+16 = 69维
        
        class biped:
            biped_num_actions = 8 # 双足关节：4个左腿 + 4个右腿

        class arm:
            arm_num_actions = 6 # 机械臂关节：J1-J6
            
        # class gripper:
        #     gripper_num_actions = 0 # 已删除夹爪关节

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        hybrid_mode = "mixed"
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-0., 1.]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class terrain( LeggedRobotCfg.env ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0,1, 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.284] # x,y,z [m] - 双足机器人初始高度
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 双足关节 - 左腿
            "abad_L_Joint": 0.0,   # 左髋部外展
            "hip_L_Joint": 0.0,    # 左髋部俯仰
            "knee_L_Joint": 0.0,   # 左膝关节
            "ankle_L_Joint": 0.0,  # 左踝关节
            # 双足关节 - 右腿  
            "abad_R_Joint": 0.0,   # 右髋部外展
            "hip_R_Joint": 0.0,    # 右髋部俯仰
            "knee_R_Joint": 0.0,   # 右膝关节
            "ankle_R_Joint": 0.0,  # 右踝关节
            # 机械臂关节
            "J1": 0.0,    # 机械臂基座旋转
            "J2": 0.0,    # 肩部俯仰
            "J3": 0.0,    # 肘部
            "J4": 0.0,    # 腕部俯仰
            "J5": 0.0,    # 腕部偏航
            "J6": 0.0,    # 腕部旋转
            # 已删除夹爪关节
        }
        
    class goal_ee:
        local_axis_z_offset = 0.3
        init_local_cube_object_pos = [0.5,0,0.35]
        num_commands = 3
        traj_time = [0.6, 1.2]
        hold_time = [0.2, 0.4]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        num_collision_check_samples = 10
        command_mode = 'cart'
        class ranges:

            init_pos_l = [0.3, 0.6]
            init_pos_p = [-1 * np.pi / 6, 1 * np.pi / 3]
            init_pos_y = [-1 * np.pi / 4, 1 * np.pi / 4]
            final_delta_orn = [[-0, 0], [-0, 0], [-0, 0]]

        class init_ranges:
            pos_l = [0.3, 0.5] # min max [m/s]
            pos_p = [np.pi / 4, 3 * np.pi / 4]   # min max [m/s]
            pos_y = [0, 0]    # min max [rad/s]

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.5
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-8
            base_height = -0.2
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits =-10.
            object_distance = 2.
            object_distance_l2=-10

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        
        # 双足机器人关节控制参数
        stiffness = {
            "J": 20,          # 机械臂关节刚性(J1-J6)
            "end": 5,         # 夹爪关节刚性
            "ankle": 20,      # 踝关节刚性(双足需要更高刚性)
            "knee": 40,       # 膝关节刚性
            "hip": 40,        # 髋关节刚性  
            "abad": 40        # 髋部外展关节刚性
        }  
        damping = {
            "J": 0.5,         # 机械臂关节阻尼
            "end": 0.1,       # 夹爪关节阻尼
            "ankle": 1.0,     # 踝关节阻尼
            "knee": 1.0,      # 膝关节阻尼
            "hip": 1.0,       # 髋关节阻尼
            "abad": 1.0       # 髋部外展关节阻尼
        }
        
        # 机械臂位置控制
        arm_control_type = 'position'
        arm_stiffness = 20.0  # 位置控制刚性
        arm_damping = 0.5     # 位置控制阻尼
        
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = False
        friction_range = [0.2, 1.5]
        randomize_base_mass = False
        added_mass_range = [-4., 4.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

        randomize_base_com = False
        added_com_range = [-0.15, 0.15]
        randomize_motor = False
        motor_strength_range = [0.8, 1.2]


    class asset( LeggedRobotCfg.asset ):
        file = '{LOCO_MANI_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf'
        name = "limx_airbot"

        # 双足机器人足部
        foot_name = "ankle"  # 双足机器人的足部链接名称
        arm_link_name = ["link"]  # 机械臂链接名称前缀
        arm_joint_name = ["J"]    # 机械臂关节名称前缀(仅J1-J6)
        biped_joint_name = ["_L_Joint", "_R_Joint"]  # 双足关节名称(左腿_L_Joint，右腿_R_Joint)
        leg_joint_name = ["_R_Joint", "_L_Joint"]     # 腿部关节名称
        arm_gripper_name = "link6"  # 机械臂末端执行器(删除夹爪后使用link6)

        # 碰撞检测配置
        penalize_contacts_on = ["base", "knee", "hip", "abad"]  # 惩罚接触的链接
        terminate_after_contacts_on = []  # 终止条件的接触链接
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            gripper_track = 1.0
        clip_observations = 100.
        clip_actions = 100.

class LimxAirbotRoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 21
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'limx_airbot'
        resume = False
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000
        save_interval =100
        load_run = -1
        checkpoint = -1
        train =True

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        activation = 'elu'   # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
