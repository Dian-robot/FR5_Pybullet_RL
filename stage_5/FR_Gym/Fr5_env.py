'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os
import pybullet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random
from reward import grasp_reward
from stable_baselines3 import PPO


class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gui=False):
        super(FR5_Env).__init__()
        self.step_num = 0
        self.Con_cube = None
        self.grasp_zero = [0, 0]
        self.grasp_zero_sym = [0.075, 0.075]
        self.grasp_effort_sym = [0.069, 0.069]
        self.grasp_effort_ori = [0.003, 0.003]
        self.grasp_center_dis = 0.169
        self.grasp_edge_dis = 0.180
        # self.last_success = False

        # 设置最小的关节变化量
        low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1, 15), dtype=np.float32)
        high = np.ones((1, 15), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 初始化pybullet环境
        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        # self.p.setTimeStep(1/240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 初始化环境
        self.init_env()

    def init_env(self):
        '''
            仿真环境初始化
        '''
        # boxId = self.p.loadURDF("plane.urdf")
        # self.first_model = PPO.load(
        #     "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_0/FR_Gym/FR5_Reinforcement-learning/models/PPO/1107-121135/best_model.zip")
        # 创建机械臂
        self.fr5 = self.p.loadURDF(
            r"/root/FR5_Pybullet_RL/stage_5/fr5_description/urdf/fr5v6.urdf",
            useFixedBase=True, basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            flags=p.URDF_USE_SELF_COLLISION
        )

        # 创建桌子
        self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # 创建按钮
        self.button_length = 0.01
        self.button_height = 0.285
        buttonId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                               radius=0.015, height=self.button_length)

        self.button = self.p.createMultiBody(baseMass=0,  # 质量
                                             baseCollisionShapeIndex=buttonId,
                                             basePosition=[0.5, 0.5, 2])

        # 创建目标
        self.cup_height = 0.08
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.03, height=self.cup_height)

        self.target = self.p.createMultiBody(baseMass=0.2,  # 质量
                                             baseCollisionShapeIndex=collisionTargetId,
                                             basePosition=[0.5, 0.5, 2])
        # self.target = self.p.loadURDF(
        #     "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_0/fr5_description/urdf/woshidhg.urdf",
        #     basePosition=[0.5, 0.5, 2])
        self.grasp_effort = [0.0030, 0.0030]
        p.changeDynamics(self.target, -1, lateralFriction=10.0, spinningFriction=1, rollingFriction=1)

        # self.targettable = self.p.loadURDF(
        #     "/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_0/fr5_description/urdf/woshidhg.urdf",
        #     basePosition=[0.5, 0.5, 2])

        # 创建初始的台子
        self.base_height = 0.04
        basetableId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                  radius=0.04, height=self.base_height)
        self.basetable = self.p.createMultiBody(baseMass=0,  # 质量
                                                baseCollisionShapeIndex=basetableId,
                                                basePosition=[0.5, 0.5, 2])
        p.changeDynamics(self.basetable, -1, lateralFriction=1.0, spinningFriction=0, rollingFriction=0)

        # # 创建障碍物
        self.obstacle_wide = 0.05
        obstacleId = self.p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                 halfExtents=[self.obstacle_wide * 2, self.obstacle_wide, 0.1])
        self.obstacle = self.p.createMultiBody(baseMass=0,  # 质量
                                               baseCollisionShapeIndex=obstacleId,
                                               basePosition=[0.5, 0.5, 2])
        # 创建目标台子
        self.targettable_height = 0.05
        targettableId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                    radius=0.04, height=self.targettable_height)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                                                  baseCollisionShapeIndex=targettableId,
                                                  basePosition=[0.5, 0.5, 2])

    def step(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        # 初始化关节角度列表
        joint_angles = []
        # 初始化夹爪位置
        gripper = [0.0, 0.0]

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        gripper = self.grasp_effort

        anglenow = np.hstack([Fr5_joint_angles, gripper])
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=anglenow)

        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward, info = grasp_reward(self)

        # observation计算
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        '''重置环境参数'''
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        # 重新设置机械臂的位置
        neutral_angle = [30, -137, 128, 9,
                         30, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]

        # 1.48354575 -1.20812401  2.08497612 -0.85981021  1.66050315  0.
        # neutral_angle = [1.48354575, -1.20812401, 2.08497612, -0.85981021, 1.66050315, 0., 0, 0]
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=neutral_angle)
        self.base_position = [0, 0.5, 1]
        self.p.resetBasePositionAndOrientation(self.obstacle, self.base_position, [0, 0, 0, 1])
        for i in range(20):
            self.p.stepSimulation()
        error_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.fr5)
        for contact_point in error_contact_points:
            link_index = contact_point[3]
            if link_index == 7 or link_index == 8:
                logger.info("夹爪干涉出现！")
                self.flashUR5()
                for i in range(10):
                    self.p.stepSimulation()
                break

        # 重新设置夹爪位置
        # 随机生成初始位置
        init_x = np.random.uniform(-0.2, 0.2, 1)[0]
        init_y = np.random.uniform(0.40, 0.65, 1)[0]
        init_z = self.cup_height / 2 + self.base_height + np.random.uniform(-0.02, 0.06, 1)[0]
        # init_x = 0.07
        # init_y = 0.55
        # init_z = 0.12
        Euler = [-1.578253446554462, 3.1415926535, 0]
        orn = p.getQuaternionFromEuler(Euler)
        pos = [init_x, init_y, init_z]
        ll = [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543]
        ul = [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543]
        jr = [6.28318530718, 6.28318530718, 5.6558, 6.28318530718, 6.28318530718, 6.28318530718]
        rp = [1.19826176, -1.2064331, 1.85829957, -0.72282605, 1.44937236, 0.]
        # 循环以使结果逼近
        for i in range(3):
            pos[1] = pos[1]
            joint_angles = p.calculateInverseKinematics(self.fr5, 6, pos, orn,
                                                        lowerLimits=ll,
                                                        upperLimits=ul,
                                                        jointRanges=jr,
                                                        restPoses=rp)
            joint_angles = list(joint_angles[:6]) + [0, 0]
            p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                        targetPositions=joint_angles)
            for i in range(20):
                self.p.stepSimulation()

        # 设置初始位置
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, self.grasp_center_dis])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        self.gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position

        self.gripperx = self.gripper_centre_pos[0]
        self.grippery = self.gripper_centre_pos[1]
        print("gripper_centre_pos", self.gripper_centre_pos)
        self.gripperz = self.gripper_centre_pos[2]
        # self.obstacle_position = [self.gripperx, self.grippery, self.button_height + Gripper_posz - 0.06]
        # self.p.resetBasePositionAndOrientation(self.obstacle, self.obstacle_position, [0, 0, 0, 1])
        Euler = [math.pi / 2, 0, 0]
        key_orn = p.getQuaternionFromEuler(Euler)

        # self.target_position = [0.5, 0.5, 1]
        self.button_position = [self.gripperx, self.grippery - self.obstacle_wide, self.button_height]
        self.p.resetBasePositionAndOrientation(self.button, self.button_position, key_orn)

        self.basetable_position = [self.gripperx, self.grippery,
                                   self.gripperz - self.cup_height / 2 - self.base_height / 2]
        self.p.resetBasePositionAndOrientation(self.basetable, self.basetable_position, [0, 0, 0, 1])

        self.cup_position = [self.gripperx, self.grippery, self.gripperz]
        self.p.resetBasePositionAndOrientation(self.target, self.cup_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.obstacle, [self.gripperx, self.grippery, self.gripperz+0.2], [0, 0, 0, 1])

        self.goalx = np.random.uniform(0.1, 0.5, 1)[0]
        self.goaly = np.random.uniform(0.35, 0.7, 1)[0]
        self.goalz = np.random.uniform(self.base_height + self.cup_height / 2 - 0.01,
                      self.base_height + self.cup_height / 2 + 0.05, 1)[0]
        self.goalx = 0.3
        self.goaly = 0.4
        self.goalz = 0.08
        self.targettable_position = [self.goalx, self.goaly,
                                     self.goalz - self.cup_height / 2 - self.targettable_height / 2]
        self.p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])

        self.ori_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
        # # 合上夹爪
        p.setJointMotorControlArray(self.fr5, [8, 9], p.POSITION_CONTROL,
                                    targetPositions=self.grasp_effort)
        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(1. / 240.)

        self.get_observation()
        infos = {'is_success': False, 'reward': 0, 'step_num': 0}
        # print("observation", self.observation)
        return self.observation, infos

    def flashUR5(self):
        """夹爪干涉出现时，反置机械臂"""
        if self.grasp_zero[0] == 0:
            self.grasp_zero = self.grasp_zero_sym
            self.grasp_effort = self.grasp_effort_sym
        elif self.grasp_zero[0] == self.grasp_zero_sym[0]:
            self.grasp_zero = [0, 0]
            self.grasp_effort = self.grasp_effort_ori

    def get_gripper_position(self):
        '''获取夹爪中心位置和朝向'''
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, self.grasp_center_dis])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = Gripper_pos + rotated_relative_position

        return gripper_centre_pos

    def get_observation(self, add_noise=False):
        """计算observation"""
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, self.grasp_center_dis])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = Gripper_pos + rotated_relative_position

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        # x,y,z坐标归一化
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.5) / 1,
                                           (gripper_centre_pos[1] + 0.5) / 2,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)

        obs_target_position = np.array([(self.goalx + 1) / 2,
                                        self.goaly / 1,
                                        self.goalz / 0.5], dtype=np.float32)
        # 夹爪朝向
        obs_gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        obs_gripper_orientation = R.from_quat(obs_gripper_orientation)
        obs_gripper_orientation = ((obs_gripper_orientation.as_euler('xyz', degrees=True) / 180) + 1) / 2

        self.observation = np.hstack(
            (obs_gripper_centre_pos, obs_joint_angles, obs_gripper_orientation, obs_target_position)).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 15)

    def get_observation_(self, add_noise=False):
        """计算observation"""
        gripper_centre_pos = self.get_gripper_position()
        obs_gripper_centre_pos = np.array(gripper_centre_pos, dtype=np.float32)

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=True)
        joint_angles = np.array(joint_angles, dtype=np.float32)
        self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        obs_target_position = np.array([self.target_position[0],
                                        self.target_position[1],
                                        self.target_position[2]], dtype=np.float32)
        self.observation = np.hstack(
            (obs_gripper_centre_pos, obs_target_position, joint_angles)).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1, 15)

    def render(self):
        '''设置观察角度'''
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=90, cameraPitch=-7.6, cameraTargetPosition=[0.39, 0.45, 0.42])

    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        '''添加噪声'''
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-5, 5)
        return angle


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    Env = FR5_Env(gui=True)
    Env.reset()

    # x 轴
    frame_start_postition, frame_posture = p.getBasePositionAndOrientation(Env.target)
    frame_start_postition = [Env.goalx, Env.goaly, Env.goalz]
    R_Mat = np.array(p.getMatrixFromQuaternion(frame_posture)).reshape(3, 3)
    x_axis = R_Mat[:, 0]
    x_end_p = (np.array(frame_start_postition) + np.array(x_axis * 5)).tolist()
    x_line_id = p.addUserDebugLine(frame_start_postition, x_end_p, [1, 0, 0])

    # y 轴
    y_axis = R_Mat[:, 1]
    y_end_p = (np.array(frame_start_postition) + np.array(y_axis * 5)).tolist()
    y_line_id = p.addUserDebugLine(frame_start_postition, y_end_p, [0, 1, 0])

    # z轴
    z_axis = R_Mat[:, 2]
    z_end_p = (np.array(frame_start_postition) + np.array(z_axis * 5)).tolist()
    z_line_id = p.addUserDebugLine(frame_start_postition, z_end_p, [0, 0, 1])

    # time.sleep(10)
    # check_env(Env, warn=True)

    for i in range(100):
        p.stepSimulation()
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
