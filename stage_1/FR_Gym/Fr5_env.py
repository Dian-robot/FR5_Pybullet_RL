
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
from math import cos, sin

class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gui=False):
        super(FR5_Env).__init__()
        self.step_num = 0
        self.Con_cube = None
        # self.last_success = False

        # 设置最小的关节变化量
        low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1, 15), dtype=np.float32)
        high = np.ones((1, 15), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.grasp_zero = [0, 0]
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

    def dh_matrix(self,alpha, a, d, theta):
        # 传入四个DH参数，根据公式3-6，输出一个T矩阵。
        alpha = alpha / 180 * np.pi
        theta = theta / 180 * np.pi
        matrix = np.identity(4)
        matrix[0, 0] = cos(theta)
        matrix[0, 1] = -sin(theta)*cos(alpha)
        matrix[0, 2] = sin(theta)*sin(alpha)
        matrix[0, 3] = a*cos(theta)
        matrix[1, 0] = sin(theta)
        matrix[1, 1] = cos(theta)*cos(alpha)
        matrix[1, 2] = -cos(theta)*sin(alpha)
        matrix[1, 3] = a*sin(theta)
        matrix[2, 0] = 0
        matrix[2, 1] = sin(alpha)
        matrix[2, 2] = cos(alpha)
        matrix[2, 3] = d
        matrix[3, 0] = 0
        matrix[3, 1] = 0
        matrix[3, 2] = 0
        matrix[3, 3] = 1
        return matrix

    def forward_kinematics(self, angles):
        joint_num = 6

        # --- Robotic Arm construction ---
        # DH参数表，分别用一个列表来表示每个关节的东西。
        joints_alpha = [90, 0, 0, 90, -90, 0]
        joints_a = [0, -425, -395, 0, 0, 0]
        joints_d = [152, 0.0, 0.4, 102, 102, 244]
        joints_theta = [0, 0, 0, 0, 0, 0]
        joint_hm = []
        for i in [1, 2, 3, 4, 5, 6]:        
            joint_hm.append(self.dh_matrix(alpha=joints_alpha[i-1], a=joints_a[i-1], d=joints_d[i-1], theta=joints_theta[i-1]+angles[i-1]))

        # -----------连乘计算----------------------
        for i in range(joint_num-1):
            joint_hm[i+1] = np.dot(joint_hm[i], joint_hm[i+1])    
        # Prepare the coordinates for plotting
        # for i in range(joint_num):
        #     print(np.round(joint_hm[i][:3, 3], 5))
        # 获取坐标值
        X = [hm[0, 3] for hm in joint_hm]
        Y = [hm[1, 3] for hm in joint_hm]
        Z = [hm[2, 3] for hm in joint_hm]
        x = -X[5]/1000
        y = -Y[5]/1000
        z = Z[5]/1000
        return [x, y, z]
    
    def flashUR5(self):
        '''夹爪干涉出现时，重置机械臂'''
        if self.grasp_zero[0] == 0:
            self.grasp_zero = [0.075, 0.075]
        elif self.grasp_zero[0] == 0.075:
            self.grasp_zero = [0, 0]

    def init_env(self):
        """
            仿真环境初始化
        """
        # 创建机械臂
        self.fr5 = self.p.loadURDF(
            r"D:\postgraduate\project\FR_Reinforcement\stage_1\fr5_description\urdf\fr5v6.urdf",
            useFixedBase=True, basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            flags=p.URDF_USE_SELF_COLLISION
        )

        # 创建桌子
        self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # 创建目标
        self.cup_height = 0.1
        #第一阶段0.025，第二阶段半径0.3
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                        radius=0.028, height=self.cup_height)

        self.target = self.p.createMultiBody(baseMass=0.5,  # 质量
                                             baseCollisionShapeIndex=collisionTargetId,
                                             basePosition=[0.5, 0.5, 2])
        p.changeDynamics(self.target, -1, lateralFriction=10.0, spinningFriction=1, rollingFriction=1)

        # 创建目标杯子的台子
        self.targettable_height = 0.04
        tableId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                              radius=0.04, height=self.targettable_height)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                                                  baseCollisionShapeIndex=tableId,
                                                  basePosition=[0.5, 0.5, 2])

    def step(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        Fr5_joint_angles = np.array(joint_angles[:6]) + (np.array(action[0:6]) / 180 * np.pi)

        # 初始化夹爪位置
        gripper = self.grasp_zero
        # gripper = [0.03, 0.03]
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

    def moveTarget(self):
        # 移动目标位置
        delta_x = np.random.uniform(-0.1, 0.1, 1)
        delta_y = np.random.uniform(-0.1, 0.1, 1)
        self.now_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])
        self.target_position = self.now_target_position + np.array([delta_x[0], delta_y[0], 0])
        p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.targettable_position = self.target_position + np.array([0, 0, -0.16])
        p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])

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
        p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                    targetPositions=neutral_angle)
        for i in range(20):
            self.p.stepSimulation()
        '''干涉检测'''
        error_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.fr5)
        for contact_point in error_contact_points:
            link_index = contact_point[3]
            if link_index == 7 or link_index == 8:
                logger.info("夹爪干涉出现！")
                self.flashUR5()
                for i in range(10):
                    self.p.stepSimulation()
                break
        # 重新设置目标位置
        #最好在0.15之外
        self.goalx = np.random.uniform(0.1, 0.35, 1)[0]
        # self.goalx = 0.3
        self.goaly = np.random.uniform(0.45, 0.6, 1)[0]
        # self.goaly = 0.5
        self.goalz = np.random.uniform(0.06, 0.1, 1)[0]
        # self.goalz = 0.08
        self.target_position = [self.goalx, self.goaly, self.goalz]
        self.init_target_position = self.target_position
        # self.target_position = [0., 0.6, 0.2]
        self.targettable_position = [self.goalx, self.goaly,
                                     self.goalz - self.cup_height / 2 - self.targettable_height / 2]


        self.p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        self.ori_target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(10./240.)

        # 第一阶段后撤
        self.goalx = self.goalx
        self.goaly = self.goaly + np.random.uniform(-0.06, -0.02, 1)[0]
        self.goalz = self.goalz + np.random.uniform(0.01, 0.05, 1)[0]
        self.goalchange = False
        self.stage1_success = False
        self.add_debug_line()
        
        self.get_observation()
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        # print("observation", self.observation)
        return self.observation, infos

    def change_goal(self):
        goal_pos = p.getBasePositionAndOrientation(self.target)[0]
        print("goal_pos", goal_pos)
        # self.goalx = goal_pos[0]
        # self.goaly = goal_pos[1]
        # self.goalz = goal_pos[2]
        self.goalx = self.init_target_position[0]
        self.goaly = self.init_target_position[1]
        self.goalz = self.init_target_position[2]
        self.goalchange = True

    def get_gripper_position(self):
        '''获取夹爪中心位置和朝向'''
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.169])

        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = Gripper_pos + rotated_relative_position

        return gripper_centre_pos

    def get_observation(self, add_noise=True,SPF = True):
        """计算observation"""
        Gripper_pos = p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.169])

        # # 固定夹爪相对于机械臂末端的相对位置转换
        # rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        # rotated_relative_position = rotation.apply(relative_position)
        # gripper_centre_pos = Gripper_pos + rotated_relative_position

        joint_angles = [0, 0, 0, 0, 0, 0]
        for i in [1, 2, 3, 4, 5, 6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i - 1] = joint_info[0] * 180 / np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i - 1] = self.add_noise(joint_angles[i - 1], range=0, gaussian=False)

        if add_noise == True and (SPF == False):
            Gripper_pos = self.forward_kinematics(joint_angles)
            Gripper_posx = Gripper_pos[0]
            Gripper_posy = Gripper_pos[1]
            Gripper_posz = Gripper_pos[2]
            gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz]
            # print("grasp_pos",gripper_centre_pos)
        else:
            Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
            Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
            Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]

            relative_position = np.array([0, 0, 0.169])
            # 固定夹爪相对于机械臂末端的相对位置转换
            rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
            rotated_relative_position = rotation.apply(relative_position)
            # print([Gripper_posx, Gripper_posy,Gripper_posz])
            gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz] + rotated_relative_position

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        # x,y,z坐标归一化
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0] + 0.5) / 1,
                                           (gripper_centre_pos[1] + 0.5) / 2,
                                           (gripper_centre_pos[2] + 0.5) / 1], dtype=np.float32)
        #未统一前
        # obs_target_position = np.array([(self.goalx + 0.4) / 0.8,
        #                                 self.goaly / 1,
        #                                 self.goalz / 0.5], dtype=np.float32)

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
    def add_debug_line(self):
        '''添加调试线'''
        # 删除之前的线
        if hasattr(self, 'x_line_id'):
            p.removeUserDebugItem(self.x_line_id)
        if hasattr(self, 'y_line_id'):
            p.removeUserDebugItem(self.y_line_id)
        if hasattr(self, 'z_line_id'):
            p.removeUserDebugItem(self.z_line_id)
        # x 轴
        # 固定夹爪相对于机械臂末端的相对位置转换
        frame_start_postition, frame_posture = p.getBasePositionAndOrientation(self.target)
        frame_start_postition = [self.goalx, self.goaly, self.goalz]
        R_Mat = np.array(p.getMatrixFromQuaternion(frame_posture)).reshape(3, 3)
        x_axis = R_Mat[:, 0]
        x_end_p = (np.array(frame_start_postition) + np.array(x_axis * 5)).tolist()
        self.x_line_id = p.addUserDebugLine(frame_start_postition, x_end_p, [1, 0, 0])

        # y 轴
        y_axis = R_Mat[:, 1]
        y_end_p = (np.array(frame_start_postition) + np.array(y_axis * 5)).tolist()
        self.y_line_id = p.addUserDebugLine(frame_start_postition, y_end_p, [0, 1, 0])

        # z轴
        z_axis = R_Mat[:, 2]
        z_end_p = (np.array(frame_start_postition) + np.array(z_axis * 5)).tolist()
        self.z_line_id = p.addUserDebugLine(frame_start_postition, z_end_p, [0, 0, 1])

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    Env = FR5_Env(gui=True)
    Env.reset()
    # check_env(Env, warn=True)
    # for i in range(100):
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
    time.sleep(100)
