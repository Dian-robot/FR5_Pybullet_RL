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
from interval import Interval


def cal_success_reward(self, distance):
    '''
        计算成功/失败奖励
    '''
    # 1.计算碰撞奖励
    # 若机械臂成功抓取目标，那么任务成功
    # 若机械臂发生其他碰撞（桌子或其他关节碰撞目标），那么任务失败
    gripper_joint_indices = [8, 9]
    button_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.button)
    cup_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.target)
    fr5_table_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
    target_table_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.targettable)
    self_obstacle_contact_points = p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle)

    # 定义碰撞变量
    gripper_button_contact = False
    other_contact = False
    fr5_target_contact = False
    target_table_contact = False
    left_contact = False
    right_contact = False
    both_contact = False
    table_contact = False
    obstacle_contact = False
    collide_with_obstacle = False  #穿模

    # 碰撞桌子
    for contact_point in fr5_table_contact_points or target_table_contact_points:
        link_index = contact_point[3]
        if not (link_index == 0 or link_index == 1):
            table_contact = True

    # 碰撞障碍物
    for contact_point in self_obstacle_contact_points or cup_contact_points:
        point_pos = contact_point[5]
        obstacle_contact = True
        #穿模检测
        if (self.obstacle_position[2] < point_pos[2] < self.obstacle_position[2] + self.obstacle_height
                and point_pos[1] > self.obstacle_position[1]):
            collide_with_obstacle = True

    # 碰撞按钮
    for contact_point in button_contact_points:
        link_index = contact_point[3]
        if link_index in gripper_joint_indices:
            gripper_button_contact = True
        else:
            other_contact = True

    success_reward = 0
    # 一阶段成功奖励：
    if self.stage1_success is True:
        success_reward = 100
        logger.info("一阶段成功！执行步数：%s  距离目标:%s" % (self.step_num, distance))
        self.stage1_success = False
    # 夹爪中心和目标之间距离小于一定值，则任务成功
    if self.success is True and self.step_num <= 100:
        success_reward = 1000
        self.terminated = True
        self.success = True
        logger.info("成功抓取！！！！！！！！！！执行步数：%s  距离目标:%s" % (self.step_num, distance))
        # self.truncated = True

    # 碰撞桌子，或者碰撞自身，或者碰撞台子
    elif table_contact:
        success_reward = - 10
        # self.truncated = True
    elif obstacle_contact:
        success_reward = - 10
        # self.terminated = True
        self.collide += 1
        # if collide_with_obstacle:
        #     # 开始的时候设置成-100
        #     success_reward = - 100
        #     self.terminated = True
        #     logger.info("穿模！ 执行步数：%s    距离目标:%s" % (self.step_num, distance))
        # self.truncated = True
    elif gripper_button_contact:
        success_reward = - 10
        # self.terminated = True
        logger.info("失败！碰撞其他物体！ 执行步数：%s    距离目标:%s" % (self.step_num, distance))
        # self.truncated = True
    # 一直碰撞障碍物
    if self.collide > 5:
        success_reward = - 100
        self.terminated = True
        logger.info("失败！碰撞次数过多！ 执行步数：%s    距离目标:%s" % (self.step_num, distance))

    # 机械臂执行步数过多
    if self.step_num > 100:
        success_reward = - 1
        self.terminated = True
        logger.info("失败！执行步数过多！ 执行步数：%s    距离目标:%s" % (self.step_num, distance))

    return success_reward


def cal_dis_reward(self, distance):
    '''计算距离奖励'''
    if self.step_num == 0:
        distance_reward = 0
    else:
        if self.goalchange is False:
            distance_reward = 1000 * (self.distance_last - distance)
        else:
            distance_reward = 10000 * (self.distance_last - distance)
    # logger.debug("相对距离：%f"%(self.distance_last-distance))
    # logger.debug("距离奖励:%f"%distance_reward)
    # 保存上一次的距离
    self.distance_last = distance
    return distance_reward


def cal_pose_reward(self):
    '''姿态奖励'''
    # 计算夹爪的朝向
    gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    gripper_orientation = R.from_quat(gripper_orientation)
    gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)
    # 计算夹爪的姿态奖励
    pose_reward = -(
            pow(gripper_orientation[0] + 90, 2) + pow(gripper_orientation[1], 2) + pow(gripper_orientation[2], 2))
    # logger.debug("姿态奖励：%f"%pose_reward)
    # 一阶段0.01
    return pose_reward * 0.1


def grasp_reward(self):
    '''获取奖励'''
    info = {}
    total_reward = 0

    distance = get_distance(self)
    pose_reward = cal_pose_reward(self)
    real_distance = get_real_distance(self)
    # 第一阶段0.02,-100第二阶段0.02,-3（姿态惩罚本身增加10倍）
    judge_success(self, distance, pose_reward, success_dis=0.02, success_pose=-5)

    # 计算奖励
    success_reward = cal_success_reward(self, distance)
    distance_reward = cal_dis_reward(self, distance)
    total_reward = success_reward + pose_reward + distance_reward

    self.truncated = False
    self.reward = total_reward
    info['reward'] = self.reward
    info['is_success'] = self.success
    info['step_num'] = self.step_num

    info['success_reward'] = (1 if self.success else 0)
    info['distance_reward'] = distance_reward
    info['pose_reward'] = pose_reward

    return total_reward, info


def judge_success(self, distance, pose, success_dis, success_pose):
    '''判断成功或失败'''
    if self.goalchange is True:
        if distance < success_dis:
            if pose > success_pose:
                self.success = True
                logger.info("姿势正确！ %s " % (pose))
            else:
                self.success = False
                logger.info("到达但姿势不正确！ %s " % (pose))
        else:
            self.success = False
    else:
        if distance < success_dis :
            # 完成一阶段目标后，切换到下一阶段目标
            self.change_goal()
            self.stage1_success = True
            self.success = False
            logger.info("到达一阶段目标%s " % (distance))
            self.distance_last = get_distance(self)
        else:
            self.success = False
        # total_reward = total_reward + (0.3 - distance)


def get_distance(self):
    '''判断机械臂与夹爪的距离'''
    Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
    Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
    Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
    relative_position = np.array([0, 0, self.grasp_center_dis])
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position
    # self.target_position = np.array(p.getBasePositionAndOrientation(self.button)[0])
    distance = math.sqrt((gripper_centre_pos[0] - self.goalx) ** 2 +
                         ((gripper_centre_pos[1] - self.goaly) ** 2) +
                         (gripper_centre_pos[2] - self.goalz) ** 2)
    # logger.info("distance:%s"%str(distance))
    return distance


def get_real_distance(self):
    '''判断机械臂与夹爪的距离'''
    Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
    Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
    Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
    relative_position = np.array([0, 0, 0.15])
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    gripper_centre_pos = [Gripper_posx, Gripper_posy, Gripper_posz] + rotated_relative_position
    self.target_position = np.array(p.getBasePositionAndOrientation(self.button)[0])
    distance = math.sqrt((gripper_centre_pos[0] - self.target_position[0]) ** 2 +
                         (gripper_centre_pos[1] - self.target_position[1]) ** 2 +
                         (gripper_centre_pos[2] - self.target_position[2]) ** 2)
    # logger.debug("distance:%s"%str(distance))
    return distance
