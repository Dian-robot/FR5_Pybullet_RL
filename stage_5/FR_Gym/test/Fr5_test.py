'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.append(r"/root/FR5_Pybullet_RL/stage_5/utils")
sys.path.append(r"/root/FR5_Pybullet_RL/stage_5/FR_Gym")
from stable_baselines3 import A2C, PPO, DDPG, TD3

sys.path.append('..')
from Fr5_env import FR5_Env
import time
from utils.arguments import get_args

if __name__ == '__main__':
    args, kwargs = get_args()
    env = FR5_Env(gui=False)
    env.render()
    model = PPO.load(r"/root/FR5_Pybullet_RL/stage_5/best_model.zip")
    # model = TD3.load("F:\\Pycharm_project\\RL\\models\\TD3\\TD3-run-eposide270.zip")
    # model = DDPG.load("F:\\Pycharm_project\\RL\\models\\DDPG\\DDPG-run-eposide282.zip")
    # model = DDPG.load(r"D:\postgraduate\project\FR_Reinforcement_all_in\FR5_Reinforcement-learning\models\PPO\0221-120301\best_model.zip")
    test_num = args.test_num  # 测试次数
    success_num = 0  # 成功次数
    print("测试次数：", test_num)
    for i in range(test_num):
        state, _ = env.reset()
        # time.sleep(1)
        done = False
        score = 0
        # time.sleep(3)
        step = 0
        while not done:
            step += 1
            # action = env.action_space.sample()     # 随机采样动作
            # print("state:", state)
            action, _ = model.predict(observation=state,deterministic=True)

            # print("action:",action)
            # if step % 40 == 0:
            #     env.moveTarget()
            state, reward, done, _, info = env.step(action=action)

            score += reward
            # env.render()
            # time.sleep(0.01)

        if info['is_success']:
            success_num += 1
        print("奖励：", score)
    success_rate = success_num / test_num
    print("成功率：", success_rate)
    env.close()
