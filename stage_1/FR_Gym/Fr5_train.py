
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(r"FR5_Reinforcement-learning/utils")

from stable_baselines3 import A2C,PPO,DDPG,TD3,SAC,DQN
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from Fr5_env import FR5_Env
import time

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback
from Callback import TensorboardCallback
from loguru import logger
from utils.arguments import get_args

now = time.strftime('%m%d-%H%M%S', time.localtime())
args, kwargs = get_args()

# HACK
models_dir = args.models_dir
logs_dir = args.logs_dir
checkpoints = args.checkpoints
test = args.test

def make_env(i):
    def _init():
        if i == 0:
            env = FR5_Env(gui=False)
        else:
            env = FR5_Env(gui=False)
        env = Monitor(env, logs_dir)
        env.render()
        env.reset()
        return env
    set_random_seed(0)
    return _init

if __name__ == '__main__':
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)

    # Instantiate the env
    num_train = 16
    env = SubprocVecEnv([make_env(i) for i in range(num_train)])

    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir,
                device="cuda", policy_kwargs=dict(net_arch=[256, 256]))
                
    # 读取已有模型，接着训练
    # model = PPO.load(r"/home/woshihg/PycharmProjects/FR5_Reinforcement-learning_1/FR_Gym/FR5_Reinforcement-learning
    # /models/PPO/1230-180559/best_model.zip",env=env,print_system_info=True)
    
    model.set_logger(new_logger)
    tensorboard_callback = TensorboardCallback()

    # 创建测试环境回调函数
    eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             log_path=logs_dir, eval_freq=3000,
                             deterministic=True, render=True,n_eval_episodes = 100)

    TIMESTEPS = args.timesteps
    for eposide in range(1000):
        # 创建 CheckpointCallback 实例来保存模型检查点
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints)
        model.learn(total_timesteps=TIMESTEPS,
                    tb_log_name=f"1PPO-run-eposide{eposide}", # TensorBoard 日志运行的名称
                    reset_num_timesteps=False,  # 是否重置模型的当前时间步数
                    callback=CallbackList([eval_callback,tensorboard_callback]),  # 在每一步调用的回调，可以用CheckpointCallback来创建一个存档点和规定存档间隔。
                    log_interval=10  #  记录一次信息的时间步数
                    )

        # 保存模型
        model.save(models_dir+f"/1PPO-run-eposide{eposide}")
        logger.info(f"**************eposide--{eposide} saved**************")