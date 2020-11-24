import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import torch.multiprocessing as mp


# 对图像进行预处理
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0

    # 执行一步游戏
    def step(self, action):
        # 执行游戏
        state, reward, done, info = self.env.step(action)
        # 对图像进行处理
        state = process_frame(state)
        # 计算当前奖励
        reward += (info["score"] - self.curr_score) / 40.
        # 记录当前总分
        self.curr_score = info["score"]
        # 游戏结束
        if done:
            # 判断结束的原因
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    # 重置游戏状态
    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    # 执行一步游戏
    def step(self, action):
        # 总奖励分数
        total_reward = 0
        # 多步的游戏状态
        last_states = []
        # 执行多步游戏
        for i in range(self.skip):
            # 执行游戏
            state, reward, done, info = self.env.step(action)
            # 记录分数
            total_reward += reward
            # 取中间帧
            if i >= self.skip / 2:
                last_states.append(state)
            # 结束游戏
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        # 将两帧图片拼接起来
        max_state = np.max(np.concatenate(last_states, 0), 0)
        # 指定前面三帧都是上三个的
        self.states[:-1] = self.states[1:]
        # 最后一个指定为当前的游戏帧
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    # 重置游戏状态
    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


# 创建游戏环境
def create_train_env(world, stage, actions):
    # 创建游戏
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    # 转换游戏对象，使得可以自定义
    env = JoypadSpace(env, actions)
    # 自定义奖励逻辑
    env = CustomReward(env)
    # 自定义执行游戏帧
    env = CustomSkipFrame(env)
    return env


# 定义多进程环境
class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        if action_type == "right":
            # 正确的动作
            actions = RIGHT_ONLY
        elif action_type == "simple":
            # 简单的动作
            actions = SIMPLE_MOVEMENT
        else:
            # 更复杂的动作
            actions = COMPLEX_MOVEMENT
        # 创建多个游戏环境
        self.envs = [create_train_env(world, stage, actions) for _ in range(num_envs)]
        # 获取游戏图像的数量
        self.num_states = self.envs[0].observation_space.shape[0]
        # 获取动作的数量
        self.num_actions = len(actions)
        # 启动多有效线程
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            # 接收发过来的游戏动作
            request, action = self.env_conns[index].recv()
            if request == "step":
                # 执行游戏
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                # 重置游戏状态
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
