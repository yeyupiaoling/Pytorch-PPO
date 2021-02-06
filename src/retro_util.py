import cv2
import retro
import numpy as np
from gym.spaces import Box
from src.discretizer import SonicDiscretizer


class RetroEnv(retro.RetroEnv):
    def __init__(self, game, state=retro.State.DEFAULT, scenario=None, info=None,
                 use_restricted_actions=retro.Actions.FILTERED, record=False, players=1,
                 inttype=retro.data.Integrations.STABLE, obs_type=retro.Observations.IMAGE,
                 resize_shape=(1, 112, 112), skill_frame=4, render_preprocess=False):
        super(RetroEnv, self).__init__(game, state=state, scenario=scenario, info=info,
                                       use_restricted_actions=use_restricted_actions,
                                       record=record, players=players, inttype=inttype, obs_type=obs_type)
        self.game = game
        self.resize_shape = resize_shape
        self.skill_frame = skill_frame
        self.render_preprocess = render_preprocess
        self.observation_space = Box(low=0, high=255, shape=(skill_frame, resize_shape[1], resize_shape[2]))
        self.game_info = None
        self.obses = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.use_restricted_actions = use_restricted_actions

        self.time_sum, self.time, self.score, self.xscrollLo = 0, 0, 0, 0

    def step(self, action):
        total_reward = 0
        last_states = []
        terminal = False
        info = {}
        # 每一次支持多个帧，让模型看到操作效果
        for i in range(self.skill_frame):
            obs, reward, terminal, info = super(RetroEnv, self).step(action)
            # 记录所有步数的总分
            total_reward += reward
            # 取中间帧
            if i >= self.skill_frame / 2:
                # 图像预处理
                obs = self.preprocess(obs)
                last_states.append(obs)
            # 死一条命就结束
            if info['lives'] != 2:
                terminal = True
            # 结束游戏
            if terminal:
                total_reward -= 50
                self.reset()
                return self.obses[None, :, :, :].astype(np.float32), total_reward, terminal, info
        # 将两帧图片拼接起来
        max_state = np.max(np.concatenate(last_states, 0), 0)
        # 指定前面三帧都是上三个的
        self.obses[:-1] = self.obses[1:]
        # 最后一个指定为当前的游戏帧
        self.obses[-1] = max_state
        return self.obses[None, :, :, :].astype(np.float32), total_reward, terminal, info

    def reset(self):
        obs = super(RetroEnv, self).reset()
        obs = self.preprocess(obs)
        self.obses = np.concatenate([obs for _ in range(self.skill_frame)], 0)
        return self.obses[None, :, :, :].astype(np.float32)

    # 图像预处理
    def preprocess(self, observation):
        if self.resize_shape is None:
            if self.render_preprocess:
                observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                # 显示处理过的图像
                cv2.imshow("preprocess1", observation)
                cv2.waitKey(1)
            return observation
        assert self.resize_shape[0] == 1 or self.resize_shape[0] == 3
        if self.resize_shape[0] == 1:
            # 把图像转成灰度图
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = cv2.resize(observation, (self.resize_shape[2], self.resize_shape[1]),
                                     interpolation=cv2.INTER_AREA)
            if self.render_preprocess:
                # 显示处理过的图像
                cv2.imshow("preprocess2", observation)
                cv2.waitKey(1)
            observation = np.expand_dims(observation, axis=0)
        else:
            observation = cv2.resize(observation, (self.resize_shape[2], self.resize_shape[1]),
                                     interpolation=cv2.INTER_AREA)
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            if self.render_preprocess:
                # 显示处理过的图像
                cv2.imshow("preprocess3", observation)
                cv2.waitKey(1)
            observation = observation.transpose((2, 0, 1))
        observation = observation / 255.0
        return observation


def retro_make_func(game, **kwargs):
    return SonicDiscretizer(RetroEnv(game, **kwargs))
