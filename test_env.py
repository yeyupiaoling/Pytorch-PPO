import time

import cv2
import numpy as np

from src import retrowrapper


def main():
    # 获取游戏
    env = retrowrapper.RetroWrapper(game="SuperMarioBros-Nes")
    print(env.observation_space.shape)
    print(env.action_space.n)

    obs = env.reset()

    while True:
        # 游戏生成的随机动作，int类型数值
        action = env.action_space.sample()
        # 执行游戏
        obs, reward, terminal, info = env.step(action)
        # 显示连续动作
        obs = np.squeeze(obs)
        obses = obs[0]
        for i in range(1, obs.shape[0]):
            obses = np.hstack([obses, obs[i]])
        cv2.imshow('obes', obses)
        cv2.waitKey(1)
        env.render()
        print("=" * 50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", terminal)
        print("info:", info)
        if terminal:
            # obs = env.reset()
            break


if __name__ == "__main__":
    main()
