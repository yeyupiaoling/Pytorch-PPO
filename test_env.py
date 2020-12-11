import retro

from src import retrowrapper


def main():
    # 获取游戏
    env = retrowrapper.RetroWrapper(game="SuperMarioBros-Nes",
                                    use_restricted_actions=retro.Actions.DISCRETE,
                                    skill_frame=4,
                                    resize_shape=(1, 84, 84),
                                    render_preprocess=False)
    obs = env.reset()

    while True:
        # 游戏生成的随机动作，int类型数值
        action = env.action_space.sample()
        # 执行游戏
        obs, reward, terminal, info = env.step(action)
        env.render()
        print("=" * 50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", terminal)
        print("info:", info)
        if terminal:
            obs = env.reset()


if __name__ == "__main__":
    main()
