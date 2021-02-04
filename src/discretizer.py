import gym
import numpy as np


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SonicDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        if env.game == "SuperMarioBros-Nes":
            combos = [[], ['LEFT'], ['RIGHT'], ['RIGHT', 'A'], ['RIGHT', 'B'], ['RIGHT', 'A', 'B'], ['A']]
        else:
            combos = [[], ['RIGHT'], ['RIGHT', 'A'], ['RIGHT', 'B'], ['RIGHT', 'A', 'B'], ['A'], ['LEFT'],
                      ['LEFT', 'A'], ['LEFT', 'B'], ['LEFT', 'A', 'B'], ['DOWN'], ['UP']]
        super().__init__(env=env, combos=combos)
