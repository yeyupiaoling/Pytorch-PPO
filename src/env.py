import retro
import torch.multiprocessing as mp

from src import retrowrapper


# 创建游戏环境
def create_train_env(is_train=True):
    env = retrowrapper.RetroWrapper(game='SuperMarioBros-Nes',
                                    use_restricted_actions=retro.Actions.DISCRETE,
                                    skill_frame=4,
                                    resize_shape=(1, 84, 84),
                                    render_preprocess=False,
                                    is_train=is_train)
    return env


# 定义多进程环境
class MultipleEnvironments:
    def __init__(self, num_envs):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        # 创建多个游戏环境
        self.envs = [create_train_env() for _ in range(num_envs)]
        # 获取游戏图像的数量
        self.num_states = self.envs[0].observation_space.shape[0]
        # 获取动作的数量
        self.num_actions = self.envs[0].action_space.n
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
