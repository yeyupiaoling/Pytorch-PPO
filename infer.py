import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game",        type=str, default="SuperMarioBros-Nes")
    parser.add_argument("--saved_path",  type=str, default="models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def infer(args):
    # 固定初始化状态
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # 创建游戏环境
    env = create_train_env(args.game)
    # 创建模型
    model = PPO(env.observation_space.shape[0], env.action_space.n)
    # 加载模型参数文件
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/model_best_{}.pth".format(args.saved_path, args.game)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/model_best_{}.pth".format(args.saved_path, args.game),
                                         map_location=lambda storage, loc: storage))
    # 切换评估模式
    model.eval()
    # 获取刚开始的游戏图像
    state = torch.from_numpy(env.reset())
    total_reward = 0
    while True:
        # 显示界面
        env.render()
        # 使用GPU计算
        if torch.cuda.is_available():
            state = state.cuda()
        # 预测动作概率和评估值
        logits, value = model(state)
        # 获取动作的序号
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        # 执行游戏
        state, reward, done, info = env.step(action)
        total_reward += reward
        # 转换每一步都游戏状态
        state = torch.from_numpy(state)
        print(info)
        # 游戏通关
        if done:
            print("游戏结束，得分：%f" % total_reward)
            break
        time.sleep(0.05)
    env.render(close=True)
    env.close()


if __name__ == "__main__":
    opt = get_args()
    infer(opt)
