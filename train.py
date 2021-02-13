import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.utils import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game",             type=str,   default="SuperMarioBros-Nes", help='游戏名称')
    parser.add_argument("--trained_model",    type=str,   default=None, help='预训练模型')
    parser.add_argument('--lr',               type=float, default=1e-4, help='模型的学习率')
    parser.add_argument('--gamma',            type=float, default=0.9,  help='奖励折扣率')
    parser.add_argument('--tau',              type=float, default=1.0,  help='GAE参数')
    parser.add_argument('--beta',             type=float, default=0.01, help='熵权')
    parser.add_argument('--epsilon',          type=float, default=0.2,  help='剪切替代目标参数')
    parser.add_argument('--batch_size',       type=int,   default=16,   help='训练数据的批量大小')
    parser.add_argument('--num_epochs',       type=int,   default=10,   help='每次采样训练多少轮')
    parser.add_argument("--num_local_steps",  type=int,   default=512,  help='每次采样的次数')
    parser.add_argument("--num_processes",    type=int,   default=16,   help='使用多少条线程启动游戏')
    parser.add_argument("--saved_path",       type=str,   default="models", help='保存模型的路径')
    parser.add_argument("--show_play",        type=bool,  default=False, help='是否显示评估游戏的界面，终端无法使用')
    args = parser.parse_args()
    return args


def train(args):
    # 固定初始化状态
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # 创建保存模型的文件夹
    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)
    # 创建多进程的游戏环境
    envs = MultipleEnvironments(args.game, args.num_processes)
    # 创建模型
    model = PPO(envs.num_states, envs.num_actions)
    # 加载预训练模型
    if args.trained_model is not None:
        model.load_state_dict(torch.load(args.trained_model))
    # 使用 GPU训练
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    # 为游戏评估单独开一个进程
    mp = _mp.get_context("spawn")
    process = mp.Process(target=eval, args=(args, model, envs.num_states, envs.num_actions))
    process.start()
    # 创建优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 刚开始给每个进程的游戏执行初始化
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    # 获取游戏初始的界面
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0
    while True:
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        # 执行游戏获取数据
        for _ in range(args.num_local_steps):
            states.append(curr_states)
            # 执行预测
            logits, value = model(curr_states)
            # 计算每个动作的概率值
            policy = F.softmax(logits, dim=1)
            # 根据每个标签的概率随机生成符合概率的标签
            old_m = Categorical(policy)
            action = old_m.sample()
            # 记录预测数据
            actions.append(action)
            values.append(value.squeeze())
            # 计算损失使用
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            # 向各个进程游戏发送动作
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
            # 将多进程的游戏数据打包
            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            # 进行数据转换
            state = torch.from_numpy(np.concatenate(state, 0))
            # 转换为pytorch数据
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            # 记录预测数据
            rewards.append(reward)
            dones.append(done)
            curr_states = state
        # 根据上面最后的图像预测
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * args.gamma * args.tau
            gae = gae + reward + args.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        total_losses = []
        for i in range(args.num_epochs):
            indice = torch.randperm(args.num_local_steps * args.num_processes)
            for j in range(args.batch_size):
                batch_indices = indice[
                                int(j * (args.num_local_steps * args.num_processes / args.batch_size)): int((j + 1) * (
                                        args.num_local_steps * args.num_processes / args.batch_size))]
                # 根据拿到的图像执行预测
                logits, value = model(states[batch_indices])
                # 计算每个动作的概率值
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                # 计算损失
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) *
                                                   advantages[batch_indices]))
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - args.beta * entropy_loss
                # 计算梯度
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_losses.append(float(total_loss))
        print("Episode: {}. Total loss: {:.4f}".format(curr_episode, np.mean(total_losses)))
        torch.save(model.state_dict(), "{}/model_{}.pth".format(args.saved_path, args.game))


if __name__ == "__main__":
    args = get_args()
    train(args)
