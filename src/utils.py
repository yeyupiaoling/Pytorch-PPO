from collections import deque

import torch
import torch.nn.functional as F

from src.env import create_train_env
from src.model import PPO


def eval(args, global_model, num_states, num_actions):
    # 固定初始化状态
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # 创建游戏动作
    env = create_train_env(args.game)
    # 获取网络模型
    local_model = PPO(num_states, num_actions)
    # 判断是否可以使用GPU
    if torch.cuda.is_available():
        local_model.cuda()
    # 切换为评估状态
    local_model.eval()
    # 将图像转换为Pytorch的数据类型
    state = torch.from_numpy(env.reset())
    # 一开始就更新模型参数
    done = True
    curr_step = 0
    # 执行动作的容器
    actions = deque(maxlen=args.max_actions)
    while True:
        # 显示界面
        if args.show_eval_play is not None and args.show_eval_play:
            env.render()
        curr_step += 1
        # 使用GPU计算
        if torch.cuda.is_available():
            state = state.cuda()
        # 每结束一次就更新模型参数
        if done:
            local_model.load_state_dict(global_model.state_dict())
        # 预测动作概率和评估值
        logits, value = local_model(state)
        # 获取动作的序号
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        # 执行游戏
        state, reward, done, info = env.step(action)
        # 记录动作
        actions.append(action)
        if curr_step > args.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        # 重置游戏状态
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        # 转换每一步都游戏状态
        state = torch.from_numpy(state)
