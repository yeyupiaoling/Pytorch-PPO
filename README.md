# Pytorch-PPO
本项目是基于[Super-mario-bros-PPO-pytorch](https://github.com/uvipen/Super-mario-bros-PPO-pytorch) 复现的PPO，使用PPO模型训练Gym Retro游戏，支持训练各种游戏，如超级马里奥，雪人兄弟，魂斗罗等等。

# 项目结构
```shell script
PPO-retro/
├── actions.py        定义每个游戏的动作
├── discretizer.py    定义有点动作的工具类
├── env.py            用于游戏环境和多线程游戏环境
├── infer.py          使用训练好的模型进行推理
├── model.py          模型结构
├── retro_util.py     retro游戏动作和图像处理
├── retrowrapper.py   用于支持多线程retro游戏环境
├── test_env.py       测试游戏环境
├── train.py          训练PPO模型
└── utils.py          用于评估和输出参数
```

# retro游戏模拟器
Gym Retro可的游戏转变为Gym环境以进行强化学习，并附带约1000种游戏的集成功能。它使用各种支持Libretro API的仿真器，从而使添加新仿真器变得相当容易。
支持平台：
- Windows 7、8、10
- macOS 10.13（High Sierra），10.14（Mojave）
- Linux（许多Linux1）

支持的Python：
- 3.6
- 3.7
- 3.8

每个游戏集成都具有列出游戏中变量的存储位置的文件，基于这些变量的奖励功能，情节结束条件。

**请注意** ，不包含ROM，必须自己获取它们。

1. 安装retro
```shell
pip install gym-retro -i https://mirrors.aliyun.com/pypi/simple/
```

2. 查看retro支持的游戏，执行这两行代码会输出retro支持的游戏。
```python
import retro
retro.data.list_games()
```

3. 导入游戏的ROM文件，我们安装了retro不是就能够直接使用的，因为retro不会提供游戏的ROM文件，所以需要我们自己去下载和导入。跟着笔者下载游戏的ROM文件并导入到Python环境中。[这里](https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed) 提供了很多游戏的ROM，笔者下载的是[Nintendo - Nintendo Entertainment System.zip](https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Nintendo%20-%20Nintendo%20Entertainment%20System.zip) ，导入命令如下：
```shell script
python -m retro.import /ROM的文件夹/
```
**特别注意：** 如果ROM带有`.bin`扩展名，需要将其重命名为具有该系统正确的扩展名。如Atari类型的，需要改成`.a26`，以便`python -m retro.import`成功！
```shell script
.md： 世嘉创世纪（Mega Drive）
.sfc：超级任天堂娱乐系统
.nes：任天堂娱乐系统
.a26：Atari 2600
.gb： 任天堂游戏男孩
.gba：任天堂Game Boy Advance
.gbc：任天堂游戏男孩Color
.gg： 世嘉游戏装备
.pce：NEC TurboGrafx-16
.sms：世嘉Master System
```

4. 测试游戏环境，通过执行`test_env.py`可以测试游戏的环境。
```python
import cv2
import numpy as np

from src.env import create_train_env


def main():
    # 获取游戏
    env = create_train_env(game="SuperMarioBros-Nes")
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
            obs = env.reset()


if __name__ == "__main__":
    main()
```

# 训练

1. 首先安装PaddlePaddle以及相关库。
```shell
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

2. 执行训练，通过game参数可以指定想要选择的游戏，前提是retro得支持这个游戏并且已经导入了ROM文件。
```shell
python train.py --game=SuperMarioBros-Nes
```

3. 验证通过的游戏如下，最新的验证通过的游戏请参考本项目的Github文档。
```shell script
SuperMarioBros-Nes
SnowBrothers-Nes
```

4. 自行训练其他游戏，理论情况下只需要更改游戏名称就可以正常训练，但是为了模型更好的拟合，最好是可以修改一下游戏可执行的动作，动作在`actions.py`。

## 预测
预测程序会使用训练时保存的最好得分模型进行预测，这个预测程序需要在界面环境下执行，如果要在终端下执行，需要注释`env.render()`这行代码。
```shell
python infer.py --game=SuperMarioBros-Nes
```

