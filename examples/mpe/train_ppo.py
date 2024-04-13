""""""

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.envs.wrappers import GIFWrapper # 用于生成gif


def train():
    # create environment
    env_num = 70
    env = make(
        "AnyLandmark",
        env_num=env_num,
        asynchronous=True,
    )
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg, device="cuda")
    # initialize the trainer
    agent = Agent(net, use_wandb=False)
    # start training, set total number of training steps to 5000000
    agent.train(total_time_steps=50000000)
    env.close()
    agent.save("./ppo_agent/")
    return agent


def evaluation(agent):
    render_model = "group_human"
    env_num = 9
    env = make(
        "AnyLandmark", render_mode=render_model, env_num=env_num, asynchronous=False
    )
    agent.load("./ppo_agent/")
    agent.set_env(env)
    obs, info = env.reset(seed=0)
    done = False
    step = 0
    total_reward = 0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
    print(f"total_reward: {total_reward}")
    env.close()

def eval():
    # 创建 MPE 环境
    env = make( "AnyLandmark", env_num=4)
    # 使用GIFWrapper，用于生成gif
    env = GIFWrapper(env, "test_AnyLandmark.gif")
    agent = Agent(Net(env)) # 创建 智能体
    # 加载训练好的模型
    agent.load('./ppo_agent/')
    # 开始测试
    obs, _ = env.reset()
    while True:
        # 智能体根据 observation 预测下一个动作
        action, _ = agent.act(obs)
        obs, r, done, info = env.step(action)
        if done.any():
            break
    env.close()


if __name__ == "__main__":
    agent = train()
    evaluation(agent)
