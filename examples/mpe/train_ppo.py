""""""
import json
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.envs.wrappers import GIFWrapper # 用于生成gif


def train(file_path):
    # create environment
    env_num = 12
    env = make(  
        "simple_spread",
        env_num=env_num,
        asynchronous=True,
    )
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg, device="cuda")
    # initialize the trainer   
    agent = Agent(net, use_wandb=False)
    if state == 2 or state == 3:
        agent.load(f"C:/Users/abing/OneDrive/基于强化学习的连续空间下多行人疏散路径选择模型/结果数据/出口{exit}/单人/{state-1}阶段/")
    # start training, set total number of training steps to 5000000
    agent.train(total_time_steps=3e7)
    env.close()
    agent.save(file_path)
    return agent

def eval(file_path):
    # 创建 MPE 环境
    env_num = 1000
    env = make( "simple_spread", env_num=env_num)
    # 使用GIFWrapper，用于生成gif
    # env = GIFWrapper(env, "./1AGENT_1EXIT_ONE.gif")
    agent = Agent(Net(env)) # 创建 智能体
    # 加载训练好的模型
    agent.load(file_path)
    # 开始测试
    obs, _ = env.reset(seed = 2)
    data = []
    # logging.basicConfig(filename="output.log", level=logging.INFO)
    while True:
        # 智能体根据 observation 预测下一个动作
        action, _ = agent.act(obs)
        obs, r, done, info = env.step(action)
        data.append(deepcopy(info))
        # logging.info(info)
        if done.all():
            break
        
    env.close()

    agent_pos, agent_vel = data_clean(data, env_num)
    agent_pos.to_csv(file_path + 'agent_pos.csv', index=False)
    agent_vel.to_csv(file_path + 'agent_vel.csv', index=False)
    save_draw(agent_pos, file_path, env_num)


def data_clean(data, env_num):

    agent_pos = []
    agent_vel = []
    agent_reward = []
    for i in range(0, len(data[0])):
        # 这个-1是因为最后一行是final，不是agent_pos之类的信息
        for j in range(0, len(data)-1):
            agent_pos.append(data[j][i][0]['agent_pos'][0])
            agent_pos.append(data[j][i][0]['agent_pos'][1])
            agent_vel.append(data[j][i][0]['agent_vel'][0])
            agent_vel.append(data[j][i][0]['agent_vel'][1])
            agent_reward.append(data[j][i][0]['individual_reward'])


    num_agents = env_num * 1
    num_steps = len(agent_pos) // num_agents

    all_agent_vel = np.zeros([num_steps//2, num_agents*2])
    all_agent_pos = np.zeros([num_steps//2, num_agents*2])

    for i, pos in enumerate(agent_pos):
        agent_id = i // num_steps * 2
        agent_time = i % num_steps
        if not agent_time % 2:
            all_agent_pos[agent_time//2][agent_id] = pos
        else:
            all_agent_pos[agent_time//2][agent_id + 1] = pos

    for i, vel in enumerate(agent_vel):
        agent_id = i // num_steps * 2
        agent_time = i % num_steps
        if not agent_time % 2:
            all_agent_vel[agent_time//2][agent_id] = vel
        else:
            all_agent_vel[agent_time//2][agent_id + 1] = vel


    df1 = pd.DataFrame(all_agent_pos)
    df2 = pd.DataFrame(all_agent_vel)
    return df1, df2

def save_draw(agent_pos, file_path, env_num):
    for i in range(2*env_num):
        if not i % 2:
            plt.plot(agent_pos[i], agent_pos[i+1])

    plt.xlim([-20, 23])
    plt.ylim([-6, 6])
    plt.savefig(file_path + 'figure.png')

if __name__ == "__main__":
    for i in range(1, 4):
        for j in range(1, 4):
            exit = i
            state = j
            print(exit, state)
            # 将参数信息保存在本地，通过本地调用的方式让simple_spread知道自己目前的状态
            df = pd.DataFrame([exit, state])
            df.to_csv('para.csv', index=False)
            file_path = f"C:/Users/abing/OneDrive/基于强化学习的连续空间下多行人疏散路径选择模型/结果数据/出口{exit}/单人/{state}阶段/"
            agent = train(file_path)
            eval(file_path)