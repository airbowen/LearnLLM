import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # 构建一个多层感知机神经网络。
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    # 计算从每一步开始的未来累积奖励。
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # 创建环境，检查空间，获取观测和动作的维度
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "此示例仅适用于具有连续状态空间的环境。"
    assert isinstance(env.action_space, Discrete), \
        "此示例仅适用于具有离散动作空间的环境。"

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # 构建策略网络的核心
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # 定义获取策略分布的函数
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # 定义动作选择函数（从策略中采样得到整数动作）
    def get_action(obs):
        return get_policy(obs).sample().item()

    # 定义损失函数，其梯度用于策略梯度更新
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # 创建优化器
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # 用于训练策略的函数
    def train_one_epoch():
        # 用于记录日志的空列表
        batch_obs = []          # 观测值
        batch_acts = []         # 动作
        batch_weights = []      # 策略梯度中的奖励到达加权
        batch_rets = []         # 每个回合的累积奖励
        batch_lens = []         # 每个回合的步数

        # 重置每个回合的变量
        obs = env.reset()       # 初始观测来自初始分布
        done = False            # 环境信号，表明回合结束
        ep_rews = []            # 回合中累积奖励的列表

        # 渲染每个回合的第一帧（如果需要）
        finished_rendering_this_epoch = False

        # 使用当前策略在环境中收集经验
        while True:

            # 渲染环境
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # 保存观测值
            batch_obs.append(obs.copy())

            # 在环境中执行动作
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # 保存动作和奖励
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # 如果回合结束，记录回合信息
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # 每个动作logprob(a_t|s_t)的权重是从t开始的未来奖励到达
                batch_weights += list(reward_to_go(ep_rews))

                # 重置回合特定变量
                obs, done, ep_rews = env.reset(), False, []

                # 不再在本轮中渲染
                finished_rendering_this_epoch = True

                # 如果收集足够的经验，则结束经验收集循环
                if len(batch_obs) > batch_size:
                    break

        # 进行一次策略梯度更新
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # 训练循环
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\n使用奖励到达策略梯度的形式。\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
