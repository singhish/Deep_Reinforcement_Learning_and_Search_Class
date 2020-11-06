from typing import List
import torch
from torch import nn
from environments.environment_abstract import Environment, State
import numpy as np

import random; random.seed(0)


def relu_forward(inputs):
    inputs[inputs < 0] = 0
    return inputs


def relu_backward(grad, inputs):
    return grad


def linear_forward(inputs, weights, biases):
    return (inputs @ weights.T) + biases


def linear_backward(grad, inputs, weights, biases):
    weights_grad = grad.T @ inputs
    biases_grad = np.sum(grad, axis=0)
    inputs_grad = grad @ weights
    return weights_grad, biases_grad, inputs_grad


def get_dqn():

    class DQN(nn.Module):

        def __init__(self):
            super(DQN, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 75),
                nn.ReLU(),
                nn.Linear(75, 50),
                nn.ReLU(),
                nn.Linear(50, 4)
            )

        def forward(self, x):
            x = x.float()
            x = self.layers(x)
            return x


    dqn = DQN()
    return dqn


def deep_q_learning_step(env: Environment, state: State, dqn: nn.Module, dqn_target: nn.Module, epsilon: float,
                         discount: float, batch_size: int, optimizer, device, replay_buffer: List):
    # get action
    dqn.eval()
    if random.random() < epsilon:
        action = random.choice(env.get_actions())
    else:
        state_nnet = torch.tensor(env.state_to_nnet_input(state), device=device)
        action = torch.argmax(dqn(state_nnet)).item()

    # get transition
    state_next, reward = env.sample_transition(state, action)

    # add to replay buffer
    replay_buffer.append((state, action, reward, state_next))

    # sample from replay buffer and train
    batch_idxs = np.random.randint(len(replay_buffer), size=batch_size)

    states_nnet_np = np.concatenate([env.state_to_nnet_input(replay_buffer[idx][0]) for idx in batch_idxs], axis=0)
    actions_np = np.array([replay_buffer[idx][1] for idx in batch_idxs])
    rewards_np = np.array([replay_buffer[idx][2] for idx in batch_idxs])

    states_next = [replay_buffer[idx][3] for idx in batch_idxs]
    states_next_nnet_np = np.concatenate([env.state_to_nnet_input(replay_buffer[idx][3]) for idx in batch_idxs], axis=0)
    is_terminal_np = np.array([env.is_terminal(state_next) for state_next in states_next])

    states_nnet = torch.tensor(states_nnet_np, device=device)
    actions = torch.unsqueeze(torch.tensor(actions_np, device=device), 1)
    rewards = torch.tensor(rewards_np, device=device)
    states_next_nnet = torch.tensor(states_next_nnet_np, device=device)
    is_terminal = torch.tensor(is_terminal_np, device=device)

    # train DQN
    dqn.train()
    optimizer.zero_grad()

    # compute target
    target = torch.unsqueeze(
        torch.tensor(
            [(r if is_ter else r + discount * torch.max(dqn_target(s_p)))
             for r, s_p, is_ter
             in zip(rewards, states_next_nnet, is_terminal)]), 1).double()

    # get output of dqn
    output = dqn(states_nnet).gather(1, actions).double()

    # loss
    loss = nn.MSELoss()(output, target)

    # backpropagation
    loss.backward()

    # optimizer step
    optimizer.step()

    return state_next, dqn, replay_buffer
