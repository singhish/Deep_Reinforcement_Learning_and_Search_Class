from typing import Tuple, List, Dict
from environments.environment_abstract import Environment, State
import random; random.seed(0)


def policy_evaluation_step(env: Environment, states: List[State], state_vals: Dict[State, float],
                           policy: Dict[State, List[float]], discount: float) -> Tuple[float, Dict[State, float]]:
    change: float = 0.0

    for s in states:
        v = state_vals[s]

        new_v = 0
        for a in env.get_actions():
            r, next_states, t_probs = env.state_action_dynamics(s, a)
            new_v += policy[s][a] * (r + discount * sum(p * state_vals[s_pr] for s_pr, p in zip(next_states, t_probs)))
        state_vals[s] = new_v

        change = max(change, abs(v - state_vals[s]))

    return change, state_vals


def q_learning_step(env: Environment, state: State, action_vals: Dict[State, List[float]], epsilon: float,
                    learning_rate: float, discount: float):
    if random.random() < epsilon:
        action = random.choice(env.get_actions())
    else:
        action = max(zip(env.get_actions(), action_vals[state]), key=lambda x: x[1])[0]

    state_next, r = env.sample_transition(state, action)

    action_vals[state][action] += learning_rate * (r + discount * max(
        action_vals[state_next][a] - action_vals[state][action] for a in env.get_actions()))

    return state_next, action_vals
