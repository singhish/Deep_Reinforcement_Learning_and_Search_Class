from typing import Tuple, List, Dict
from environments.environment_abstract import Environment, State


def value_iteration_step(env: Environment, states: List[State], state_vals: Dict[State, float],
                         discount: float) -> Tuple[float, Dict[State, float]]:
    change: float = 0

    for s in states:
        v = state_vals[s]

        max_v = -float('inf')
        for a in env.get_actions():
            reward, next_states, t_probs = env.state_action_dynamics(s, a)
            bellman_term = reward + discount*sum([t_probs[i] * state_vals[s_pr] for i, s_pr in enumerate(next_states)])
            if bellman_term > max_v:
                max_v = bellman_term

        state_vals[s] = max_v

        change = max(change, abs(v - state_vals[s]))

    return change, state_vals


def get_action(env: Environment, state: State, state_vals: Dict[State, float], discount: float) -> int:
    action: int = 0  # DUMMY VALUE

    max_v = -float('inf')
    for a in env.get_actions():
        reward, next_states, t_probs = env.state_action_dynamics(state, a)
        bellman_term = reward + discount*sum([t_probs[i] * state_vals[s_pr] for i, s_pr in enumerate(next_states)])
        if bellman_term > max_v:
            max_v = bellman_term
            action = a

    return action
