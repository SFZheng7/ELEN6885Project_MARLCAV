import random
from collections import namedtuple
import numpy as np

Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "dones"))


# Extend Experience to include old_action_log_probs
ExperienceClip = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "dones", "old_action_log_probs"))
ExperienceA2C = namedtuple("Experience",
                        ("states", "actions", "rewards", "policies", "action_masks", "next_states", "dones"))

class ReplayMemoryA2C(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, policies, action_masks, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = ExperienceA2C(state, action, reward, policies, action_masks, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, policies, action_masks, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s,a,r,pi,am, n_s,d in zip(states, actions, rewards, policies, action_masks, next_states, dones):
                    self._push_one(s, a, r, pi, am, n_s, d)
            else:
                for s,a,r, pi, am in zip(states, actions, rewards, policies, action_masks):
                    self._push_one(s, a, r, pi, am)
        else:
            self._push_one(states, actions, rewards, policies, action_masks, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = ExperienceA2C(*zip(*transitions))

        # reset the memory
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)
    

class OnPolicyReplayMemoryClip(object):
    """
    Replay memory buffer for On-Policy algorithms like PPO-Clip
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None, old_action_log_prob=None):
        # Standardize input to NumPy arrays
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        state = np.array(state, dtype=np.float32)
        if next_state is not None:
            next_state = np.array(next_state, dtype=np.float32)
        self.memory[self.position] = ExperienceClip(state, action, reward, next_state, done, old_action_log_prob)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None, old_action_log_probs=None):
        # Validate input consistency
        assert len(states) == len(actions) == len(rewards), "Inconsistent input lengths!"
        if next_states is not None:
            assert len(next_states) == len(states), "Inconsistent next_states length!"
        if dones is not None:
            assert len(dones) == len(states), "Inconsistent dones length!"
        if old_action_log_probs is not None:
            assert len(old_action_log_probs) == len(states), "Inconsistent old_action_log_probs length!"

        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d, log_prob in zip(states, actions, rewards, next_states, dones, old_action_log_probs):
                    self._push_one(s, a, r, n_s, d, log_prob)
            else:
                for s, a, r, log_prob in zip(states, actions, rewards, old_action_log_probs):
                    self._push_one(s, a, r, old_action_log_prob=log_prob)
        else:
            self._push_one(states, actions, rewards, next_states, dones, old_action_log_probs)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = ExperienceClip(*zip(*transitions))

        # Convert batch fields to NumPy arrays
        batch = batch._replace(
            states=np.array(batch.states, dtype=np.float32),
            actions=np.array(batch.actions, dtype=np.float32),
            rewards=np.array(batch.rewards, dtype=np.float32),
            next_states=np.array(batch.next_states, dtype=np.float32) if batch.next_states[0] is not None else None,
            dones=np.array(batch.dones, dtype=np.float32),
            old_action_log_probs=np.array(batch.old_action_log_probs, dtype=np.float32)
        )
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)



class OnPolicyReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s,a,r,n_s,d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s,a,r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        # reset the memory
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        return batch

    def __len__(self):
        return len(self.memory)