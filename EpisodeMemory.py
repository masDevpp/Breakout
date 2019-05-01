import numpy as np
from PIL import Image

class EpisodeMemory():
    def __init__(self, min_size, max_size, do_preprocess, state_seq_length, skip_frame, reward_filter_prob=0.0, reward_filter_min=0):
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.episode_rewards = []

        self.min_size = min_size
        self.max_size = max_size
        self.do_preprocess = do_preprocess
        self.state_seq_length = state_seq_length
        self.skip_frame = skip_frame
        self.reward_filter_prob = reward_filter_prob
        self.reward_filter_min = reward_filter_min
    
    def preprocess_state(self, state):
        # Resize and convert to gray scale
        new_state = np.array(Image.fromarray(state).resize((90, 102), Image.ANTIALIAS).convert("L"))
        # Crop
        new_state = new_state[14:-4,3:-3]

        #new_state = (new_state / 255).astype(np.float32)
        new_state = (new_state / 255).astype(np.float16)
        return new_state

    def remove_old_episode(self):
        if len(self.states) > self.max_size:
            self.states = list(self.states[-self.max_size:])
            self.actions = list(self.actions[-self.max_size:])
            self.rewards = list(self.rewards[-self.max_size:])
            self.terminals = list(self.terminals[-self.max_size:])
            self.episode_rewards = list(self.episode_rewards[-self.max_size:])
            
    def add_one_step(self, state, action, reward, terminal, episode_reward=None):
        if self.do_preprocess: state = self.preprocess_state(state)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        
        if episode_reward != None:
            self.episode_rewards.append(episode_reward)

    def has_enough_memory(self):
        return len(self.states) >= self.min_size
    
    def get_states(self, index):
        return_states = [np.zeros(self.states[0].shape) for _ in range(self.state_seq_length)]

        target_index = self.state_seq_length - 1
        for i, j in enumerate(list(range(index, index - (self.state_seq_length * self.skip_frame), -1))):
            if j < 0: break
            
            if i % self.skip_frame == 0:
                return_states[target_index] = self.states[j]
                target_index -= 1

            if self.terminals[j]:# and i != 0: 
                break
            
        return np.array(return_states).transpose([1,2,0])
    
    def get_last_states(self):
        return self.get_states(len(self.states) - 1)

    def get_batch(self, batch_size):
        states_batch = []
        actions_batch = []
        rewards_batch = []
        states_next_batch = []
        terminals_batch = []

        index = np.random.randint(1, len(self.states) - 1, batch_size)

        for i in index:
            states_batch.append(self.get_states(i))
            actions_batch.append(self.actions[i + 1])
            rewards_batch.append(self.rewards[i + 1])
            states_next_batch.append(self.get_states(i + 1))
            terminals_batch.append(self.terminals[i + 1])

        return states_batch, actions_batch, rewards_batch, states_next_batch, terminals_batch

    def get_batch2(self, batch_size):
        states_batch = []
        actions_batch = []
        rewards_batch = []
        states_next_batch = []
        terminals_batch = []
        episode_rewards_batch = []

        index = np.arange(len(self.states))
        np.random.shuffle(index)
        #index = np.random.randint(0, len(self.states), batch_size)

        for i in index:
            if i - 1 < self.state_seq_length or i + 1 > len(self.states) - 1: continue
            
            #if np.random.random() > 0.1:
            #    if self.rewards[i + 1] == 0 and self.terminals[i + 1] == False:
            #        continue
            if self.reward_filter_prob > 0.0:
                if self.reward_filter_prob > np.random.random():
                    if self.episode_rewards[i + 1] < self.reward_filter_min:
                        continue

            states_batch.append(self.get_states(i))
            actions_batch.append(self.actions[i + 1])
            rewards_batch.append(self.rewards[i + 1])
            states_next_batch.append(self.get_states(i + 1))
            terminals_batch.append(self.terminals[i + 1])
            episode_rewards_batch.append(self.episode_rewards[i + 1])

            if len(states_batch) == batch_size: break

        return states_batch, actions_batch, rewards_batch, states_next_batch, terminals_batch
