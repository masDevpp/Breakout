import os
import tensorflow as tf
import gym
import numpy as np
from PIL import Image
import time

LOG_DIR = os.path.join(os.getcwd(), "main2_log")
slim = tf.contrib.slim

class QAgent():
    def __init__(self, sess, input_shape, num_actions, epsilon_initial, epsilon_end, gamma):
        # input_shape should shape [height, width, num state history]
        # gamma is reward discount ratio
        self.sess = sess
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.epsilon_initial = epsilon_initial
        self.epsilon_end = epsilon_end
        self.gamma = gamma

        self.build_prediction_network()
        self.build_target_network()
        self.build_training_operator()
        self.summary = tf.summary.merge_all()
    
    def build_prediction_network(self):
        #with tf.device('/cpu:0'):
        with tf.variable_scope("prediction_network"):
            self.prediction_input_placeholder = tf.placeholder("float", [None] + self.input_shape)

            conv_0 = slim.conv2d(self.prediction_input_placeholder, 16, 8, 4, scope="conv_0")
            conv_1 = slim.conv2d(conv_0, 32, 4, 2, scope="conv_1")
            flatten = slim.flatten(conv_1)
            fc_0 = slim.fully_connected(flatten, 256, scope="fc_0")
            self.output = slim.fully_connected(fc_0, self.num_actions, activation_fn=None, scope="q_values")
        
    def build_target_network(self):
        #with tf.device('/cpu:0'):
        with tf.variable_scope("target_network"):
            self.target_net_input_placeholder = tf.placeholder("float", [None] + self.input_shape)

            conv_0 = slim.conv2d(self.target_net_input_placeholder, 16, 8, 4, scope="conv_0")
            conv_1 = slim.conv2d(conv_0, 32, 4, 2, scope="conv_1")
            flatten = slim.flatten(conv_1)
            fc_0 = slim.fully_connected(flatten, 256, scope="fc_0")
            self.output_target = slim.fully_connected(fc_0, self.num_actions, activation_fn=None, scope="q_values")

    def build_training_operator(self):
        #with tf.device('/cpu:0'):
        with tf.variable_scope("training_operator"):
            self.target_q_placeholder = tf.placeholder("float", [None])
            self.reward_placeholder = tf.placeholder("float", [None])
            self.action_placeholder = tf.placeholder("int32", [None])
            self.learning_rate_placeholder = tf.placeholder("float")

            output_flat = tf.reshape(self.output, [-1])

            index = tf.range(tf.shape(self.output)[0]) * tf.shape(self.output)[1]
            index = index + self.action_placeholder

            selected_output = tf.gather(output_flat, index)

            target_q = self.reward_placeholder + self.gamma * self.target_q_placeholder

            self.loss = tf.reduce_mean(tf.square(selected_output - target_q))

            tf.summary.scalar("loss", self.loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
            self.train_op = optimizer.minimize(self.loss)

    def predict_action(self, state):
        # Predict Q value for single state
        q = self.sess.run(self.output, feed_dict={self.prediction_input_placeholder:[state]})
        # Return index of max Q
        return np.argmax(q)
    
    def predict_action_with_epsilon_greedy(self, state, epsilon_ratio):
        # epsilon_ratio indicates progress of epsilon decay, start from 0.0 to 1.0
        if epsilon_ratio > 1.0: epsilon_ratio = 1.0
        elif epsilon_ratio < 0.0: epsilon_ratio = 0.0
        #epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * ((1.0 - epsilon_ratio)**2)
        epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * (1.0 - epsilon_ratio)

        if np.random.random() > epsilon:
            action = self.predict_action(state)
            return action
        else:
            return np.random.randint(self.num_actions)

    def train(self, state, action, reward, state_next, tarminal, learning_rate=0.001):
        # All argument should shape like [batch, ...]

        # Get target Q value
        # Make numpy target Q array to avoid train target network
        target_q = self.sess.run(self.output_target, feed_dict={self.target_net_input_placeholder:state_next})
        target_q = np.max(target_q, axis=1)

        # Q is zero if terminal
        tarminal = np.array(tarminal).astype("int")
        target_q = target_q * (1 - tarminal)

        feed_dict = {
            self.prediction_input_placeholder: state,
            self.target_q_placeholder: target_q,
            self.reward_placeholder: reward,
            self.learning_rate_placeholder: learning_rate,
            self.action_placeholder: action
        }

        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)

        return loss, summary
    
    def update_target_network(self):
        pred_net_variables = tf.global_variables(scope="prediction_network")
        target_net_variables = tf.global_variables(scope="target_network")

        if len(pred_net_variables) != len(target_net_variables):
            assert "Variable size unmatch"

        for p, t in zip(pred_net_variables, target_net_variables):
            if p.shape != t.shape:
                assert "Variable shape unmatch"
            self.sess.run(t.assign(p))

class EpisodeMemory():
    def __init__(self, global_memory, num_hold_episode=30000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.tarminals = []
        self.discounted_rewards = []

        self.global_memory = global_memory
        self.num_hold_episode = num_hold_episode
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.tarminals = []
        self.discounted_rewards = []

    def remove_old_episode(self):
        if len(self.states) > self.num_hold_episode:
            self.states = self.states[-self.num_hold_episode:]
            self.actions = self.actions[-self.num_hold_episode:]
            self.rewards = self.rewards[-self.num_hold_episode:]
            self.states_next = self.states_next[-self.num_hold_episode:]
            self.tarminals = self.tarminals[-self.num_hold_episode:]
            self.discounted_rewards = self.discounted_rewards[-self.num_hold_episode:]

    def add_episode(self, episode):
        if not self.global_memory:
            assert "Wrong operation"
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.states_next += episode.states_next
        self.tarminals += episode.tarminals
        self.discounted_rewards += episode.discounted_rewards

        self.remove_old_episode()

    def add_one_step(self, state, action, reward, state_next, tarminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_next.append(state_next)
        self.tarminals.append(tarminal)

    def calculate_discounted_rewards(self, discount_rate):
        if self.global_memory:
            assert "Wrong operation"

        self.discounted_rewards = [0 for _ in range(len(self.rewards))]
        self.discounted_rewards[-1] = self.rewards[-1]

        for i in range(len(self.rewards) - 2, -1, -1):
            self.discounted_rewards[i] = self.rewards[i] + self.discounted_rewards[i + 1] * discount_rate

    def has_enough_memory(self):
        return len(self.states) >= self.num_hold_episode
    
    def get_batch(self, batch_size):
        states_batch = []
        actions_batch = []
        rewards_batch = []
        states_next_batch = []
        tarminals_batch = []

        index = np.random.randint(0, len(self.states), batch_size)

        for i in index:
            states_batch.append(self.states[i])
            actions_batch.append(self.actions[i])
            rewards_batch.append(self.rewards[i])
            states_next_batch.append(self.states_next[i])
            tarminals_batch.append(self.tarminals[i])

        return states_batch, actions_batch, rewards_batch, states_next_batch, tarminals_batch

class StateHoler():
    def __init__(self, num_states, initial_state, skip_frame, do_preprocess):
        if skip_frame < 1: assert "Invalid param"

        self.num_states = num_states
        self.skip_frame = skip_frame

        self.do_preprocess = do_preprocess
        if self.do_preprocess:
            initial_state = self.preprocess_state(initial_state)
        
        self.states = np.array([initial_state for _ in range(self.num_states * self.skip_frame)])
    
    def preprocess_state(self, state):
        # Resize and convert to gray scale
        new_state = np.array(Image.fromarray(state).resize((84, 110), Image.ANTIALIAS).convert("L"))
        # Crop
        new_state = new_state[14:-4,:]

        return new_state

    def add_state(self, new_state):
        if self.do_preprocess:
            new_state = self.preprocess_state(new_state)

        self.states = np.concatenate((self.states, [new_state]))
        self.states = self.states[-(self.num_states*self.skip_frame):]
    
    def get_state(self):
        # Return shape [height, width, channel]
        return self.states[self.skip_frame-1::self.skip_frame].transpose([1,2,0])

def main():
    num_states_to_hold = 4
    skip_frame = 1
    epsilon_initial = 1.0
    epsilon_end = 0.1
    epsilon_decay_end_episode = 1000000
    training_frequency = 4
    target_network_update_frequency = 10000
    batch_size = 32
    discount_rate = 0.99
    memory_size = 512#20000
    learning_rate = 0.001

    should_render = True

    env = gym.make("Breakout-v0")
    num_actions = env.action_space.n

    # Initialize StateHolder once to check state shape
    state = env.reset()
    state_holder = StateHoler(num_states_to_hold, state, skip_frame, True)
    num_states = list(state_holder.get_state().shape)

    with tf.Session() as sess:#tf.Session(config=tf.ConfigProto(log_device_placement=True))
        agent = QAgent(sess, num_states, num_actions, epsilon_initial, epsilon_end, discount_rate)

        episode_count_variable = tf.Variable(0, trainable=False, name="episode_count")
        global_step_variable = tf.Variable(0, trainable=False, name="global_step")
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load checkpoint " + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initialize variables")
            sess.run(tf.global_variables_initializer())
            agent.update_target_network()

        episode_count = sess.run(episode_count_variable)
        global_step = sess.run(global_step_variable)

        episode_memory = EpisodeMemory(True, memory_size)

        state = env.reset()
        state_holder = StateHoler(num_states_to_hold, state, skip_frame, True)
        
        episode_reward = 0
        start_time = time.time()
        terminal = True
        loss = 0

        while True:
            if should_render: env.render()

            action = agent.predict_action_with_epsilon_greedy(state_holder.get_state(), global_step / epsilon_decay_end_episode)                
            if terminal: action = 1
            
            state_next, reward, terminal, info_dict = env.step(action)

            state_initial = state_holder.get_state()
            state_holder.add_state(state_next)
            
            episode_memory.add_one_step(state_initial, action, reward, state_holder.get_state(), terminal)
            episode_reward += reward

            if episode_memory.has_enough_memory():
                if global_step % training_frequency == 0:
                    s, a, r, s_next, t = episode_memory.get_batch(batch_size)
                    loss, summary = agent.train(s, a, r, s_next, t, learning_rate=learning_rate)
                
                if global_step % target_network_update_frequency == 0:
                    print("Update target network")
                    agent.update_target_network()

            global_step += 1

            if terminal:
            
                current_time = time.time()
                print("Ep " + str(episode_count) + ", EpReward " + str(episode_reward) + ", Elapse " + format(current_time - start_time, ".2f") + " LastLoss " + format(loss, ".4f") + ", EpsilonProg " + format(global_step / epsilon_decay_end_episode, ".2f"))
                start_time = current_time
                
                state = env.reset()
                episode_reward = 0
                episode_count += 1

                if episode_memory.has_enough_memory():
                    sess.run(episode_count_variable.assign(episode_count))
                    sess.run(global_step_variable.assign(global_step))
                    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step = episode_count)
                    summary_writer.add_summary(summary, episode_count)


if __name__ == "__main__":
    main()
