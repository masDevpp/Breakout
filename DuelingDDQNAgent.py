import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

class DuelingDDQNAgent():
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
        self.build_assign_operator()
        self.summary = tf.summary.merge_all()
    
    def build_prediction_network(self):
        with tf.variable_scope("prediction_network"):
            self.prediction_input_placeholder = tf.placeholder("float", [None] + self.input_shape, name="state")

            conv_0 = slim.conv2d(self.prediction_input_placeholder, 32, [8, 8], 4, scope="conv_0")
            conv_1 = slim.conv2d(conv_0, 64, [4, 4], 2, scope="conv_1")
            conv_2 = slim.conv2d(conv_1, 64, [3, 3], 1, scope="conv_2")
            
            flatten = slim.flatten(conv_2)
            
            fc_0_value = slim.fully_connected(flatten, 512, scope="fc_0_value")
            fc_0_advantage = slim.fully_connected(flatten, 512, scope="fc_0_advantage")
            
            self.value = slim.fully_connected(fc_0_value, 1, scope="value", activation_fn=None)
            self.advantage = slim.fully_connected(fc_0_advantage, self.num_actions, scope="advantage", activation_fn=None)

            advantage_ave = tf.reduce_mean(self.advantage, axis=1, keep_dims=True)

            self.output = self.value + (self.advantage - advantage_ave)
        
    def build_target_network(self):
        with tf.variable_scope("target_network"):
            self.target_input_placeholder = tf.placeholder("float", [None] + self.input_shape, name="target_state")

            conv_0 = slim.conv2d(self.target_input_placeholder, 32, [8, 8], 4, scope="conv_0", trainable=False)
            conv_1 = slim.conv2d(conv_0, 64, [4, 4], 2, scope="conv_1", trainable=False)
            conv_2 = slim.conv2d(conv_1, 64, [3, 3], 1, scope="conv_2", trainable=False)

            flatten = slim.flatten(conv_2)

            fc_0_value = slim.fully_connected(flatten, 512, scope="fc_0_value", trainable=False)
            fc_0_advantage = slim.fully_connected(flatten, 512, scope="fc_0_advantage", trainable=False)

            self.target_value = slim.fully_connected(fc_0_value, 1, scope="target_value", trainable=False, activation_fn=None)
            self.target_advantage = slim.fully_connected(fc_0_advantage, self.num_actions, scope="target_advantage", trainable=False, activation_fn=None)

            target_advantage_ave = tf.reduce_mean(self.target_advantage, axis=1, keep_dims=True)

            self.target_output = self.target_value + (self.target_advantage - target_advantage_ave)

    def build_training_operator(self):
        with tf.variable_scope("training_operator"):
            self.target_q_placeholder = tf.placeholder("float", [None])
            self.reward_placeholder = tf.placeholder("float", [None], name="reward")
            self.action_placeholder = tf.placeholder("int32", [None], name="action")
            self.learning_rate_placeholder = tf.placeholder("float", name="learning_rate")
            
            action_one_hot = tf.one_hot(self.action_placeholder, self.num_actions)
            self.selected_q = tf.reduce_sum(self.output * action_one_hot, axis=1)

            self.target_q = self.reward_placeholder + self.gamma * self.target_q_placeholder

            self.loss = tf.reduce_mean(tf.square(self.target_q - self.selected_q))

            tf.summary.scalar("loss", self.loss)

            #optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_placeholder)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
            self.train_op = optimizer.minimize(self.loss)

    def build_assign_operator(self):
        pred_net_variables = tf.global_variables(scope="prediction_network")
        target_net_variables = tf.global_variables(scope="target_network")

        if len(pred_net_variables) != len(target_net_variables):
            assert "Variable size unmatch"

        self.assign_operator = []

        for p, t in zip(pred_net_variables, target_net_variables):
            if p.shape != t.shape:
                assert "Variable shape unmatch"
            self.assign_operator.append(t.assign(p))

    def predict_action(self, state):
        # Predict Q value for single state
        q = self.sess.run(self.output, feed_dict={self.prediction_input_placeholder:[state]})
        q = q.reshape([-1])
        
        # Return index of max Q
        max_q = q[0]
        max_q_index = [0]
        for i in range(1, len(q)):
            if q[i] > max_q:
                max_q = q[i]
                max_q_index = [i]
            elif q[i] == max_q:
                max_q_index.append(i)

        return np.random.choice(max_q_index), q
        #return np.argmax(q)
    
    def predict_action_with_epsilon_greedy(self, state, epsilon_ratio):
        # epsilon_ratio indicates progress of epsilon decay, start from 0.0 to 1.0
        if epsilon_ratio > 1.0: epsilon_ratio = 1.0
        elif epsilon_ratio < 0.0: epsilon_ratio = 0.0
        
        #epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * ((1.0 - epsilon_ratio)**2)
        epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * (1.0 - epsilon_ratio)

        if np.random.random() > epsilon:
            action, q = self.predict_action(state)
            return action, q
        else:
            return np.random.randint(self.num_actions), np.zeros(self.num_actions)

    def train(self, state, action, reward, state_next, terminal, learning_rate=0.001):
        # All argument should shape like [batch, ...]
        
        # Get target Q value
        # Make numpy target Q array to avoid train target network
        next_q, target_q = self.sess.run(
            [self.output, self.target_output], 
            feed_dict={self.prediction_input_placeholder: state_next, self.target_input_placeholder:state_next}
            )
        
        next_action = np.argmax(next_q, axis=1)
        next_action_one_hot = np.eye(self.num_actions)[next_action]

        selected_target_q = np.sum(target_q * next_action_one_hot, axis=1)

        # Q is zero if terminal
        terminal = np.array(terminal).astype("int")
        selected_target_q = selected_target_q * (1 - terminal)

        feed_dict = {
            self.prediction_input_placeholder: state,
            self.target_q_placeholder: selected_target_q,
            self.reward_placeholder: reward,
            self.learning_rate_placeholder: learning_rate,
            self.action_placeholder: action
        }
        
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)

        return loss, summary
    
    def update_target_network(self):
        self.sess.run(self.assign_operator)
