import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

class DDQNAgent():
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
        self.build_training_operator2()
        self.build_assign_operator()
        self.summary = tf.summary.merge_all()
    
    def build_prediction_network(self):
        with tf.variable_scope("prediction_network"):
            self.prediction_input_placeholder = tf.placeholder("float", [None] + self.input_shape, name="state")

            conv_0 = slim.conv2d(self.prediction_input_placeholder, 16, [8, 8], 4, scope="conv_0")
            conv_1 = slim.conv2d(conv_0, 32, [4, 4], 2, scope="conv_1")
            flatten = slim.flatten(conv_1)
            fc_0 = slim.fully_connected(flatten, 256, scope="fc_0")
            self.output = slim.fully_connected(fc_0, self.num_actions, activation_fn=None, scope="q_values")
        
    def build_target_network(self):
        with tf.variable_scope("target_network"):
            self.target_net_input_placeholder = tf.placeholder("float", [None] + self.input_shape, name="target_state")

            conv_0 = slim.conv2d(self.target_net_input_placeholder, 16, [8, 8], 4, scope="conv_0", trainable=False)
            conv_1 = slim.conv2d(conv_0, 32, [4, 4], 2, scope="conv_1", trainable=False)
            flatten = slim.flatten(conv_1)
            fc_0 = slim.fully_connected(flatten, 256, scope="fc_0", trainable=False)
            self.output_target = slim.fully_connected(fc_0, self.num_actions, activation_fn=None, scope="q_values", trainable=False)

    def build_training_operator(self):
        with tf.variable_scope("training_operator"):
            self.reward_placeholder = tf.placeholder("float", [None], name="reward")
            self.action_placeholder = tf.placeholder("int32", [None], name="action")
            self.learning_rate_placeholder = tf.placeholder("float", name="learning_rate")
            self.terminal_placeholder = tf.placeholder("float", name="terminal")

            # Select Q value
            output_flat = tf.reshape(self.output, [-1])

            index = tf.range(tf.shape(self.output)[0]) * tf.shape(self.output)[1]
            index = index + self.action_placeholder

            self.selected_output = tf.gather(output_flat, index)
            
            # Select target Q value
            target_action = tf.reshape(tf.argmax(self.output, axis=1, output_type=tf.int32), [-1])
            target_output_flat = tf.reshape(self.output_target, [-1])

            self.target_index = tf.range(tf.shape(self.output_target)[0]) * tf.shape(self.output_target)[1]
            self.target_index = self.target_index + target_action
            
            self.selected_target_output = tf.gather(target_output_flat, self.target_index)
            self.selected_target_output = self.selected_target_output * (1 - self.terminal_placeholder)
            self.selected_target_output_discount = self.reward_placeholder + self.gamma * self.selected_target_output

            self.loss = tf.losses.mean_squared_error(self.selected_target_output_discount, self.selected_output)

            tf.summary.scalar("loss", self.loss)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_placeholder)
            self.train_op = optimizer.minimize(self.loss)
    
    def build_training_operator2(self):
        with tf.variable_scope("training_operator"):
            self.target_q_placeholder = tf.placeholder("float", [None])
            self.reward_placeholder = tf.placeholder("float", [None], name="reward")
            self.action_placeholder = tf.placeholder("int32", [None], name="action")
            self.learning_rate_placeholder = tf.placeholder("float", name="learning_rate")
            
            output_flat = tf.reshape(self.output, [-1])

            index = tf.range(tf.shape(self.output)[0]) * tf.shape(self.output)[1]
            index = index + self.action_placeholder

            selected_output = tf.gather(output_flat, index)

            target_q = self.reward_placeholder + self.gamma * self.target_q_placeholder

            self.loss = tf.losses.mean_squared_error(target_q, selected_output)

            tf.summary.scalar("loss", self.loss)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_placeholder)
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

        feed_dict = {
            self.prediction_input_placeholder: state,
            self.target_net_input_placeholder: state_next,
            self.reward_placeholder: reward,
            self.learning_rate_placeholder: learning_rate,
            self.action_placeholder: action,
            self.terminal_placeholder: terminal
        }

        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)

        return loss, summary
    
    def train2(self, state, action, reward, state_next, terminal, learning_rate=0.001):
        # All argument should shape like [batch, ...]
        
        # Get target Q value
        # Make numpy target Q array to avoid train target network
        target_q, next_output = self.sess.run([self.output_target, self.output], feed_dict={self.prediction_input_placeholder:state_next, self.target_net_input_placeholder:state_next})
        
        target_q_flat = target_q.reshape([-1])
        target_q_argmax = np.argmax(next_output, axis=1).reshape([-1])
        index = np.arange(target_q.shape[0]) * target_q.shape[1]
        index = index + target_q_argmax
        selected_target_q = target_q_flat[[index]]

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
