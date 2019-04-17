import os
import tensorflow as tf
import gym
import numpy as np
from PIL import Image
from DQNAgent import DQNAgent
from EpisodeMemory import EpisodeMemory
import time

LOG_DIR = os.path.join(os.getcwd(), "log")

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
    memory_size = 550000#1000000
    learn_start = 200000#50000
    learning_rate = 0.00010#0.00020#0.00025
    eval_ep_frequency = 25
    eval_max_step = 2000
    ckpt_backup_ep_frequeny = int(200 / eval_ep_frequency) * eval_ep_frequency
    render_during_train = True

    should_render = False

    env = gym.make("Breakout-v0")
    num_actions = env.action_space.n
    # Action 0: NOP, 1: Fire, 2: Right, 3: Left

    state = env.reset()

    episode_memory = EpisodeMemory(learn_start, memory_size, True, num_states_to_hold, skip_frame, 0.2, 30)
    num_states = list(episode_memory.preprocess_state(state).shape) + [num_states_to_hold]

    with tf.Session() as sess:#tf.Session(config=tf.ConfigProto(log_device_placement=True))
        agent = DQNAgent(sess, num_states, num_actions, epsilon_initial, epsilon_end, discount_rate)

        episode_count_variable = tf.Variable(0, trainable=False, name="episode_count")
        global_step_variable = tf.Variable(0, trainable=False, name="global_step")
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        saver = tf.train.Saver()
        saver2 = tf.train.Saver(max_to_keep=10000)
        
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

        state = env.reset()
        episode_memory.add_one_step(state, 0, 0.0, False, 0)

        episode_reward = 0
        local_step = 0
        start_time = time.time()
        terminal = True
        loss_sum = 0
        num_loss_added = 0
        summary = None
        continuous_save_fail_count = 0
        
        while True:
            if should_render: env.render()

            action = agent.predict_action_with_epsilon_greedy(episode_memory.get_last_states(), global_step / epsilon_decay_end_episode)
            
            state, reward, terminal, _ = env.step(action)
            if reward > 1: reward = 1

            if terminal: state = env.reset()

            episode_reward += reward
            local_step += 1

            episode_memory.add_one_step(state, action, reward, terminal, episode_reward)

            if episode_memory.has_enough_memory():
                if global_step % training_frequency == 0:
                    should_render = render_during_train
                    s, a, r, s_next, t = episode_memory.get_batch(batch_size)
                    loss, summary = agent.train(s, a, r, s_next, t, learning_rate=learning_rate)
                    loss_sum += loss
                    num_loss_added += 1
                    
                if global_step % target_network_update_frequency == 0:
                    print("Update target network")
                    agent.update_target_network()

                global_step += 1

            if terminal:
                if episode_memory.has_enough_memory():
                    current_time = time.time()
                    print("Ep " + str(episode_count) + ", EpReward " + str(episode_reward) + ", Elapse " + format(current_time - start_time, ".2f") + " Loss " + format(loss_sum / (num_loss_added+0.00001), ".5f") + ", EpsilonProg " + format(global_step / epsilon_decay_end_episode, ".4f"))
                    start_time = current_time
                
                    if episode_count % eval_ep_frequency == 0:
                        eval_reward_0 = evaluation(agent, num_states_to_hold, skip_frame, env, False, eval_max_step)
                        eval_reward_1 = evaluation(agent, num_states_to_hold, skip_frame, env, False, eval_max_step)
                        eval_reward_2 = evaluation(agent, num_states_to_hold, skip_frame, env, False, eval_max_step)
                        eval_reward_f0 = evaluation(agent, num_states_to_hold, skip_frame, env, True, eval_max_step)
                        env.reset()
                        print("EvalReward " + str(eval_reward_0) + " " + str(eval_reward_1) + " " + str(eval_reward_2) + ", " + str(eval_reward_f0))

                        sess.run(episode_count_variable.assign(episode_count))
                        sess.run(global_step_variable.assign(global_step))
                        
                        try:
                            # Sometime save failed
                            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step = episode_count)
                            continuous_save_fail_count = 0
                            if episode_count % ckpt_backup_ep_frequeny == 0:
                                saver2.save(sess, os.path.join(LOG_DIR, "save", "model.ckpt"), global_step = episode_count)
                        except:
                            print("saver.save() failed!", str(continuous_save_fail_count))
                            continuous_save_fail_count += 1

                        with open(os.path.join(LOG_DIR, "log.txt"), "a") as f:
                            f.write("Ep " + str(episode_count) + ", TrainReward " + str(episode_reward) + ", EvalReward " + str(eval_reward_0) + " " + str(eval_reward_1) + " " + str(eval_reward_2) + " f" + str(eval_reward_f0) + ", Loss " + format(loss_sum / (num_loss_added+0.00001), ".5f") + ", GlobalStep " + str(global_step) + ", " + time.asctime() + "\n")
                        
                    if summary is not None:
                        summary_writer.add_summary(summary, episode_count)
                    
                    episode_count += 1
                
                episode_reward = 0
                local_step = 0
                episode_memory.remove_old_episode()
                loss_sum = 0
                num_loss_added = 0

def evaluation(agent, num_states_to_hold, skip_frame, env, fire_support = False, max_step = 1000):

    state = env.reset()
    
    episode_memory = EpisodeMemory(0, max_step * 2, True, num_states_to_hold, skip_frame)
    episode_memory.add_one_step(state, 0, 0.0, False, 0)

    eval_reward = 0

    terminal = False
    live = 0
    dropped = True

    for i in range(max_step):
        env.render()
        action = agent.predict_action(episode_memory.get_last_states())
        if fire_support and dropped: action = 1

        state, reward, terminal, info_dict = env.step(action)
        
        if live > info_dict["ale.lives"]: dropped = True
        else: dropped = False
        live = info_dict["ale.lives"]

        eval_reward += reward

        if terminal: break

        episode_memory.add_one_step(state, action, reward, terminal, eval_reward)
    
    return eval_reward

if __name__ == "__main__":
    main()
