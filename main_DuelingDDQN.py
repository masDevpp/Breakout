import os
import tensorflow as tf
import gym
import numpy as np
from PIL import Image
from DuelingDDQNAgent import DuelingDDQNAgent
from EpisodeMemory import EpisodeMemory
import time

parameters = {
    "Breakout":
    {
        "name": "Breakout",
        "env": "Breakout-v0",
        "frameskip": 4,
        "resize": (90, 102),
        "crop": [14, -4, 3, -3] # [high, low, left, right]
    },
    "SpaceInvaders":
    {
        "name": "SpaceInvaders",
        "env": "SpaceInvaders-v0",
        "frameskip": 3,
        "resize": (94, 124),
        "crop": [0, -1, 0, -1]
    }
}

def main(env_param):
    num_states_to_hold = 4
    epsilon_initial = 1.0
    epsilon_end = 0.1
    epsilon_decay_end_episode = 1000000
    training_frequency = 4
    target_network_update_frequency = 10000
    batch_size = 32
    discount_rate = 0.99
    memory_size = 810000#1000000
    learn_start = 800000#50000
    learning_rate = 9e-6#1e-5#5e-5 # lr wold better to decay to arround 1e-5
    eval_ep_frequency = 25
    eval_max_step = 3000
    ckpt_backup_ep_frequeny = int(200 / eval_ep_frequency) * eval_ep_frequency
    render_during_train = False

    should_render = False

    env = gym.make(env_param["env"], frameskip=env_param["frameskip"])
    num_actions = env.action_space.n

    state = env.reset()

    episode_memory = EpisodeMemory(learn_start, memory_size, num_states_to_hold, True, env_param["resize"], env_param["crop"], 0.2, 30)
    num_states = list(episode_memory.preprocess_state(state).shape) + [num_states_to_hold]

    with tf.Session() as sess:#tf.Session(config=tf.ConfigProto(log_device_placement=True))
        agent = DuelingDDQNAgent(sess, num_states, num_actions, epsilon_initial, epsilon_end, discount_rate)

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
        episode_memory.add_one_step(state, 0, 0.0, False)

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

            action, _ = agent.predict_action_with_epsilon_greedy(episode_memory.get_last_states(), global_step / epsilon_decay_end_episode)
            
            state, reward, terminal, _ = env.step(action)
            if reward > 0: reward = 1.0
            elif reward < 0: reward = -1.0

            if terminal: state = env.reset()

            episode_reward += reward
            local_step += 1

            episode_memory.add_one_step(state, action, reward, terminal)

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
                    elapse_time = time.time() - start_time
                    print("Ep " + str(episode_count) + ", EpReward " + str(episode_reward) + ", Elapse " + format(elapse_time, ".2f") + "(" + format(elapse_time / local_step, ".5f") + ") Loss " + format(loss_sum / (num_loss_added+0.00001), ".5f") + ", EpsilonProg " + format(global_step / epsilon_decay_end_episode, ".4f"))
                    start_time = time.time()
                
                    if episode_count % eval_ep_frequency == 0:
                        eval_reward_0, q0 = evaluation(agent, num_states_to_hold, env, False, eval_max_step)
                        eval_reward_1, q1 = evaluation(agent, num_states_to_hold, env, False, eval_max_step)
                        eval_reward_2, q2 = evaluation(agent, num_states_to_hold, env, False, eval_max_step)
                        eval_reward_f0, qf0 = evaluation(agent, num_states_to_hold, env, True, eval_max_step)
                        env.reset()
                        print("EvalReward " + str(eval_reward_0) + " " + str(eval_reward_1) + " " + str(eval_reward_2) + " f" + str(eval_reward_f0) + ", AveQ " + format(q0, ".5f") + " " + format(q1, ".5f") + " " + format(q2, ".5f") + " f" + format(qf0, ".5f"))

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
                            f.write("Ep " + str(episode_count) + ", TrainReward " + str(episode_reward) + ", EvalReward " + str(eval_reward_0) + " " + str(eval_reward_1) + " " + str(eval_reward_2) + " f" + str(eval_reward_f0) + ", AveQ " + format(q0, ".5f") + " " + format(q1, ".5f") + " " + format(q2, ".5f") + " " + " f" + format(qf0, ".5f") + ", Loss " + format(loss_sum / (num_loss_added+0.00001), ".5f") + ", GlobalStep " + str(global_step) + ", " + time.asctime() + "\n")
                        
                    if summary is not None:
                        summary_writer.add_summary(summary, episode_count)
                    
                    episode_count += 1
                
                episode_reward = 0
                local_step = 0
                episode_memory.remove_old_episode()
                loss_sum = 0
                num_loss_added = 0

def evaluation(agent, num_states_to_hold, env, fire_support = False, max_step = 1000):

    state = env.reset()
    
    episode_memory = EpisodeMemory(0, max_step * 2, num_states_to_hold, True, env_param["resize"], env_param["crop"])
    episode_memory.add_one_step(state, 0, 0.0, False)

    eval_reward = 0

    terminal = False
    live = 0
    dropped = True

    q_sum = 0
    step = 0    

    for i in range(max_step):
        try:
            env.render()
        except:
            pass

        action, q = agent.predict_action(episode_memory.get_last_states())
        if fire_support and dropped: action = 1
        
        q_sum += np.max(q)
        step += 1

        state, reward, terminal, info_dict = env.step(action)
        
        if live > info_dict["ale.lives"]: dropped = True
        else: dropped = False
        live = info_dict["ale.lives"]

        eval_reward += reward

        if terminal: break

        episode_memory.add_one_step(state, action, reward, terminal)
    
    return eval_reward, q_sum / step

def predict(env_param):
    checkpoint = os.path.join(LOG_DIR, "save", "model.ckpt-29400")
    #checkpoint = os.path.join(LOG_DIR, "model.ckpt-325")

    num_states_to_hold = 4
    discount_rate = 0.99
    sleep_duration = 0.016
    max_step = 2500

    env = gym.make(env_param["env"], frameskip=env_param["frameskip"])
    num_actions = env.action_space.n
    
    state = env.reset()

    episode_memory = EpisodeMemory(0, 10000, num_states_to_hold, True, env_param["resize"], env_param["crop"])
    num_states = list(episode_memory.preprocess_state(state).shape) + [num_states_to_hold]
    episode_memory.add_one_step(state, 0, 0.0, False)

    sess = tf.Session()

    agent = DuelingDDQNAgent(sess, num_states, num_actions, 1.0, 1.0, discount_rate)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    episode_reward = 0
    local_step = 0

    while True:
        state = env.reset()
        episode_memory.reset()
        episode_memory.add_one_step(state, 0, 0.0, False)
        episode_reward = 0
        local_step = 0
        images = [Image.fromarray(state)]

        for i in range(max_step):
            env.render()
            time.sleep(sleep_duration)

            action, q = agent.predict_action(episode_memory.get_last_states())

            state, reward, terminal, info_dict = env.step(action)
            images.append(Image.fromarray(state))

            episode_reward += reward
            local_step += 1

            episode_memory.add_one_step(state, action, reward, terminal)

            if terminal: break
        
        print("Step " + str(local_step) + ", Reward " + str(episode_reward) + ", " + str(terminal))
    

if __name__ == "__main__":
    env_param = parameters["Breakout"]
    LOG_DIR = os.path.join(os.getcwd(), "log_DuelingDDQN_" + env_param["name"])

    #main(env_param)
    predict(env_param)
