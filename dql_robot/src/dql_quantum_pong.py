import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from imagetranformer import ImageTransformer
from rl_common import ReplayMemory, update_state, learn
from dqn_model import DQN
from gym import wrappers
import cv2
import time


MAX_EXPERIENCE = 50000
MIN_EXPERIENCE = 5000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 3
n_history = 4

    

            

        


def play_ones(env,
              sess,
              total_t,
              experience_replay_buffer,
              model,
              target_model,
              image_transformer,
              gamma,
              batch_size,
              epsilon,
              epsilon_change,
              epsilon_min,
              pathOut,
              record):
    
    t0 = datetime.now()
    obs = env.reset()
    print(obs.shape)
    obs_small = image_transformer.transform(obs, sess)
    state = np.stack([obs_small] * n_history, axis = 2)
    loss = None
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0
    
    done = False
    if record == True:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 10.0, (640,480))
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("model is been copied!")
        
        action = model.sample_action(state, epsilon)
        obs, reward, done, _ = env.step(action)
        obs_small = image_transformer.transform(obs, sess)
        next_state = update_state(state, obs_small)
        
        episode_reward += reward
        
        experience_replay_buffer.add_experience(action, obs_small, reward, done)
        t0_2 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        if record == True:
            frame = cv2.cvtColor(obs, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame,(640,480))
            out.write(frame)
            #cv2.imshow("frame", frame)
    if record == True:
        out.release()
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, loss

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y
        

if __name__ == '__main__':
    conv_layer_sizes = [(32,8,4), (64,4,2), (64,3,1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_sz = 32
    num_episodes = 3500
    total_t = 0
    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)
    episode_lens = np.zeros(num_episodes)
    
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000
    
    env = gym.make('gym_quantum_pong:Quantum_Pong-v0')
    
    #monitor_dir = 'video'
    #env = wrappers.Monitor(env, monitor_dir)
    
    model = DQN(
            K = K,
            conv_layer_sizes=conv_layer_sizes,
            hidden_layer_sizes=hidden_layer_sizes,
            scope="model",
            image_size=IM_SIZE
            )
    
    target_model = DQN(
            K = K,
            conv_layer_sizes=conv_layer_sizes,
            hidden_layer_sizes=hidden_layer_sizes,
            scope="target_model",
            image_size=IM_SIZE
            )
    
    image_transformer = ImageTransformer(IM_SIZE)
    
    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        #model.load()
        #target_model.load()
        sess.run(tf.global_variables_initializer())
        print("Initializing experience replay buffer...")
        obs = env.reset()
        
        for i in range(MIN_EXPERIENCE):
            action = np.random.choice(K)
            obs, reward, done, _ = env.step(action)
            obs_small = image_transformer.transform(obs, sess)
            experience_replay_buffer.add_experience(action, obs_small, reward, done)
            
            if done:
                obs = env.reset()
                
        t0 = datetime.now()
        record = True
        for i in range(num_episodes):
            video_path = 'video/video'+str(i)+'.avi'
            if i%100 == 0:
                record = True
            else:
                record = False
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_ones(
                    env,
                    sess,
                    total_t,
                    experience_replay_buffer,
                    model,
                    target_model,
                    image_transformer,
                    gamma,
                    batch_sz,
                    epsilon,
                    epsilon_change,
                    epsilon_min,
                    video_path,
                    record)
            episode_rewards[i] = episode_reward
            episode_lens[i] = num_steps_in_episode
            last_100_avg = episode_rewards[max(0,i-100):i+1].mean()
            print("Episode:", i ,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" %time_per_step,
                  "Avg Reward:", "%.3f"%last_100_avg,
                  "Epsilon:", "%.3f"%epsilon)
            sys.stdout.flush()
        print("Total duration:", datetime.now()-t0)
        model.save()
        
        y = smooth(episode_rewards)
        plt.plot(episode_rewards, label='orig')
        plt.plot(y, label='smoothed')
        plt.legend()
        plt.show()
        
    
    
    
    
    
    
        
    
