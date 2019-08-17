#!/usr/bin/env python

import gym
import numpy as np
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from imagetranformer import ImageTransformer
from rl_common import ReplayMemory,ReplayMemory_multicamera, update_state,update_state_multicamera, learn, learn_multicamera
from dqn_model import DQN, DQN_multicamera
import cv2
import tensorflow as tf
from datetime import datetime
import sys
from std_msgs.msg import Int16
import matplotlib.pyplot as plt

MAX_EXPERIENCE = 50000
MIN_EXPERIENCE = 100
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 128
K = 3
n_history = 3

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y

def shuffle_models(models, target_models, experience_replay_buffer,):
    models_temp = []
    target_models_temp = []
    experience_replay_buffer_temp = []
    Nm = len(models)
    for ii in range(Nm):
        idx = np.random.randint(0,Nm-ii)
        models_temp.append(models.pop(idx))
        target_models_temp.append(target_models.pop(idx))
        experience_replay_buffer_temp.append(experience_replay_buffer.pop(idx))
    return models_temp, target_models_temp, experience_replay_buffer_temp

def play_ones(
            env,
            sess,
            total_t,
            experience_replay_buffer_prey,
            experience_replay_buffer_predator,
            prey_model,
            target_models_prey,
            predator_model,
            target_models_predator,
            image_transformer,
            gamma,
            batch_sz,
            epsilon,
            epsilon_change,
            epsilon_min):
    
    t0 = datetime.now()
    obs = env.reset()
    

    obs_small1 = image_transformer.transform(obs[0][0], sess)
    obs_small2 = image_transformer.transform(obs[0][1], sess)
    state_prey1 = np.stack([obs_small1] * n_history, axis = 2)
    state_prey2 = np.stack([obs_small2] * n_history, axis = 2)

    obs_small1 = image_transformer.transform(obs[1][0], sess)
    obs_small2 = image_transformer.transform(obs[1][1], sess)
    state_predator1 = np.stack([obs_small1] * n_history, axis = 2)
    state_predator2 = np.stack([obs_small2] * n_history, axis = 2)
    loss = None
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = [0,0,0,0]
    record = True
    done = False
    
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_models_prey.copy_from(prey_model)
            target_models_predator.copy_from(predator_model)
            print("model is been copied!")
        action = []
        action.append(prey_model.sample_action(state_prey1, state_prey2, epsilon))
        action.append(predator_model.sample_action(state_predator1, state_predator2, epsilon))
        obs, reward, done, _ = env.step(action)
        next_state = []
        for ii in range(2):
            episode_reward[ii] += reward[ii]

        obs_small1 = image_transformer.transform(obs[0][0], sess)
        obs_small2 = image_transformer.transform(obs[0][1], sess)
        next_state_prey1, next_state_prey2 = update_state_multicamera(state_prey1,state_prey2, obs_small1, obs_small2)
        experience_replay_buffer_prey.add_experience(action[0], obs_small1,obs_small2, reward[0], done)

        obs_small1 = image_transformer.transform(obs[1][0], sess)
        obs_small2 = image_transformer.transform(obs[1][1], sess)
        next_state_predator1, next_state_predator2 = update_state_multicamera(state_predator1,state_predator2, obs_small1, obs_small2)
        experience_replay_buffer_predator.add_experience(action[1], obs_small1,obs_small2, reward[1], done)

        t0_2 = datetime.now()
        
        loss = learn_multicamera(prey_model, target_models_prey, experience_replay_buffer_prey, gamma, batch_sz)
        loss = learn_multicamera(predator_model, target_models_predator, experience_replay_buffer_predator, gamma, batch_sz)
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon


if __name__ == '__main__':
    print "Starting training!!!"
    

    rospy.init_node('sumo_dqlearn',
                    anonymous=True, log_level=rospy.WARN)
    episode_counter_pub = rospy.Publisher('/episode_counter', Int16)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot2/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")   
    obs = env.reset()
    
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('dql_robot')
    rospy.loginfo("Monitor Wrapper started")
    last_time_steps = np.ndarray(0)
    gamma = rospy.get_param("/turtlebot2/gamma")
    batch_sz = 32
    num_episodes = rospy.get_param("/turtlebot2/nepisodes")
    total_t = 0
    start_time = time.time()
    highest_reward = 0
    epsilon = rospy.get_param("/turtlebot2/epsilon")
    epsilon_min = rospy.get_param("/turtlebot2/epsilon_min")
    epsilon_change = (epsilon - epsilon_min) / MAX_EXPERIENCE
    
    experience_replay_buffer_prey = ReplayMemory_multicamera(frame_height = IM_SIZE, fram_width=IM_SIZE, agent_history_lenth=n_history)
    prey_model = DQN_multicamera(
        K = K,
        scope="prey_model",
        image_size1=IM_SIZE,
        image_size2=IM_SIZE,
        n_history = n_history
        )
    target_models_prey = DQN_multicamera(
        K = K,
        scope="prey_target_model",
        image_size1=IM_SIZE,
        image_size2=IM_SIZE,
        n_history = n_history
        )

    experience_replay_buffer_predator = ReplayMemory_multicamera(frame_height = IM_SIZE, fram_width=IM_SIZE,agent_history_lenth=n_history)
    predator_model = DQN_multicamera(
        K = K,
        scope="predator_model",
        image_size1=IM_SIZE,
        image_size2=IM_SIZE,
        n_history = n_history
        )
    target_models_predator = DQN_multicamera(
        K = K,
        scope="predator_target_model",
        image_size1=IM_SIZE,
        image_size2=IM_SIZE,
        n_history = n_history
        )   
    image_transformer = ImageTransformer(IM_SIZE)
    episode_rewards = np.zeros((2,num_episodes))
    episode_lens = np.zeros(num_episodes)
    obs = env.reset()
    with tf.Session() as sess:
        prey_model.set_session(sess)
        target_models_prey.set_session(sess)
        predator_model.set_session(sess)
        target_models_predator.set_session(sess)
        sess.run(tf.global_variables_initializer())
        print("Initializing experience replay buffer...")
        obs = env.reset()
        
        for i in range(MIN_EXPERIENCE):
            action = []
            for ii in range(2):
                action.append(np.random.choice(K))
            obs, reward, done, _ = env.step(action)
            obs_small1 = image_transformer.transform(obs[0][0], sess)
            obs_small2 = image_transformer.transform(obs[0][1], sess)
            experience_replay_buffer_prey.add_experience(action[0],obs_small1, obs_small2, reward[0], done)

            obs_small1 = image_transformer.transform(obs[1][0], sess)
            obs_small2 = image_transformer.transform(obs[1][1], sess)
            experience_replay_buffer_predator.add_experience(action[1],obs_small1, obs_small2, reward[1], done)

            if done:
                obs = env.reset()

        
        print("Done! Starts Training!!")     
        t0 = datetime.now()
        for i in range(num_episodes):
            msg_data = Int16()
            msg_data.data = i
            episode_counter_pub.publish(msg_data)
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_ones(
                    env,
                    sess,
                    total_t,
                    experience_replay_buffer_prey,
                    experience_replay_buffer_predator,
                    prey_model,
                    target_models_prey,
                    predator_model,
                    target_models_predator,
                    image_transformer,
                    gamma,
                    batch_sz,
                    epsilon,
                    epsilon_change,
                    epsilon_min)
            last_100_avg = []
            for ii in range(2):
                episode_rewards[ii,i] = episode_reward[ii]
                last_100_avg.append(episode_rewards[ii,max(0,i-100):i+1].mean())
            episode_lens[i] = num_steps_in_episode
            print("Episode:", i ,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" %time_per_step,
                  "Avg Reward : "+str(last_100_avg),
                  "Epsilon:", "%.3f"%epsilon)
            sys.stdout.flush()
        print("Total duration:", datetime.now()-t0)
        
        y1 = smooth(episode_rewards[0,:])
        y2 = smooth(episode_rewards[1,:])

        plt.plot(y1, label='prey')
        plt.plot(y2, label='predator')

        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()
        env.close()    



