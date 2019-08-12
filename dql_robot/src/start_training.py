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
IM_SIZE = 84
K = 5
n_history = 4

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

def play_ones(env,
                sess,
                total_t,
                experience_replay_buffer,
                experience_replay_buffer_prey,
                models,
                prey_model,
                target_models,
                target_models_prey,
                image_transformer,
                gamma,
                batch_sz,
                epsilon,
                epsilon_change,
                epsilon_min):
    
    t0 = datetime.now()
    obs = env.reset()
    state = []
    
    for ii in range(3):
        obs_small = image_transformer.transform(obs[ii], sess)
        state.append(np.stack([obs_small] * n_history, axis = 2))

    obs_small1 = image_transformer.transform(obs[3][0], sess)
    obs_small2 = image_transformer.transform(obs[3][1], sess)
    state_prey1 = np.stack([obs_small1] * n_history, axis = 2)
    state_prey2 = np.stack([obs_small2] * n_history, axis = 2)
    loss = None
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = [0,0,0,0]
    record = True
    done = False
    
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            for ii in range(3):
                target_models[ii].copy_from(models[ii])
            target_models_prey.copy_from(prey_model)
            print("model is been copied!")
        action = []
        for ii in range(3):
            action.append(models[ii].sample_action(state[ii], epsilon))
        action.append(prey_model.sample_action(state_prey1, state_prey2, epsilon))
        obs, reward, done, _ = env.step(action)
        next_state = []
        for ii in range(3):
            obs_small = image_transformer.transform(obs[ii], sess)
            next_state.append(update_state(state[ii], obs_small))
        
            episode_reward[ii] += reward[ii]
        
            experience_replay_buffer[ii].add_experience(action[ii], obs_small, reward[ii], done)

        obs_small1 = image_transformer.transform(obs[3][0], sess)
        obs_small2 = image_transformer.transform(obs[3][1], sess)
        next_state_prey1, next_state_prey1 = update_state_multicamera(state_prey1,state_prey2, obs_small1, obs_small2)
        experience_replay_buffer_prey.add_experience(action[3], obs_small1,obs_small2, reward[3], done)
        t0_2 = datetime.now()
        for ii in range(3):
            loss = learn(models[ii], target_models[ii], experience_replay_buffer[ii], gamma, batch_sz)
        loss = learn_multicamera(prey_model, target_models_prey, experience_replay_buffer_prey, gamma, batch_sz)
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        state = next_state
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, loss


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
    #outdir = pkg_path + '/training_results'
    #env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")

    
    gamma = 0.99
    batch_sz = 32
    num_episodes = 2000
    total_t = 0
    start_time = time.time()
    highest_reward = 0
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000
    experience_replay_buffer = []
    models = []
    target_models = []
    for ii in range(3):
        experience_replay_buffer.append(ReplayMemory())
        models.append(DQN(
            K = K,
            scope="model"+str(ii),
            image_size=IM_SIZE
            ))
        target_models.append(DQN(
            K = K,
            scope="target_model"+str(ii),
            image_size=IM_SIZE
            ))
    experience_replay_buffer_prey = ReplayMemory_multicamera()
    prey_model = DQN_multicamera(
        K = K,
        scope="prey_model",
        image_size1=IM_SIZE,
        image_size2=IM_SIZE
        )
    target_models_prey = DQN_multicamera(
        K = K,
        scope="prey_target_model",
        image_size1=IM_SIZE,
        image_size2=IM_SIZE
        )
    image_transformer = ImageTransformer(IM_SIZE)
    episode_rewards = np.zeros((4,num_episodes))
    episode_lens = np.zeros(num_episodes)
    obs = env.reset()
    with tf.Session() as sess:
        for ii in range(3):
            models[ii].set_session(sess)
            target_models[ii].set_session(sess)

        prey_model.set_session(sess)
        target_models_prey.set_session(sess)
        sess.run(tf.global_variables_initializer())
        print("Initializing experience replay buffer...")
        obs = env.reset()
        
        for i in range(MIN_EXPERIENCE):
            action = []
            for ii in range(4):
                action.append(np.random.choice(K))
            obs, reward, done, _ = env.step(action)
            for ii in range(3):
                obs_small = image_transformer.transform(obs[ii], sess)
                experience_replay_buffer[ii].add_experience(action[ii], obs_small, reward[ii], done)
            obs_small1 = image_transformer.transform(obs[3][0], sess)
            obs_small2 = image_transformer.transform(obs[3][1], sess)
            experience_replay_buffer_prey.add_experience(action[3],obs_small1, obs_small2, reward[3], done)
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
                    experience_replay_buffer,
                    experience_replay_buffer_prey,
                    models,
                    prey_model,
                    target_models,
                    target_models_prey,
                    image_transformer,
                    gamma,
                    batch_sz,
                    epsilon,
                    epsilon_change,
                    epsilon_min)
            last_100_avg = []
            for ii in range(4):
                episode_rewards[ii,i] = episode_reward[ii]
                last_100_avg.append(episode_rewards[ii,max(0,i-100):i+1].mean())
            episode_lens[i] = num_steps_in_episode
            models[0:3], target_models[0:3], experience_replay_buffer[0:3] = shuffle_models(models[0:3], target_models[0:3], experience_replay_buffer[0:3])
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
        y3 = smooth(episode_rewards[2,:])
        y4 = smooth(episode_rewards[3,:])
        plt.plot(y1, label='robot 1')
        plt.plot(y2, label='robot 2')
        plt.plot(y3, label='robot 3')
        plt.plot(y4, label='prey')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()
        env.close()    



