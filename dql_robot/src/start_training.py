#!/usr/bin/env python

import gym
import numpy as np
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from imagetranformer import ImageTransformer
from rl_common import ReplayMemory, update_state, learn
from dqn_model import DQN
import cv2
import tensorflow as tf
from datetime import datetime
import sys




if __name__ == '__main__':
    print "Starting training!!!"
    

    rospy.init_node('sumo_dqlearn',
                    anonymous=True, log_level=rospy.WARN)

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
    pkg_path = rospack.get_path('gazebo_sumo')
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

    # Initialises the algorithm that we are going to use for learning
    #qlearn = qlearn.QLearn(actions=range(env.action_space.n),
     #                      alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    #initial_epsilon = qlearn.epsilon
    obs = env.reset()
    for ii in range(100):
        obs, reward, done, _ = env.step([1,1,1])

    env.close()

    



