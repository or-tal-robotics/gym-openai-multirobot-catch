import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import multirobot_catch_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import cv2
import time

def get_image_moment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0]) 
    upper = np.array([255,255,10]) 
    mask = cv2.inRange(hsv, lower, upper) 
    M = cv2.moments(mask)
    return M['m00']


class CatchEnv(multirobot_catch_env.TurtleBot2catchEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="dql_robot",
                    launch_file_name="catch_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        rospy.logdebug("finish loading sumo_world.launch")

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros_multirobot",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/multirobot_catch/config",
                               yaml_file_name="multirobot_catch.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(CatchEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        
        
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        
        
        # We only use two integers
        self.observation_space = spaces.Box(low=0, high=255, shape= (640, 480, 3))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.catch_reward = rospy.get_param("/turtlebot2/catch_reward")
        self.cooperative_catch_reward = rospy.get_param("/turtlebot2/cooperative_catch_reward")
        self.time_penelty = rospy.get_param("/turtlebot2/time_penelty")
        self.robot_out_of_bounds_penalty = rospy.get_param("/turtlebot2/robot_out_of_bounds_penalty")
        self.robot_hit_robot_penalty = rospy.get_param("/turtlebot2/robot_hit_robot_penalty")
        self.max_x = rospy.get_param("/turtlebot2/max_x") 
        self.max_y = rospy.get_param("/turtlebot2/max_y") 
        self.min_x = rospy.get_param("/turtlebot2/min_x") 
        self.min_y = rospy.get_param("/turtlebot2/min_y") 

        self.cumulated_steps = 0.0
        
        self.init_linear_forward_speed = 0.0
        self.init_linear_turn_speed = 0.0
        self.win = [0,0,0]
        self.prey_win = 0

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        for ii in range(1,4):
            self.move_base(ii, self.init_linear_forward_speed,
                            self.init_linear_turn_speed,
                            sleep_time=0,
                            epsilon=0.05,
                            update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        
        camera_data = self.get_camera_rgb_image_raw(1)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        for ii in range(3):
            if action[ii] == 0: #FORWARD
                linear_speed = self.linear_forward_speed
                angular_speed = 0.0
                self.last_action = "FORWARDS"
            elif action[ii] == 1: #LEFT
                linear_speed = self.linear_turn_speed
                angular_speed = self.angular_speed
                self.last_action = "TURN_LEFT"
            elif action[ii] == 2: #RIGHT
                linear_speed = self.linear_turn_speed
                angular_speed = -1*self.angular_speed
                self.last_action = "TURN_RIGHT"
            elif action[ii] == 3: #RIGHT FORWARD
                linear_speed = self.linear_forward_speed
                angular_speed = -1*self.angular_speed
                self.last_action = "FORWARDS_TURN_RIGHT"
            elif action[ii] == 4: #LEFT FORWARD
                linear_speed = self.linear_forward_speed
                angular_speed = self.angular_speed
                self.last_action = "FORWARDS_TURN_LEFT"

            
            # We tell TurtleBot2 the linear and angular speed to set to execute
            self.move_base(ii+1,linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        if action[3] == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action[3] == 1: #LEFT
            linear_speed = 0.0
            angular_speed = 2*self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action[3] == 2: #RIGHT
            linear_speed = 0.0
            angular_speed = -2*self.angular_speed
            self.last_action = "TURN_RIGHT"
        elif action[3] == 3: #RIGHT FORWARD
            linear_speed = 2*self.linear_forward_speed
            angular_speed = -2*self.angular_speed
            self.last_action = "FORWARDS_TURN_RIGHT"
        elif action[3] == 4: #LEFT FORWARD
            linear_speed = 2*self.linear_forward_speed
            angular_speed = 2*self.angular_speed
            self.last_action = "FORWARDS_TURN_LEFT"

        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(4,linear_speed, angular_speed, epsilon=0.05, update_rate=10)    
        time.sleep(0.1)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        observations = [self.get_camera_rgb_image_raw(1),self.get_camera_rgb_image_raw(2),self.get_camera_rgb_image_raw(3),self.get_camera_rgb_image_raw(4)]
        rospy.logdebug("END Get Observation ==>")
        return observations
        

    def _is_done(self, observations):
        r_robot = 0.351/2
        r_prey = 0.2
        p = 0.1
        self._episode_done = False
        if self.step_number > 500:
            self._episode_done = True
            print("To much steps==> GAME OVER!")
        else:
            prey_position = np.array(self.get_prey_position())
            if prey_position[0] > self.max_x or prey_position[0] < self.min_x or prey_position[1] > self.max_y or prey_position[1] < self.min_y:
                    self._episode_done = True
                    self.prey_win = -1
                    print("Prey hit the wall!")
            for ii in range (3):
                current_position = np.array(self.get_robot_position(ii+1))
                #print (np.linalg.norm(current_position - prey_position))
                if (r_robot + r_prey + p) >= np.linalg.norm(current_position - prey_position):
                    self.win[ii] = 1
                    self.prey_win = -1
                    print("Robot "+str(ii+1)+" catched the prey!")
                    self._episode_done = True
                if current_position[0] > self.max_x or current_position[0] < self.min_x or current_position[1] > self.max_y or current_position[1] < self.min_y:
                    self._episode_done = True
                    self.win[ii] = -1
                    print("Robot "+str(ii+1)+" hit the wall!")
        return self._episode_done

    def _compute_reward(self, observations, done):
        r_robot = 0.351/2
        p = 0.1
        reward = [0.0, 0.0, 0.0, 0.0]
        if done:     
            if self.prey_win == -1:
                reward[3] = -1
            else:
                reward[3] = 0

            if 1 in self.win:
                for ii in range(3):
                    if self.win[ii] == 1:
                        reward[ii] = 100.0*self.catch_reward/(self.step_number+1.0)
                    else:
                        reward[ii] = 100.0*self.cooperative_catch_reward/(self.step_number+1.0)
            elif -1 in self.win:
                for ii in range(3):
                    if self.win[ii] == -1:
                        reward[ii] = self.robot_out_of_bounds_penalty

            elif self.prey_win == -1:
                reward = [0,0,0,-1]
                   
            else:
                reward = [self.time_penelty,self.time_penelty,self.time_penelty,-self.time_penelty]


        else:
            for ii in range (3):
                current_position = np.array(self.get_robot_position(ii+1))
                for jj in range (3):
                    if ii == jj:
                        continue
                    else:
                        othere_position = np.array(self.get_robot_position(jj+1))
                        if (2*r_robot + p) >= np.linalg.norm(current_position - othere_position):
                            img1 = self.get_camera_rgb_image_raw(ii+1)
                            img2 = self.get_camera_rgb_image_raw(jj+1)
                            M1 = get_image_moment(img1)
                            M2 = get_image_moment(img2)
                            if M1 > M2:
                                reward[ii] = self.robot_hit_robot_penalty
                                print("robots "+str(ii+1)+" hit robot "+str(jj+1))
                                break
                            elif M2 > M1:
                                reward[jj] = self.robot_hit_robot_penalty
                                print("robots "+str(jj+1)+" hit robot "+str(ii+1))
                                break

                            





        #print("reward=" + str(reward))
        
        return reward


    # Internal TaskEnv Methods
