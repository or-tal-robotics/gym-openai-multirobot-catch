import rospy, tf
import numpy as np
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Quaternion,Pose, Point
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from openai_ros.msg import RLExperimentInfo
from modify_launchers import modify_launch
from openai_ros.openai_ros_common import ROSLauncher


def create_random_launch_files():
    initial_pose = np.zeros((4,2))
    initial_pose[0,:] = np.random.uniform(-4,4,2)
    ii = 1
    while ii <= 3:
        pose = np.random.uniform(-4,4,2)
        norms = np.linalg.norm(initial_pose-pose)
        if np.sum(norms<0.4):
            continue
        else:
            initial_pose[ii,:] = pose
            ii += 1
            #print(initial_pose)
    modify_launch(initial_pose)

# https://github.com/openai/gym/blob/master/gym/core.py
class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):

        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()
        self.step_number = 0
        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = [0.0,0.0,0.0]
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        rospy.logdebug("END init RobotGazeboEnv")

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        #if self.step_number > 200:
            #self.reset()
        rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        #self._prey_step()
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        for ii in range(3):
            self.cumulated_episode_reward[ii] = self.cumulated_episode_reward[ii]+ reward[ii]
        self.step_number += 1
        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

    def reset(self):
        
       
        self.win = [0,0,0]
        self.prey_win = 0
        self.step_number = 0
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs
        

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        rospy.logwarn("PUBLISHING REWARD...")
        self._publish_reward_topic(
                                    self.cumulated_episode_reward[0],
                                    self.episode_num
                                    )
        rospy.logwarn("PUBLISHING REWARD...DONE="+str(self.cumulated_episode_reward)+",EP="+str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = [0.0,0.0,0.0]


    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _spwan(self):
            rospy.wait_for_service("gazebo/spawn_sdf_model")
            self.spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

            with open("/home/lab/openai_ws/src/dql_sumo/gazebo_sumo/spwan_node/ball/ball.sdf", "r") as f:
                product_xml = f.read()
            rx = np.random.uniform(low=-2.4, high=2.4) 
            if rx < 1.0 and rx > -1.0:
                ry = np.random.uniform(low=1.0, high=2.4)*(-1)**np.random.randint(0,2) 
            else:
                ry = np.random.uniform(low=-2.4, high=2.4) 

            random_pose = np.array([rx,ry])

            orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
            item_name   =   "ball"
            item_pose   =   Pose(Point(x=random_pose[0], y=random_pose[1],    z=0.5),   orient)
            self.spawn_model(item_name, product_xml, "", item_pose, "world")
            #print("Spawning model:%s", item_name)

    def _del_model(self):
        rospy.wait_for_service("gazebo/delete_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        item_name = "prey"
        #print("Deleting model:%s", item_name)
        delete_model(item_name)
                

    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls :
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            #if self.episode_num > 0:
                #self._del_model()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            #self._spwan()
            #ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
            #create_random_launch_files()
            #ROSLauncher(rospackage_name="dql_robot",
                #launch_file_name="put_prey_in_world.launch",
                #ros_ws_abspath=ros_ws_abspath)
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()

        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            #if self.episode_num > 0:
                #self._del_model()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            #self._spwan()
            #ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
            #create_random_launch_files()
            #ROSLauncher(rospackage_name="dql_robot",
                #launch_file_name="put_prey_in_world.launch",
                #ros_ws_abspath=ros_ws_abspath)
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _prey_step(self):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()

