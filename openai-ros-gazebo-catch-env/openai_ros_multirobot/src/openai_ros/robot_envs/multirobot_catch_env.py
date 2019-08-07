import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image


class TurtleBot2catchEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        
        self.bridge = CvBridge()
        rospy.logdebug("Start TurtleBot2catchEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="dql_robot",
                    launch_file_name="put_robots_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        ROSLauncher(rospackage_name="dql_robot",
                    launch_file_name="put_prey_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtleBot2catchEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/robot1/odom", Odometry, self._odom_callback1)
        rospy.Subscriber("/robot2/odom", Odometry, self._odom_callback2)
        rospy.Subscriber("/robot3/odom", Odometry, self._odom_callback3)
        rospy.Subscriber("/robot1/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback1)
        rospy.Subscriber("/robot2/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback2)
        rospy.Subscriber("/robot3/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback3)
        

        self._cmd_vel_pub1 = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=1)
        self._cmd_vel_pub2 = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=1)
        self._cmd_vel_pub3 = rospy.Publisher('/robot3/cmd_vel', Twist, queue_size=1)
        self._cmd_vel_pub_prey = rospy.Publisher('/prey/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber("/gazebo/model_states", ModelStates ,self._model_state_callback)
        #rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_depth_image_raw_callback)
        #rospy.Subscriber("/camera/depth/points", PointCloud2, self._camera_depth_points_callback)
        
        

        

        self._check_publishers_connection()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished TurtleBot2Env INIT...")
        

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    def _model_state_callback(self,msg):
        models = msg.name
        robot_idx1 = models.index('mobile_base1')
        robot_idx2 = models.index('mobile_base2')
        robot_idx3 = models.index('mobile_base3')
        prey_idx = models.index('prey')
        self.robot_position1 = [msg.pose[robot_idx1].position.x, msg.pose[robot_idx1].position.y]
        self.robot_position2 = [msg.pose[robot_idx2].position.x, msg.pose[robot_idx2].position.y]
        self.robot_position3 = [msg.pose[robot_idx3].position.x, msg.pose[robot_idx3].position.y]
        self.prey_position = [msg.pose[prey_idx].position.x, msg.pose[prey_idx].position.y]

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom1 = None
        self.odom2 = None
        self.odom3 = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while (self.odom1 is None or self.odom2 is None or self.odom3 is None) and not rospy.is_shutdown():
            try:
                self.odom1 = rospy.wait_for_message("/robot1/odom", Odometry, timeout=5.0)
                self.odom2 = rospy.wait_for_message("/robot2/odom", Odometry, timeout=5.0)
                self.odom3 = rospy.wait_for_message("/robot3/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")

        return self.odom1
        
        

        

    def _odom_callback1(self, data):
        self.odom1 = data

    def _odom_callback2(self, data):
        self.odom2 = data

    def _odom_callback3(self, data):
        self.odom3 = data
        
    def _camera_rgb_image_raw_callback1(self, data):
        self.camera_rgb_image_raw1 = self.bridge.imgmsg_to_cv2(data,"rgb8")

    def _camera_rgb_image_raw_callback2(self, data):
        self.camera_rgb_image_raw2 = self.bridge.imgmsg_to_cv2(data,"rgb8")

    def _camera_rgb_image_raw_callback3(self, data):
        self.camera_rgb_image_raw3 = self.bridge.imgmsg_to_cv2(data,"rgb8")
            
    def _laser_scan_callback1(self, data):
        self.laser_scan1 = data

    def _laser_scan_callback2(self, data):
        self.laser_scan2 = data

    def _laser_scan_callback3(self, data):
        self.laser_scan3 = data

        
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._cmd_vel_pub1.get_num_connections() == 0 or self._cmd_vel_pub2.get_num_connections() == 0 or self._cmd_vel_pub3.get_num_connections() == 0 ) and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self,robot_id, linear_speed, angular_speed,sleep_time = 0.2, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("TurtleBot2 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        if robot_id == 1:
            self._cmd_vel_pub1.publish(cmd_vel_value)
        if robot_id == 2:
            self._cmd_vel_pub2.publish(cmd_vel_value)
        if robot_id == 3:
            self._cmd_vel_pub3.publish(cmd_vel_value)
        time.sleep(sleep_time)


    def _move_prey(self, linear_speed, angular_speed,sleep_time = 0.2):
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        self._cmd_vel_pub_prey.publish(cmd_vel_value)
        time.sleep(sleep_time)

    def _prey_step(self):
        if self.prey_position[0] > 4.5 or self.prey_position[0]<-4.5 or self.prey_position[1] > 4.5 or self.prey_position[1]<-4.5:
            linear_speed = numpy.random.uniform(-0.2,0.0)
            angular_speed = numpy.random.uniform(-0.2,0.2)
        elif self.prey_position[0] > 4.0 or self.prey_position[0]<-4.0 or self.prey_position[1] > 4.0 or self.prey_position[1]<-4.0:
            linear_speed = numpy.random.uniform(0.0,0.1)
            angular_speed = numpy.random.uniform(0.2,0.5)
        else:
            linear_speed = numpy.random.uniform(0.0,0.5)
            angular_speed = numpy.random.uniform(-0.5,0.5)
        self._move_prey(linear_speed, angular_speed, sleep_time=0.5)
                        
        

    def get_odom(self, robot_id):
        if robot_id == 1:
            return self.odom1
        if robot_id == 2:
            return self.odom2
        if robot_id == 3:
            return self.odom3
        
    def get_camera_rgb_image_raw(self, robot_id):
        if robot_id == 1:
            return self.camera_rgb_image_raw1
        if robot_id == 2:
            return self.camera_rgb_image_raw2
        if robot_id == 3:
            return self.camera_rgb_image_raw3
        
    def get_laser_scan(self, robot_id):
        if robot_id == 1:
            return self.laser_scan1
        if robot_id == 2:
            return self.laser_scan2
        if robot_id == 3:
            return self.laser_scan3

    def get_prey_position(self):
        return self.prey_position

    def get_robot_position(self, robot_id):
        if robot_id == 1:
            return self.robot_position1
        if robot_id == 2:
            return self.robot_position2
        if robot_id == 3:
            return self.robot_position3
        
    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and 
        
        """
        
