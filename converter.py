from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation
from vod.visualization import Visualization2D
from vod.frame import FrameLabels
import numpy as np
import cv2

# ros msgs
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
from autoware_perception_msgs.msg import DynamicObjectWithFeatureArray, DynamicObjectWithFeature
import sensor_msgs.point_cloud2 as pcl2
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import rospy
import tf
from cv_bridge import CvBridge, CvBridgeError
import rosbag
# import uuid
from uuid_msgs.msg import UniqueID

rospy.init_node("converter")
compression = rosbag.Compression.NONE

bag = rosbag.Bag('delft.bag', 'w',compression=compression)

# cv bridge
bridge = CvBridge()

topic_camera = "/image_raw"
topic_lidar = "/velodyne_points"
topic_radar = "/radar_points"
topic_tf = "/tf"
topic_gt = "/perception/object_recognition/detection/objects"




kitti_locations = KittiLocations(root_dir="/view_of_delft_PUBLIC",
                                output_dir="/delft_to_ros/output/")

limit = 1000 # 1000 frames is about 9GB of data (raw images)

t_start = rospy.Time.now()

for frame in range(0, limit):
    if not rospy.is_shutdown():

        t_frame = 0.1 * frame 
        # convert to ros time
        t_frame = rospy.Duration.from_sec(t_frame)
        t_frame = t_start + t_frame
        # add 00 in front of frame number
        frame = str(frame).zfill(5)
        frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                                frame_number=frame)
        
        # print("Processing frame: ", frame)
        transforms = FrameTransformMatrix(frame_data)
        # print("Transforms: ", transforms)

        # print(transforms.t_target_origin)
        
        coordinate = np.array([[0, 0, 0, 1]])

        
        # load data and convert to ros topics
        image_data = frame_data.get_image()
        lidar_data = frame_data.get_lidar_scan()
        radar_data = frame_data.get_radar_scan()
        gt_data = frame_data.get_labels()

        # framelabels 
        frame_labels = FrameLabels(gt_data)
        gt_data_processed = frame_labels.get_labels_dict()
        # print("gt_data_processed: ", gt_data_processed)

        
        # pred_data = frame_data.get_predictions()

        # utm_coordinates = homogeneous_transformation(coordinate, transforms.t_map_camera).T

        # print("map-camera: ", utm_coordinates)


        def create_tf_msg(frame_id, child_frame_id, translation, rotation_matrix):
            tf_msg = TransformStamped()
            tf_msg.header = Header()
            tf_msg.header.stamp = t_frame
            tf_msg.header.frame_id = frame_id
            tf_msg.child_frame_id = child_frame_id
            tf_msg.transform.translation.x = translation[0]
            tf_msg.transform.translation.y = translation[1]
            tf_msg.transform.translation.z = translation[2]

            r, p, y = tf.transformations.euler_from_matrix(rotation_matrix)
            q_ref = tf.transformations.quaternion_from_euler(r, p, y)
            tf_msg.transform.rotation.x = q_ref[0]
            tf_msg.transform.rotation.y = q_ref[1]
            tf_msg.transform.rotation.z = q_ref[2]
            tf_msg.transform.rotation.w = q_ref[3]

            return tf_msg

        # tf msgs
        # base_link is velodyne frame
        tf_base_link = create_tf_msg("base_link", "velodyne", [0, 0, 0], np.eye(3))

        # tf from lidar to camera
        rotation = transforms.t_lidar_camera[0:3, 0:3]
        translation = transforms.t_lidar_camera[0:3, 3]
        tf_lidar_camera = create_tf_msg("base_link", "camera", translation, rotation)

        # tf from lidar to radar
        rotation = transforms.t_lidar_radar[0:3, 0:3]
        translation = transforms.t_lidar_radar[0:3, 3]
        tf_lidar_radar = create_tf_msg("base_link", "radar", translation, rotation)

        # publish tf2_msgs/TFMessage
        tf_msg = TFMessage()
        tf_msg.transforms.append(tf_base_link)
        tf_msg.transforms.append(tf_lidar_camera)
        tf_msg.transforms.append(tf_lidar_radar)

        bag.write(topic_tf, tf_msg, t=t_frame)



        # camera intrinsic matrix
        # came_intrinsics = transforms.get_sensor_transforms("camera")
        
        # print("camera intrinsics: ", came_intrinsics)



        # camera data
        # process image, convert back to opencv image
        image_data = np.array(image_data)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

        # convert to ros image
        image_data = bridge.cv2_to_imgmsg(image_data, encoding="passthrough")
        image_data.header = Header()
        image_data.header.stamp = t_frame
        image_data.header.frame_id = "camera"

        # publish image
        # image_pub.publish(image_data)
        bag.write(topic_camera, image_data, t=t_frame)

        # lidar data
        # convert to ros pointcloud2
        if lidar_data is not None:
            lidar_data = np.array(lidar_data, dtype=np.float32).reshape(-1, 4)
            lidar_header = Header()
            lidar_header.frame_id = "velodyne"
            lidar_header.stamp = t_frame

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('i', 12, PointField.FLOAT32, 1)]

            # convert to ros pointcloud2
            lidar_data = pcl2.create_cloud(lidar_header, fields, lidar_data)
            # lidar_pub.publish(lidar_data)
            bag.write(topic_lidar, lidar_data, t=t_frame)




        # radar data
        if radar_data is not None:
            # print("radar data: ", radar_data)
            # print("radar data shape: ", radar_data.shape)
            radar_data = np.array(radar_data, dtype=np.float32).reshape(-1, 7)

            radar_header = Header()
            radar_header.frame_id = "radar"
            radar_header.stamp = t_frame

            # radar fields
            # [x, y, z, RCS, v_r, v_r_compensated, time]

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('RCS', 12, PointField.FLOAT32, 1),
                    PointField('v_r', 16, PointField.FLOAT32, 1),
                    PointField('v_r_compensated', 20, PointField.FLOAT32, 1),
                    PointField('time', 24, PointField.FLOAT32, 1)]

            # convert to ros pointcloud2
            radar_data = pcl2.create_cloud(radar_header, fields, radar_data)
            # radar_pub.publish(radar_data)
            bag.write(topic_radar, radar_data, t=t_frame)

        # ground truth data
        if gt_data_processed is not None:
            # print("gt_data_processed: ", gt_data_processed)
            gt_msgs = DynamicObjectWithFeatureArray()
            gt_msgs.header = Header()
            gt_msgs.header.stamp = t_frame
            gt_msgs.header.frame_id = "camera"
            for obj in gt_data_processed:
                obj_msg = DynamicObjectWithFeature()

                obj_class = obj["label_class"]
                h = obj["h"]
                w = obj["w"]
                l = obj["l"]

                x = obj["x"]
                y = obj["y"]
                z = obj["z"]

                rotation = obj["rotation"]
              
                # generate uuid uint8[16] uuid
                obj_msg.object.id = UniqueID()
                # print("obj_msg.object.id: ", obj_msg.object.id)

                # handle class and score
                obj_msg.object.semantic.confidence = obj["score"]
                if obj_class == "Pedestrian":
                    obj_msg.object.semantic.type = 6
                elif obj_class == "Car":
                    obj_msg.object.semantic.type = 1
                elif obj_class == "Cyclist":
                    obj_msg.object.semantic.type = 4
                elif obj_class == "Truck":
                    obj_msg.object.semantic.type = 2
                elif obj_class == "Bus":
                    obj_msg.object.semantic.type = 3
                elif obj_class == "UNKNOWN":
                    obj_msg.object.semantic.type = 0
                elif obj_class == "MOTORBIKE":
                    obj_msg.object.semantic.type = 5

                # handle position
                obj_msg.object.shape.type = 1
                obj_msg.object.shape.dimensions.x = l
                obj_msg.object.shape.dimensions.y = w
                obj_msg.object.shape.dimensions.z = h

                obj_msg.object.state.pose_covariance.pose.position.x = x
                obj_msg.object.state.pose_covariance.pose.position.y = y
                obj_msg.object.state.pose_covariance.pose.position.z = z

                # handle orientation
                # Rotation around -Z axis of the LiDAR sensor

                rotation = tf.transformations.quaternion_from_euler(0, 0, rotation)
                obj_msg.object.state.pose_covariance.pose.orientation.x = rotation[0]
                obj_msg.object.state.pose_covariance.pose.orientation.y = rotation[1]
                obj_msg.object.state.pose_covariance.pose.orientation.z = rotation[2]
                obj_msg.object.state.pose_covariance.pose.orientation.w = rotation[3]

                # append to array
                gt_msgs.feature_objects.append(obj_msg)

            bag.write(topic_gt, gt_msgs, t=t_frame)


        # show image
        # cv2.imshow("image", image_data)
        # cv2.waitKey(0)

bag.close()
