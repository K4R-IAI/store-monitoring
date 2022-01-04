#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3, PointStamped
from sklearn.cluster import DBSCAN
from std_msgs.msg import ColorRGBA

import giskardpy.utils.tfwrapper as tf
from giskardpy.utils.math import inverse_frame
from store_monitoring_messages.srv import ClusterSeparators, ClusterSeparatorsRequest, ClusterSeparatorsResponse


class SeparatorClustering(object):
    prefix = 'separator detector'

    def __init__(self):
        self.clustering_srv = rospy.Service('~clustering', ClusterSeparators, self.clustering_srv_cb)
        self.width_threshold = rospy.get_param('~width_threshold', default=0.035)
        self.height_threshold = rospy.get_param('~height_threshold', default=0.06)
        self.min_samples = rospy.get_param('~min_samples', default=4)
        self.max_dist = rospy.get_param('~max_dist', default=0.02)
        self.detections = []
        self.map_frame_id = 'map'
        self.separator_maker_color = ColorRGBA(.8, .8, .8, .8)
        self.separator_maker_scale = Vector3(.01, .5, .05)

    def clustering_srv_cb(self, req: ClusterSeparatorsRequest) -> ClusterSeparatorsResponse:
        rospy.loginfo(f'{rospy.get_name()} Received request.')
        res = ClusterSeparatorsResponse()
        centers = self.cluster(req.separators, req.shelf_layer, req.shelf_layer_width, req.plot)
        res.separators = centers
        return res

    def separator_on_shelf_layer(self, layer_P_separator, layer_width):
        x = layer_P_separator[0]
        z = layer_P_separator[2]
        return self.width_threshold <= x <= layer_width - self.width_threshold and \
               -self.height_threshold <= z <= self.height_threshold

    def cluster(self, original_separators, shelf_layer, shelf_layer_width, visualize=False):
        map_T_layer = tf.transform_pose('map', shelf_layer)
        layer_T_map = inverse_frame(tf.msg_to_homogeneous_matrix(map_T_layer))
        separators = []
        for separator in original_separators:
            map_P_separator = tf.msg_to_homogeneous_matrix(tf.transform_point('map', separator))
            layer_P_separator = np.dot(layer_T_map, map_P_separator)
            if self.separator_on_shelf_layer(layer_P_separator, shelf_layer_width):
                separators.append(layer_P_separator[:-1])
            else:
                print('muh')

        data = np.array(separators)
        centers = []
        if len(data) == 0:
            rospy.loginfo('no separators detected')
        else:
            clusters = DBSCAN(eps=self.max_dist, min_samples=self.min_samples).fit(data)
            labels = np.unique(clusters.labels_)
            rospy.loginfo('Detected {} separators.'.format(len(labels)))
            for i, label in enumerate(labels):
                if label != -1:
                    separator = PointStamped()
                    separator.header.frame_id = 'map'
                    separator.point = Point(*self.cluster_to_separator(data[clusters.labels_ == label]))
                    centers.append(separator)
            if len(separators) == 0:
                rospy.logwarn(f'{rospy.get_name()} didn\'t detect any clusters.')
            if visualize:
                self.visualize_detections(clusters.labels_, separators, centers)
        return centers

    def cluster_to_separator(self, separator_cluster):
        """
        :param separator_cluster: 3*x
        :type separator_cluster: np.array
        :return: 3*1
        :rtype: np.array
        """
        return separator_cluster.mean(axis=0)

    def point_list_to_np(self, poses):
        """
        :param poses: list of PoseStamped
        :type poses: list
        :return: 3*x numpy array
        :rtype: np.array
        """
        l = []
        for p in poses:  # type: PointStamped
            l.append([p.point.x,
                      p.point.y,
                      p.point.z])
        return np.array(l)

    def visualize_detections(self, labels, original_separators, centers):
        centers = self.point_list_to_np(centers)
        # original_separators = self.point_list_to_np(original_separators)
        original_separators = np.array(original_separators)
        import pylab as plt
        from mpl_toolkits.mplot3d import Axes3D  # needed to make projection='3d' work

        ulabels = np.unique(labels)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(ulabels):
            if i % 2 == 0:
                color = 'g'
            else:
                color = 'y'
            if label == -1:
                color = 'r'
            ax.scatter(original_separators[labels == label, 0], original_separators[labels == label, 1],
                       original_separators[labels == label, 2],
                       c=color,
                       linewidth=0.0)

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='k',
                   marker='x', s=80)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 1)
        ax.set_ylim(-.5, .5)
        ax.set_zlim(-.5, .5)
        plt.show()


if __name__ == '__main__':
    rospy.init_node('separator_clustering')
    sc = SeparatorClustering()
    rospy.loginfo(f'{rospy.get_name()} is alive.')
    rospy.spin()
