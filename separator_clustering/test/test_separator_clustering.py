from copy import deepcopy

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped

from store_monitoring_messages.srv import ClusterSeparatorsRequest, ClusterSeparators


@pytest.fixture(scope='module')
def ros(request):
    try:
        rospy.loginfo('deleting tmp test folder')
        # shutil.rmtree(folder_name)
    except Exception:
        pass

        rospy.loginfo('init ros')
    rospy.init_node('tests')

    def kill_ros():
        rospy.loginfo('shutdown ros')
        rospy.signal_shutdown('die')
        try:
            rospy.loginfo('deleting tmp test folder')
            # shutil.rmtree(folder_name)
        except Exception:
            pass

    request.addfinalizer(kill_ros)


def test1(ros):
    srv = rospy.ServiceProxy('separator_clustering/clustering', ClusterSeparators)
    req = ClusterSeparatorsRequest()
    layer = PoseStamped()
    layer.header.frame_id = 'map'
    layer.pose.position.z = 1
    layer.pose.orientation.w = 1
    req.shelf_layer = layer
    req.shelf_layer_width = 1
    number_of_separtors = 10
    number_of_samples = 10
    distance = (req.shelf_layer_width / number_of_separtors)
    for i in range(number_of_separtors):
        separator = PointStamped()
        separator.header.frame_id = 'map'
        separator.point.z = layer.pose.position.z
        x = distance * i + distance/2
        for _ in range(number_of_samples):
            separator = deepcopy(separator)
            separator.point.x = max(0, min(req.shelf_layer_width, x + np.random.normal(scale=0.01)))
            req.separators.append(separator)
    req.plot = True
    res = srv.call(req)
    assert len(res.separators) == number_of_separtors
