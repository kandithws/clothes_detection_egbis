#! /usr/bin/env python
import rospy
import roslib
roslib.load_manifest('clothing_type_classification')
import actionlib
import clothing_type_classification.msg
import std_msgs
from sensor_msgs.msg import Image
from clothing_type_classification.msg import ClothesArray, Clothes


#  Specified target Centroid Points and Area of ClothesObject Here [x,y,z,area]
result_clothes = [[0.5, 0.0, 0.7, 50]]


class ClothesDetectionDummy(object):
    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name,
                                                clothing_type_classification.msg.FindClothesAction,
                                                self.execute_cb, False)
        self._feedback = clothing_type_classification.msg.FindClothesFeedback()
        self._result = clothing_type_classification.msg.FindClothesResult()
        self._as.start()
        print "Current Clothes: "
        index = 0
        for i in result_clothes:
            print "clothes[" + str(index) + "] = " + str(i)
            index += 1

        print "-------------Complete Initialization------------------"

    def execute_cb(self, goal):
        global result_clothes
        rospy.loginfo("-------Start Execution-----")
        ca = ClothesArray()
        ca.header.frame_id = "base_link"
        ca.header.stamp = rospy.Time.now()
        for i in result_clothes:
            ca.array.append(self.create_clothes(i))

        print "array => " + str(ca)

        self._result.result = ca

        print "result : " + str(type(self._result.result))
        print str(self._result.result)
        #self._result.result = ClothesArray()
        self._as.set_succeeded(self._result)

    def create_clothes(self, centroid_and_area):
        tmp = Clothes()
        tmp.type = "Unknown"
        tmp.image = Image()
        tmp.centroid.x = centroid_and_area[0]
        tmp.centroid.y = centroid_and_area[1]
        tmp.centroid.z = centroid_and_area[2]
        tmp.area = centroid_and_area[3]
        return tmp

if __name__ == '__main__':
    rospy.init_node('clothes_detection_node')
    ClothesDetectionDummy(rospy.get_name())
    rospy.spin()
