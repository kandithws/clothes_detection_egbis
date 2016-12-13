#include <cstdlib>
#include <algorithm>
#include "ros/ros.h"
#include "ros/package.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <ClothesDetector.h>
#include <actionlib/server/simple_action_server.h>
#include <clothes_detection/FindClothesAction.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

//#define DEFAULT_CLOUD_TOPIC "/camera/depth_registered/points"
#define DEFAULT_CLOUD_TOPIC "/cloud_pcd"
#define ANALYZE_FROM_PLANE 0
#define ANALYZE_FROM_CLUSTER 1
#define ANALYZE_FROM_FINE_CROPPED_CLUSTER 2


//typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class ClothesDetectionRunner
{
	private:
		ros::NodeHandle nh;
		PointCloudT::Ptr cloud_obj;
        ros::Subscriber cloub_sub;
        std::string camera_frame;
        std::string robot_base_frame;
        std::string package_path;
        std::string cloud_topic;
        bool new_cloud_available;
        bool offline_test;
        double cloud_waiting_time;
        int threshold_sv[4];
        double ece_constraint[3];
        double egbis_coarse_constraint[3];
        double egbis_fine_constraint[3];
        double pass_through_range[6];
        bool pass_scene_enable_y;
        bool transform_cloud_to_base_link;
        double egbis_coarse_percent_area_th;
        double egbis_fine_percent_area_th;
        int algorithm;
        int total_clothes;
        tf::TransformListener tf_listener;
		actionlib::SimpleActionServer<clothes_detection::FindClothesAction> find_clothes_as_;
		//-------Clothes Detector Instance-------
	    ClothesDetector clothes_detector;
	
	public:
		ClothesDetectionRunner(std::string node_name):
		nh("~"),
		find_clothes_as_(nh, node_name.c_str(), boost::bind(&ClothesDetectionRunner::executeFindClothesCallback, this, _1), false),
        cloud_obj(new pcl::PointCloud<PointT>())
		{
            this->camera_frame = "/camera_link"; //Default
            this->robot_base_frame = "/base_link";
            this->new_cloud_available = false;
            this->package_path = ros::package::getPath("clothes_detection");

            nh.param<std::string>( "input_topic", this->cloud_topic, DEFAULT_CLOUD_TOPIC);
            ROS_INFO( "input_topic: %s", this->cloud_topic.c_str());

            nh.param( "threshold_sat_lower", this->threshold_sv[0], 0 );
            ROS_INFO( "threshold_sat_lower: %d", this->threshold_sv[0] );

            nh.param( "threshold_sat_upper", this->threshold_sv[1], 15 );
            ROS_INFO( "threshold_sat_upper: %d", this->threshold_sv[1] );

            nh.param( "threshold_value_lower", this->threshold_sv[2], 200 );
            ROS_INFO( "threshold_value_lower: %d", this->threshold_sv[2] );

            nh.param( "threshold_value_upper", this->threshold_sv[3], 255 );
            ROS_INFO( "threshold_value_upper: %d", this->threshold_sv[3] );


            nh.param( "egbis_coarse_sigma", this->egbis_coarse_constraint[0], 2.0 );
            ROS_INFO( "egbis_coarse_sigma: %lf", this->egbis_coarse_constraint[0] );

            nh.param( "egbis_coarse_k", this->egbis_coarse_constraint[1], 600.00 );
            ROS_INFO( "egbis_coarse_k: %lf", this->egbis_coarse_constraint[1] );


            nh.param( "egbis_coarse_min_size", this->egbis_coarse_constraint[2], 2000.00 );
            ROS_INFO( "egbis_coarse_min_size: %lf", this->egbis_coarse_constraint[2] );

            nh.param("egbis_coarse_percent_area_th", this->egbis_coarse_percent_area_th, 3.00 );
            ROS_INFO( "egbis_coarse_percent_area_th: %lf", this->egbis_coarse_percent_area_th);

            //clothes_detector.setEgbisConstraint((float)egbis_coarse_constraint[0],
            //                                    (int)egbis_coarse_constraint[1], (int)egbis_coarse_constraint[2]);
            nh.param( "egbis_fine_sigma", this->egbis_fine_constraint[0], 1.5 );
            ROS_INFO( "egbis_fine_sigma: %lf", this->egbis_fine_constraint[0] );

            nh.param( "egbis_fine_k", this->egbis_fine_constraint[1], 300.00 );
            ROS_INFO( "egbis_fine_k: %lf", this->egbis_fine_constraint[1] );


            nh.param( "egbis_fine_min_size", this->egbis_fine_constraint[2], 2000.00 );
            ROS_INFO( "egbis_fine_min_size: %lf", this->egbis_fine_constraint[2] );

            nh.param("egbis_fine_percent_area_th", this->egbis_fine_percent_area_th, 3.00 );
            ROS_INFO( "egbis_fine_percent_area_th: %lf", this->egbis_fine_percent_area_th);

            nh.param( "ece_cluster_tolerance", this->ece_constraint[0], 0.02 );
            ROS_INFO( "ece_cluster_tolerance: %lf", this->ece_constraint[0] );

            nh.param( "ece_min_cluster_size", this->ece_constraint[1], 10000.00 );
            ROS_INFO( "ece_min_cluster_size: %lf", this->ece_constraint[1] );

            nh.param( "ece_max_cluster_size", this->ece_constraint[2], 25000.00 );
            ROS_INFO( "ece_max_cluster_size: %lf", this->ece_constraint[2] );

            nh.param( "pass_scene_enable_y", this->pass_scene_enable_y, false );
            ROS_INFO( "pass_scene_enable_y: %d", this->pass_scene_enable_y );

            nh.param("transform_cloud_to_base_link", this->transform_cloud_to_base_link, false );
            ROS_INFO( "transform_cloud_to_base_link: %d", this->transform_cloud_to_base_link);

            nh.param( "pass_through_min_z", this->pass_through_range[0], 0.3 );
            ROS_INFO( "pass_through_min_z: %lf", this->pass_through_range[0] );

            nh.param( "pass_through_max_z", this->pass_through_range[1], 2.0 );
            ROS_INFO( "pass_through_max_z: %lf", this->pass_through_range[1] );

            nh.param( "pass_through_min_y", this->pass_through_range[2], 0.0 );
            ROS_INFO( "pass_through_min_y: %lf", this->pass_through_range[2]);

            nh.param( "pass_through_max_y", this->pass_through_range[3], 2.0 );
            ROS_INFO( "pass_through_max_y: %lf", this->pass_through_range[3] );

            nh.param( "pass_through_min_x", this->pass_through_range[4], 0.0 );
            ROS_INFO( "pass_through_min_x: %lf", this->pass_through_range[4] );

            nh.param( "pass_through_max_x", this->pass_through_range[5], 2.0 );
            ROS_INFO( "pass_through_max_x: %lf", this->pass_through_range[5] );

            nh.param( "cloud_waiting_time", this->cloud_waiting_time, 20.00 );
            ROS_INFO( "cloud_waiting_time: %lf", this->cloud_waiting_time );

            nh.param( "offline_test", this->offline_test, false );
            ROS_INFO( "offline_test: %d", this->offline_test );

            if ((offline_test) && (transform_cloud_to_base_link))
            {
                ROS_WARN("OFFLINE TEST CANNOT TRANSFORM CLOUD : set transform_cloud_to_base_link = false");
                this->transform_cloud_to_base_link = false;
            }

            nh.param( "total_clothes", this->total_clothes, 3 );
            ROS_INFO( "total_clothes: %d", this->total_clothes );

            clothes_detector.setWhiteColorThreshold(this->threshold_sv[0], this->threshold_sv[1],
                                                    this->threshold_sv[2], this->threshold_sv[3]);

            clothes_detector.setClusteringConstraint( (float)this->ece_constraint[0],
                                                      (int)this->ece_constraint[1], (int)this->ece_constraint[2]);

            nh.param( "algorithm", this->algorithm, ANALYZE_FROM_CLUSTER );
            //std::string algorithm_str = (this->algorithm)?("ANALYZE_FROM_CLUSTERS"):("ANALYZE_FROM_PLANE");
            std::string algorithm_str;
            switch(this->algorithm)
            {
                case (ANALYZE_FROM_PLANE): algorithm_str = "ANALYZE_FROM_PLANE"; break;
                case (ANALYZE_FROM_CLUSTER): algorithm_str = "ANALYZE_FROM_CLUSTER"; break;
                case (ANALYZE_FROM_FINE_CROPPED_CLUSTER): algorithm_str = "ANALYZE_FROM_FINE_CROPPED_CLUSTER"; break;
                default: algorithm_str = "UNKNOWN"; break;
            }


            ROS_INFO( "algorithm: %s", algorithm_str.c_str());

            this->find_clothes_as_.start();
            ROS_INFO("Starting %s", node_name.c_str());
		};

		void executeFindClothesCallback(const clothes_detection::FindClothesGoalConstPtr &goal)
		{
            ROS_INFO("I'm in Callback");
            this->cloub_sub = nh.subscribe(this->cloud_topic, 1, &ClothesDetectionRunner::cloudCallback, this);
            ros::Time start_time = ros::Time::now();
            int current_percent = 0;
            find_clothes_as_.publishFeedback(this->generateActionFeedBack(current_percent));

            while (!this->new_cloud_available)
            {
                //Check Waiting Time
                double current_wait_time = ros::Time::now().toSec() - start_time.toSec();
                std::stringstream out;
                out << "Waiting for point_cloud : " << current_wait_time;
                ROS_INFO("%s", out.str().c_str());
                if(current_wait_time > this->cloud_waiting_time)
                {
                    ROS_ERROR("No input Cloud within %d seconds Abort Processing", (int)this->cloud_waiting_time);
                    find_clothes_as_.setAborted(clothes_detection::FindClothesResult(), "No Input PointCloud");
                    this->new_cloud_available = false;
                    this->cloub_sub.shutdown();
                    return;
                }
                ros::Duration(0.5).sleep();
            }
            ROS_INFO("----Processing----");
            try
            {
                find_clothes_as_.publishFeedback(this->generateActionFeedBack((current_percent = 10)));
                ROS_INFO("Extracting Plane");
                if(this->transform_cloud_to_base_link)
                {
                    clothes_detector.setPlaneSearchSpaceCloudTF((float)this->pass_through_range[0],
                                                                (float)this->pass_through_range[1],
                                                                (float)this->pass_through_range[4],
                                                                (float)this->pass_through_range[5]);
                }
                else
                {
                    clothes_detector.setPlaneSearchSpace((float)this->pass_through_range[0],
                                                         (float)this->pass_through_range[1],
                                                         this->pass_scene_enable_y,
                                                         (float)this->pass_through_range[2],
                                                         (float)this->pass_through_range[3]);
                }


                pcl::PCLImage plane_pcl_img, original_pcl_img;
                cv::Mat plane_img, original_img, egbis_img;
                std::vector<pcl::PCLImage> cluster_pcl_img;
                std::vector<cv::Mat> segment_img, bin_img, cluster_img;
                ClothesDetector::ClothesContainer out;
                //-------------------------------- Extract RGB from Plane / Clusters -------------------------
                find_clothes_as_.publishFeedback(this->generateActionFeedBack((current_percent = 20)));
                ROS_INFO("Cloud Width: %d, Height: %d", this->cloud_obj->width, this->cloud_obj->height);
                if(this->algorithm == ANALYZE_FROM_CLUSTER)
                {
                    clothes_detector.extractClustersImages(this->cloud_obj, cluster_pcl_img, original_pcl_img,
                                                           (this->package_path + "/output/") );
                    ROS_INFO("Total Clusters : %d", (int)cluster_pcl_img.size());
                    for(int i=0 ; i < cluster_pcl_img.size() ; i++)
                    {
                        cv::Mat tmp;
                        this->convertPCLImage2CVMat(cluster_pcl_img[i], tmp);
                        cluster_img.push_back(tmp);
                    }
                    this->saveImagesToFolder(cluster_img, "cluster");
                }
                else if(this->algorithm == ANALYZE_FROM_FINE_CROPPED_CLUSTER)
                {
                    clothes_detector.extractClustersFineCroppedImages(this->cloud_obj,
                                                                      cluster_pcl_img,
                                                                      original_pcl_img,
                                                                      true,
                                                                      (this->package_path + "/output/") );

                    ROS_INFO("Total Clusters : %d", (int)cluster_pcl_img.size());
                    for(int i=0 ; i < cluster_pcl_img.size() ; i++)
                    {
                        cv::Mat tmp;
                        this->convertPCLImage2CVMat(cluster_pcl_img[i], tmp);
                        cluster_img.push_back(tmp);
                    }
                    ROS_INFO("Saving CLUSTERS IMAGES");
                    this->saveImagesToFolder(cluster_img, "cluster");
                    //this->saveImagesToFolderPCL(cluster_pcl_img, "cluster");
                }
                else if(this->algorithm == ANALYZE_FROM_PLANE)
                {
                    clothes_detector.extractPlaneImage(this->cloud_obj, plane_pcl_img,
                                                       original_pcl_img);
                    this->convertPCLImage2CVMat(plane_pcl_img, plane_img);
                    cv::imwrite((this->package_path + "/output/plane.jpg"), plane_img);
                }

                else
                    ROS_WARN("No specified Algorithm");


                this->convertPCLImage2CVMat(original_pcl_img, original_img);

                find_clothes_as_.publishFeedback(this->generateActionFeedBack((current_percent = 40)));
                ROS_INFO("Finish PointCloud Processing");
                cv::imwrite((this->package_path + "/output/original.jpg"), original_img);

                //----------------------- EGBIS -----------------
                if(this->algorithm == ANALYZE_FROM_CLUSTER)
                {
                    //Do Coarse EGBIS First to Extract + Denoise Cluster Image
                    //Then Do Fine EGBIS to Get Each Clothes Image in cluster
                    std::vector<cv::Mat> segment_coarse;
                    ROS_INFO("--------Do Coarse EGBIS-------");
                    for(int i=0; i < cluster_img.size() ; i++)
                    {
                        std::vector<cv::Mat> segment_tmp;
                        std::vector<double> segment_area_percent;
                        clothes_detector.setEgbisConstraint((float)egbis_coarse_constraint[0],
                                                            (int)egbis_coarse_constraint[1], (int)egbis_coarse_constraint[2]);
                        int normal_num = clothes_detector.getEgbisSegmentVisualize(cluster_img[i], egbis_img);
                        segment_area_percent.clear();
                        clothes_detector.getEgbisSegment(cluster_img[i], segment_tmp, segment_area_percent, this->egbis_coarse_percent_area_th);
                        ROS_INFO("Total Segment of Cluster %d: %d", i,normal_num);
                        ROS_INFO("Total Segment After Threshold of Cluster %d: %d", i,(int)segment_tmp.size());

                        if(!segment_area_percent.empty())
                        {
                            //Deleting Max area Image -> Assuming that it is a Background
                            int index = 0;
                            index = (int)std::distance(segment_area_percent.begin(),
                                                       std::max_element(segment_area_percent.begin(), segment_area_percent.end()));
                            segment_area_percent.erase(segment_area_percent.begin() + index);
                            segment_tmp.erase(segment_tmp.begin() + index);
                        }

                        int index_out = 0;
                        if(!segment_area_percent.empty())
                        {
                            //FIX CODE : SELECT ONLY BIGGEST SEGMENT
                            index_out = (int)std::distance(segment_area_percent.begin(),
                                                       std::max_element(segment_area_percent.begin(), segment_area_percent.end()));
                        }

                        std::stringstream ss;
                        ss << this->package_path << "/output/egbis_" << i << "_coarse.jpg";
                        cv::imwrite(ss.str(), egbis_img);
                        std::stringstream ss_1;
                        ss_1 << "segment_cluster_" << i << "_coarse.jpg";
                        //this->saveImagesToFolder(segment_tmp, ss_1.str());
                        cv::imwrite((this->package_path + "/output/" + ss_1.str()), segment_tmp[index_out]);
                        //segment_img.insert(segment_img.end(), segment_tmp.begin(), segment_tmp.end());
                        segment_coarse.push_back(segment_tmp[index_out]);
                    }
                    ROS_INFO("--------Finish Coarse EGBIS-------");


                    for(int i = 0 ; i < segment_coarse.size(); i++) //One Cluster will get only One Coarse
                    {
                        std::vector<cv::Mat> segment_tmp, segment_out;
                        std::vector<double> segment_area_percent;
                        //---------- Do Fine EGBIS ---------
                        clothes_detector.setEgbisConstraint((float)egbis_fine_constraint[0],
                                                            (int)egbis_fine_constraint[1], (int)egbis_fine_constraint[2]);
                        int normal_num = clothes_detector.getEgbisSegmentVisualize(segment_coarse[i], egbis_img);
                        segment_area_percent.clear();
                        clothes_detector.getEgbisSegment(segment_coarse[i], segment_tmp, segment_area_percent, this->egbis_fine_percent_area_th);

                        ROS_INFO("Total Segment of Cluster %d: %d", i,normal_num);
                        ROS_INFO("Total Segment After Threshold of Cluster %d: %d", i,(int)segment_tmp.size());

                        if(!segment_area_percent.empty())
                        {
                            //Deleting Max area Image -> Assuming that it is a Background
                            int index = 0;
                            index = (int)std::distance(segment_area_percent.begin(),
                                                       std::max_element(segment_area_percent.begin(), segment_area_percent.end()));
                            segment_tmp.erase(segment_tmp.begin() + index);
                            segment_area_percent.erase(segment_area_percent.begin() + index);
                        }

                        std::vector<int> out;
                        if(!segment_area_percent.empty())
                        {
                            //Select $(this->total_clothes) maximum area component to analyze
                            int total_iteration = std::min(this->total_clothes, (int)segment_tmp.size());
                            for(int i = 0; i < total_iteration ; i++)
                            {
                                int index = 0;
                                index = (int)std::distance(segment_area_percent.begin(),
                                                           std::max_element(segment_area_percent.begin(), segment_area_percent.end()));
                                out.push_back(index);
                                //segment_tmp.erase(segment_tmp.begin() + index);
                                segment_area_percent.erase(segment_area_percent.begin() + index);
                            }
                        }
                        for(int j = 0; j < out.size(); j ++)
                        {
                            segment_out.push_back(segment_tmp[out[j]]);
                        }
                        std::stringstream ss;
                        ss << this->package_path << "/output/egbis_" << i << "_fine.jpg";
                        cv::imwrite(ss.str(), egbis_img);
                        std::stringstream ss_1;
                        ss_1 << "segment_cluster_" << i << "_fine";
                        this->saveImagesToFolder(segment_out, ss_1.str());
                        //segment_img.insert(segment_img.end(), segment_tmp.begin(), segment_tmp.end());
                        segment_img.insert(segment_img.end(), segment_out.begin(), segment_out.end());
                    }
                    ROS_INFO("--------Finish FINE EGBIS-------");

                }
                else if(this->algorithm == ANALYZE_FROM_PLANE)
                {
                    std::vector<double> segment_area_percent;
                    clothes_detector.setEgbisConstraint((float)egbis_coarse_constraint[0],
                                                        (int)egbis_coarse_constraint[1], (int)egbis_coarse_constraint[2]);
                    int normal_num = clothes_detector.getEgbisSegmentVisualize(plane_img, egbis_img);
                    clothes_detector.getEgbisSegment(plane_img, segment_img, segment_area_percent, this->egbis_coarse_percent_area_th);
                    ROS_INFO("Total Segment : %d", normal_num);
                    ROS_INFO("Total Segment After Threshold : %d", (int)segment_img.size());
                    if(!segment_area_percent.empty())
                    {
                        //Deleting Max area Image -> Assuming that it is a Background
                        int index = 0;
                        index = (int)std::distance(segment_area_percent.begin(),
                                                   std::max_element(segment_area_percent.begin(), segment_area_percent.end()));
                        segment_img.erase(segment_img.begin() + index);

                    }

                    cv::imwrite((this->package_path + "/output/egbis.jpg"), egbis_img);
                    this->saveImagesToFolder(segment_img, "segment");
                }
                else if(this->algorithm == ANALYZE_FROM_FINE_CROPPED_CLUSTER)
                {
                    std::vector<double> segment_area_percent;
                    clothes_detector.setEgbisConstraint((float)egbis_fine_constraint[0],
                                                        (int)egbis_fine_constraint[1], (int)egbis_fine_constraint[2]);
                    for(int i =0 ; i < cluster_img.size(); i++)
                    {
                        int normal_num = clothes_detector.getEgbisSegmentVisualize(cluster_img[i], egbis_img);
                        clothes_detector.getEgbisSegment(cluster_img[i], segment_img, segment_area_percent, this->egbis_fine_percent_area_th);
                        ROS_INFO("Total Segment : %d", normal_num);
                        ROS_INFO("Total Segment After Threshold : %d", (int)segment_img.size());
                        if(!segment_area_percent.empty())
                        {
                            //Deleting Max area Image -> Assuming that it is a Background
                            int index = 0;
                            index = (int)std::distance(segment_area_percent.begin(),
                                                       std::max_element(segment_area_percent.begin(), segment_area_percent.end()));
                            segment_img.erase(segment_img.begin() + index);

                        }
                    }


                    cv::imwrite((this->package_path + "/output/egbis.jpg"), egbis_img);
                    this->saveImagesToFolder(segment_img, "segment");
                }
                else
                    ROS_WARN("No specified Algorithm");

                ROS_INFO("-----Complete EGBIS-----");
                find_clothes_as_.publishFeedback(this->generateActionFeedBack((current_percent = 60)));
                //-------------------------- Detect Clothes --------------------------------
                clothes_detector.setOriginalImage(original_img);
                clothes_detector.getBinaryImage(segment_img, bin_img, 0);

                if((this->algorithm == ANALYZE_FROM_CLUSTER) ||(this->algorithm == ANALYZE_FROM_FINE_CROPPED_CLUSTER) )
                    clothes_detector.detectClothesObjects(bin_img, out, false);
                else if(this->algorithm == ANALYZE_FROM_PLANE)
                    clothes_detector.detectClothesObjects(bin_img, out, true);

                find_clothes_as_.publishFeedback(this->generateActionFeedBack((current_percent = 80)));
                for(int i=0; i< out.size(); i++)
                    clothes_detector.findDominantColor(out[i], 2);
                try
                {
                    //TODO -- FIX ERROR IN DRAWING Contour Descriptors
                    clothes_detector.saveOutputImages(out, (this->package_path + "/output/out") );
                }
                catch(cv::Exception& e)
                {
                    ROS_WARN("Error in saving Descriptors Files");
                    //ROS_WARN("%s", e.what());
                }
                
                find_clothes_as_.publishFeedback(this->generateActionFeedBack((current_percent = 90)));

                for(int i=0; i< out.size(); i++)
                {
                    clothes_detector.map2DPointToPointCloud(this->cloud_obj, out[i]);
                    ROS_INFO("OUTPUT[%d] POSITION: x = %lf, y = %lf, z = %lf", i, out[i].position.x,
                                out[i].position.y, out[i].position.z);
                }
                ROS_INFO("Saving File to %s", this->package_path.c_str());
                find_clothes_as_.publishFeedback(this->generateActionFeedBack(current_percent = 100));
                find_clothes_as_.setSucceeded(*this->generateActionResult(out), "Complete");
                this->new_cloud_available = false;
                ROS_INFO("------------- Complete Processing ------------------");
            }
            catch (cv::Exception& e)
            {
                find_clothes_as_.setAborted(clothes_detection::FindClothesResult(), "Error While Executing");
                ROS_ERROR("Error While Executing : Find Clothes Action in clothes_detection_node (%d %s)",
                          current_percent, "%");
                ROS_ERROR("cv::Exception: %s", e.what());
                this->new_cloud_available = false;
                return;
            }
            catch (std::exception& e)
            {
                find_clothes_as_.setAborted(clothes_detection::FindClothesResult(), "Error While Executing");
                ROS_ERROR("Error While Executing : Find Clothes Action in clothes_detection_node (%d %s)",
                          current_percent, "%");
                ROS_ERROR("std::exception: %s", e.what());
                this->new_cloud_available = false;
                return;
            }
            catch ( ... )
            {
                find_clothes_as_.setAborted(clothes_detection::FindClothesResult(), "Error While Executing");
                ROS_ERROR("Error While Executing : Find Clothes Action in clothes_detection_node (%d %s)",
                          current_percent, "%");
                this->new_cloud_available = false;
                return;
            }

		}

		void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
		{
            ROS_INFO("Receiving Clouds");
            PointCloudT::Ptr cloud(new pcl::PointCloud<PointT>());
            //pcl::fromROSMsg (*cloud_in, *cloud_obj);
            pcl::fromROSMsg (*cloud_in, *cloud);
            //this->camera_frame.clear();
            this->camera_frame = cloud_in->header.frame_id;
            //pcl_ros::transformPointCloud(robot_frame, *cloud, *cloud_obj, *listener);
            if((this->transform_cloud_to_base_link) && (!this->offline_test))
            {
                tf::StampedTransform  transform;
                this->tf_listener.waitForTransform(this->robot_base_frame, cloud_in->header.frame_id,
                                                   cloud_in->header.stamp, ros::Duration(1.0));
                //this->tf_listener.lookupTransform();
                this->tf_listener.lookupTransform(this->robot_base_frame, cloud_in->header.frame_id, ros::Time(0), transform);
                //pcl_ros::transformPointCloud(this->robot_base_frame,*cloud, *cloud_obj, this->tf_listener);
                pcl_ros::transformPointCloud(*cloud, *cloud_obj, transform);
            }
            else
            {
                this->cloud_obj = cloud;
            }

            this->new_cloud_available = true;
            this->cloub_sub.shutdown();
		}

        void convertPCLImage2CVMat(pcl::PCLImage& pcl_img, cv::Mat& cv_img)
        {
            sensor_msgs::Image tmp_img;
            cv_bridge::CvImagePtr cv_in_ptr;
            pcl_conversions::fromPCL(pcl_img, tmp_img);
            cv_in_ptr = cv_bridge::toCvCopy( tmp_img, sensor_msgs::image_encodings::BGR8 );
            cv_img = cv_in_ptr->image;
        }

        clothes_detection::FindClothesFeedback generateActionFeedBack(int percent_complete)
        {
            clothes_detection::FindClothesFeedback tmp;
            tmp.percent_complete = percent_complete;
            return tmp;
        }

        clothes_detection::FindClothesResult::Ptr generateActionResult(ClothesDetector::ClothesContainer& data)
        {
            clothes_detection::FindClothesResult::Ptr tmp(new clothes_detection::FindClothesResult);
            tmp->result.header = this->generateHeader();
            for(int i=0 ; i < data.size() ; i++)
            {
                if(data[i].dominant_color == ClothesDetector::UNKNOWN)
                    break;

                clothes_detection::Clothes clothes;
                clothes.area = data[i].contour_area;
                geometry_msgs::Point centroid;
                centroid.x = data[i].position.x;
                centroid.y = data[i].position.y;
                centroid.z = data[i].position.z;

                if(!this->offline_test)
                {
                    clothes.centroid = this->transformCamera2Base(centroid);
                }
                else
                {
                    clothes.centroid = centroid;
                }

                clothes.color = data[i].dominant_color;
                clothes.type = data[i].type;
                /*cv_bridge::CvImage img_msg;
                img_msg.header = this->generateHeader();
                img_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
                img_msg.image = data[i].cropped;
                img_msg.toImageMsg(clothes->image);*/
                tmp->result.array.push_back(clothes);
            }
            return tmp;
        }

        std_msgs::Header generateHeader(std::string frame_id = "base_link")
        {
            std_msgs::Header tmp;
            //tmp.frame_id = this->camera_frame;

            if(this->offline_test)
                tmp.frame_id = this->camera_frame;
            else
                tmp.frame_id = this->robot_base_frame;
            //if(this->transform_cloud_to_base_link)
                //tmp.frame_id = this->robot_base_frame;
            //else
                //tmp.frame_id = this->camera_frame;
            tmp.stamp = ros::Time::now();
            return tmp;
        }

        void saveImagesToFolder(std::vector<cv::Mat>& input, std::string filename)
        {
            std::string directory;
            directory = this->package_path + "/output/";

            for(int i = 0; i < input.size(); i++)
            {
                std::stringstream ss;
                ss << directory << filename << '_' << i << ".jpg";
                cv::imwrite(ss.str().c_str(), input[i]);
            }
        }

        geometry_msgs::Point transformCamera2Base(geometry_msgs::Point input)
        {
            tf::StampedTransform transform;
            geometry_msgs::PointStamped stamped_msgs, stamped_out;
            stamped_msgs.header.frame_id = this->camera_frame;
            stamped_msgs.header.stamp = ros::Time(0);
            stamped_msgs.point = input;
            try
            {
                this->tf_listener.waitForTransform(this->robot_base_frame, this->camera_frame,
                                                   ros::Time(0), ros::Duration(2.0));
                this->tf_listener.transformPoint(this->robot_base_frame, stamped_msgs, stamped_out);
                return stamped_out.point;
            }
            catch (tf::TransformException ex){
                ROS_ERROR("ERROR IN POINT TRANSFORM CONVERSION");
                ROS_ERROR("%s",ex.what());
            }
        }
};

int main( int argc, char **argv )
{
    ros::init( argc, argv, "clothes_detection_node");
    ClothesDetectionRunner runner(ros::this_node::getName());
    ros::spin();
    return 0;
}	