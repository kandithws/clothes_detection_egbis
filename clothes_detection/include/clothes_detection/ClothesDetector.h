//
// Created by kandithws on 28/2/2559.
//

#ifndef CLOTHES_DETECTOR_CLOTHES_DETECTOR_H
#define CLOTHES_DETECTOR_CLOTHES_DETECTOR_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

#include "egbis.h"

typedef pcl::PointXYZRGBA PointT;

class ClothesDetector
{
    public:

        class DetectorDescriptors
        {
        public:
            DetectorDescriptors();
            std::vector<cv::Point> contour;
            double contour_area;
            cv::Point2f centroid; //image centroid
            pcl::PointXYZ position; //position in pointcloud
            cv::Rect rect;
            cv::Mat mask;
            cv::Mat cropped;
            uint8_t type;
            uint8_t dominant_color;
            void copyTo(DetectorDescriptors &target);
        };

        typedef std::vector<DetectorDescriptors> ClothesContainer;

        enum
        {
            TABLE = 0,
            CLOTHES = 1,
            WHITE = 2,
            NON_WHITE = 3,
            UNKNOWN = 4
        };

        ClothesDetector();
        //Morphological Kernel MORPH_RECT,MORPH_CROSS,MORPH_ELLIPSE
        void setOriginalImage(cv::Mat img);
        void setPlaneSearchSpace(float min_z, float max_z, bool y_enable = false, float min_y = 0.0, float max_y = 0.0);
        void setPlaneSearchSpaceCloudTF(float min_z, float max_z, float min_x = 0.0, float max_x = 0.0);
        void setWhiteColorThreshold(int sat_lower, int sat_upper, int value_lower, int value_upper);
        void setClusteringConstraint(float tolerance, int min_size, int max_size);
        void extractPlaneImage(pcl::PointCloud<PointT>::Ptr cloud, pcl::PCLImage& output, pcl::PCLImage& original_img);
        pcl::PointCloud<PointT>::Ptr extractPlaneCloud(pcl::PointCloud<PointT>::Ptr cloud);
        void extractClustersImages(pcl::PointCloud<PointT>::Ptr cloud, std::vector<pcl::PCLImage>& output, pcl::PCLImage& original_img , std::string debug= "");

        void extractClustersFineCroppedImages(pcl::PointCloud<PointT>::Ptr cloud, std::vector<pcl::PCLImage>& output,
                                           pcl::PCLImage& original_img , bool find_only_max_clusters = false, std::string debug= "");

        void map2DPointToPointCloud(pcl::PointCloud<PointT>::Ptr cloud, DetectorDescriptors& input, int window = 3);
        void setEgbisConstraint(float sigma, float k, int min_size);
        int  getEgbisSegmentVisualize(cv::Mat &input, cv::Mat& output);
        int  getEgbisSegment(cv::Mat &input, std::vector<cv::Mat>& output, std::vector<double>& out_percent, double percent_th = 5.0);
        bool findDominantColor(DetectorDescriptors &input, int cluster_number);
        void getBinaryImage(std::vector<cv::Mat>& images, std::vector<cv::Mat>& image_th, int threshold_value,
                            int closing_window_size = 5, int opening_window_size = 5, int kernel_type = cv::MORPH_ELLIPSE );
        void detectClothesObjects(std::vector<cv::Mat> &images_th, ClothesContainer& out,
                                    bool check_table = false, bool crop_original = true);
        void saveOutputImages(ClothesContainer& images, std::string filename = "out",
                                bool draw_descriptors = true);

    private:
        cv::Mat original;
        int sat_lower_th;
        int sat_upper_th;
        int value_upper_th;
        int value_lower_th;

        bool scene_enable_y;
        float max_scene_z;
        float min_scene_z;
        float max_scene_y;
        float min_scene_y;
        float max_scene_x;
        float min_scene_x;

        float egbis_sigma;
        float egbis_k;
        int   egbis_min_size;
        float down_y_plane;
        float up_y_plane;
        float right_x_plane;
        float left_x_plane;
        float pos_z_plane;
        float neg_z_plane;

        int min_cluster_size;
        int max_cluster_size;
        float cluster_tolerance;

        bool is_cloud_transform;


        pcl::PassThrough<PointT>::Ptr pass_scene;
        //pcl::io::PointCloudImageExtractorFromRGBField<PointT>::Ptr rgb_extractor;
        void computeDescriptors(cv::Mat images_th, DetectorDescriptors &out);
        void drawDescriptors(DetectorDescriptors& input, cv::Mat& output);
        void cropOriginal(ClothesContainer& out);
        pcl::PointCloud<PointT>::Ptr removeNormalPlane(const pcl::PointCloud<PointT>::Ptr &cloud);
        pcl::PointIndices::Ptr getNormalPlaneInliers(const pcl::PointCloud<PointT>::Ptr &cloud);
        pcl::PointCloud<PointT>::Ptr cropCloudInArea(const pcl::PointCloud<PointT>::Ptr &cloud, bool keep_organized = false);
        void findCroppedAreaFromCloud(const pcl::PointCloud<PointT>::Ptr &cloud );
        void changeNaN2Black(const pcl::PointCloud<PointT>::Ptr &cloud );
        pcl::PointCloud<PointT>::Ptr filterScene(const pcl::PointCloud<PointT>::Ptr &cloud);
        bool extractRGBFromCloud(const pcl::PointCloud<PointT>& cloud, pcl::PCLImage& img);


};


#endif //CLOTHES_DETECTOR_CLOTHES_DETECTOR_H
