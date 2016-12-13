//
// Created by kandithws on 28/2/2559.
//

#include <ClothesDetector.h>



using namespace cv;


ClothesDetector::ClothesDetector():
        pass_scene(new pcl::PassThrough<PointT>())
        //rgb_extractor(new pcl::io::PointCloudImageExtractorFromRGBField<PointT>())
{
    this->sat_lower_th = 0;
    this->sat_upper_th = 15;
    this->value_lower_th = 200;
    this->value_upper_th = 255;
    this->egbis_sigma = 0.2;
    this->egbis_k = 500;
    this->egbis_min_size = 100;
    this->down_y_plane = 9999;
    this->up_y_plane = -9999;
    this->right_x_plane = -9999;
    this->left_x_plane = 9999;
    this->pos_z_plane = -9999;
    this->neg_z_plane = 9999;
    this->min_cluster_size = 10000;
    this->max_cluster_size = 25000;
    this->cluster_tolerance = 0.02;
    this->max_scene_z = 1.5;
    this->min_scene_z = 0.3;
    this->max_scene_y = -2.0f;
    this->min_scene_y = 0.0;
    this->scene_enable_y = false;
    this->is_cloud_transform = false;
}

ClothesDetector::DetectorDescriptors::DetectorDescriptors()
{
    this->type = UNKNOWN;
    this->dominant_color = UNKNOWN;
}

void ClothesDetector::setOriginalImage(cv::Mat img)
{
    this->original = img;
}

void ClothesDetector::setWhiteColorThreshold(int sat_lower, int  sat_upper,
                                             int  value_lower, int  value_upper)
{
    this->sat_lower_th = sat_lower;
    this->sat_upper_th = (sat_upper<=100)?(sat_upper):(100);
    this->value_lower_th = value_lower;
    this->value_upper_th = value_upper;
}

void ClothesDetector::setEgbisConstraint(float sigma, float k, int min_size)
{
    this->egbis_sigma = sigma; //Gaussian Blur Variance
    this->egbis_k = k; //Algorithm Gain
    this->egbis_min_size = min_size; //Minimum Area foreach Segment
}

void ClothesDetector::setPlaneSearchSpace(float min_z, float max_z, bool y_enable, float min_y, float max_y)
{
    this->is_cloud_transform = false;
    this->min_scene_z = min_z;
    this->max_scene_z = max_z;
    this->scene_enable_y = y_enable;
    if(this->scene_enable_y)
    {
        this->min_scene_y = min_y;
        this->max_scene_y = max_y;
    }
}

void ClothesDetector::setPlaneSearchSpaceCloudTF(float min_z, float max_z, float min_x, float max_x)
{
    this->is_cloud_transform = true;
    this->min_scene_z = min_z;
    this->max_scene_z = max_z;
    this->min_scene_x = min_x;
    this->min_scene_x = max_x;

}

void ClothesDetector::DetectorDescriptors::copyTo(DetectorDescriptors &target)
{
    this->contour_area = target.contour_area;
    this->centroid = target.centroid;
    this->contour.clear();
    this->contour.insert(this->contour.begin(), target.contour.begin(), target.contour.end());
    this->mask = cv::Mat::zeros(target.mask.rows, target.mask.cols, target.mask.type());
    if(!target.mask.empty())
        target.mask.copyTo(this->mask);
    if(!target.cropped.empty())
       target.cropped.copyTo(this->cropped);
    this->rect = target.rect;
    this->type = target.type;
}

void ClothesDetector::setClusteringConstraint(float tolerance, int min_size, int max_size)
{
    this->cluster_tolerance = tolerance;
    this->min_cluster_size = min_size;
    this->max_cluster_size = max_size;
}



void ClothesDetector::extractPlaneImage(pcl::PointCloud<PointT>::Ptr cloud, pcl::PCLImage& output, pcl::PCLImage& original_img)
{
    pcl::PointCloud<PointT>::Ptr cloud_for_segmentation (new pcl::PointCloud<PointT>(*cloud));
    pcl::PointCloud<PointT>::Ptr cloud_plane1(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_plane2(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_plane3(new pcl::PointCloud<PointT>());
    pcl::NormalEstimation<PointT, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<PointT, pcl::Normal>());
    boost::shared_ptr<pcl::SACSegmentationFromNormals<PointT, pcl::Normal> > seg(new pcl::SACSegmentationFromNormals<PointT, pcl::Normal>());
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::ExtractIndices<PointT>::Ptr extract(new pcl::ExtractIndices<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::ModelCoefficients::Ptr coefficients_plane(new  pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices());
    pcl::PassThrough<PointT>::Ptr pass(new pcl::PassThrough<PointT>());
    float Max_x = -9999, Max_y = -9999, Max_z = -9999;
    float Min_x = 9999, Min_y = 9999, Min_z = 9999;
    //rgb_extractor->extract(*cloud, original_img);
    this->extractRGBFromCloud(*cloud, original_img);
    // Build a passthrough filter to remove spurious NaNs
    cloud_filtered = this->filterScene(cloud);

    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

    ne->setSearchMethod (tree);
    ne->setKSearch (50);
    // Estimate point normals
    ne->setInputCloud (cloud_filtered);
    ne->compute (*cloud_normals);

    // Create the segmentation object for the planar model and set all the parameters
    seg->setInputCloud (cloud_filtered);
    seg->setInputNormals (cloud_normals);
    seg->setOptimizeCoefficients (true);
    seg->setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg->setNormalDistanceWeight (0.1);
    seg->setMethodType (pcl::SAC_RANSAC);
    seg->setMaxIterations (100);
    seg->setDistanceThreshold (0.03);
    // Obtain the plane inliers and coefficients
    seg->segment (*inliers_plane, *coefficients_plane);

    // Extract the planar inliers from the input cloud
    extract->setInputCloud (cloud_filtered);
    extract->setIndices (inliers_plane);
    extract->setNegative (false);
    extract->filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
    this->findCroppedAreaFromCloud(cloud_plane);
    cloud_plane3 = this->cropCloudInArea(cloud, true);
    std::cout << "Input W x H = " << cloud->width << " x " << cloud->height << std::endl;
    std::cout << "Segmented Cloud W x H = " << cloud_plane3->width << " x " << cloud_plane3->height << std::endl;
    std::cout << "Input Size = " << cloud->size() << " x " << cloud->height << std::endl;
    std::cout << "Segmented Cloud Size = " << cloud_plane3->size() << " x " << cloud_plane3->height << std::endl;
    //If NaN change rgb to Black Color;
    this->changeNaN2Black(cloud_plane3);
    //rgb_extractor->extract(*cloud_plane3, output);
    this->extractRGBFromCloud(*cloud_plane3, output);
}

pcl::PointCloud<PointT>::Ptr ClothesDetector::extractPlaneCloud(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::NormalEstimation<PointT, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<PointT, pcl::Normal>());
    boost::shared_ptr<pcl::SACSegmentationFromNormals<PointT, pcl::Normal> > seg(new pcl::SACSegmentationFromNormals<PointT, pcl::Normal>());
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::ExtractIndices<PointT>::Ptr extract(new pcl::ExtractIndices<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::ModelCoefficients::Ptr coefficients_plane(new  pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices());
    pcl::PassThrough<PointT>::Ptr pass(new pcl::PassThrough<PointT>());
    float Max_x = -9999, Max_y = -9999, Max_z = -9999;
    float Min_x = 9999, Min_y = 9999, Min_z = 9999;
    //rgb_extractor->extract(*cloud, original_img);
    //this->extractRGBFromCloud(*cloud, original_img);
    // Build a passthrough filter to remove spurious NaNs
    cloud_filtered = this->filterScene(cloud);

    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

    ne->setSearchMethod (tree);
    ne->setKSearch (50);
    // Estimate point normals
    ne->setInputCloud (cloud_filtered);
    ne->compute (*cloud_normals);

    // Create the segmentation object for the planar model and set all the parameters
    seg->setInputCloud (cloud_filtered);
    seg->setInputNormals (cloud_normals);
    seg->setOptimizeCoefficients (true);
    seg->setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg->setNormalDistanceWeight (0.1);
    seg->setMethodType (pcl::SAC_RANSAC);
    seg->setMaxIterations (100);
    seg->setDistanceThreshold (0.03);
    // Obtain the plane inliers and coefficients
    seg->segment (*inliers_plane, *coefficients_plane);

    // Extract the planar inliers from the input cloud
    extract->setInputCloud (cloud_filtered);
    extract->setIndices (inliers_plane);
    extract->setNegative (false);
    extract->setKeepOrganized(true);
    extract->filter (*cloud_plane);
    return cloud_plane;
}

void ClothesDetector::extractClustersImages(pcl::PointCloud<PointT>::Ptr cloud, std::vector<pcl::PCLImage>& output,
                                            pcl::PCLImage& original_img, std::string debug)
{
    output.clear();
    pcl::PCDWriter writer;
    this->down_y_plane = 9999;
    this->up_y_plane = -9999;
    this->right_x_plane = -9999;
    this->left_x_plane = 9999;
    this->pos_z_plane = -9999;
    this->neg_z_plane = 9999;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    //rgb_extractor->extract(*cloud, original_img);
    this->extractRGBFromCloud(*cloud, original_img);
    // Build a passthrough filter to remove spurious NaNs
    cloud_filtered = this->filterScene(cloud);
    std::cout << "PassThrough; PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;
    pcl::PointCloud<PointT>::Ptr remove_plane_cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr plane_area_cloud(new pcl::PointCloud<PointT>());
    remove_plane_cloud = this->removeNormalPlane(cloud_filtered);
    std::cout << "Complete Normal Plane Extraction" << std::endl;
    plane_area_cloud = this->cropCloudInArea(remove_plane_cloud);
    writer.write<PointT> (debug + "plane_area_cloud.pcd", *plane_area_cloud, false);

    //Finding Clusters and Extract its RGB
    // Creating the KdTree object for the search method of the extraction
    tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
    tree->setInputCloud (plane_area_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    //pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
    pcl::EuclideanClusterExtraction<PointT> ec;
    //ec.setClusterTolerance (0.02); // 2cm
    ec.setClusterTolerance (this->cluster_tolerance);
    ec.setMinClusterSize (this->min_cluster_size);
    ec.setMaxClusterSize (this->max_cluster_size);
    ec.setSearchMethod (tree);
    ec.setInputCloud (plane_area_cloud);
    ec.extract (cluster_indices);

    std::cout << "Complete ECE We have = " << cluster_indices.size() << " clusters" << std::endl;
    int l = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
        std::vector<uint32_t> labels_cropped;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            cloud_cluster->points.push_back (plane_area_cloud->points[*pit]);
        }
        std::cout << "Labels_cropped Size =" << labels_cropped.size() << std::endl;
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        this->findCroppedAreaFromCloud(cloud_cluster);
        pcl::PointCloud<PointT>::Ptr cropped(new pcl::PointCloud<PointT>);
        cropped = this->cropCloudInArea(cloud, true);
        this->changeNaN2Black(cropped);
        pcl::PCLImage out;
        //rgb_extractor->extract(*cropped, out);
        this->extractRGBFromCloud(*cropped, out);
        output.push_back(out);


        std::cout << "Cluster " << l <<", total pointcloud size: " << cloud_cluster->points.size () << " data points." << std::endl;
        l++;
    }
}

void ClothesDetector::extractClustersFineCroppedImages(pcl::PointCloud<PointT>::Ptr cloud, std::vector<pcl::PCLImage>& output,
                       pcl::PCLImage& original_img , bool find_only_max_clusters, std::string debug)

{
    output.clear();
    pcl::PCDWriter writer;
    unsigned long max_size_cluster = 0;
    unsigned long max_size_index = 0;
    unsigned long total_clusters = 0;
    this->down_y_plane = 9999;
    this->up_y_plane = -9999;
    this->right_x_plane = -9999;
    this->left_x_plane = 9999;
    this->pos_z_plane = -9999;
    this->neg_z_plane = 9999;

    this->extractRGBFromCloud(*cloud, original_img);
    cloud = this->filterScene(cloud);
    // Estimate Normals
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    pcl::EdgeAwarePlaneComparator<PointT, pcl::Normal>::Ptr edge_aware_comparator_(new pcl::EdgeAwarePlaneComparator<PointT, pcl::Normal>());
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud (cloud);
    ne.compute (*normal_cloud);
    float* distance_map = ne.getDistanceMap ();
    boost::shared_ptr<pcl::EdgeAwarePlaneComparator<PointT,pcl::Normal> > eapc = boost::dynamic_pointer_cast<pcl::EdgeAwarePlaneComparator<PointT,pcl::Normal> >(edge_aware_comparator_);
    eapc->setDistanceMap (distance_map);
    eapc->setDistanceThreshold (0.01f, false);

    // Segment Planes
    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;
    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    //pcl::OrganizedConnectedComponentSegmentation
    mps.setInputNormals (normal_cloud);
    mps.setInputCloud (cloud);

    //mps.segment (regions);
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);


    //Segment Objects
    pcl::PointCloud<PointT>::CloudVectorType clusters;
    pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
            euclidean_cluster_comparator_(new pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>());

    if (regions.size () > 0)
    {
        std::vector<bool> plane_labels;
        plane_labels.resize (label_indices.size (), false);
        std::cout << "label indices = " << label_indices.size() << std::endl;
        for (size_t i = 0; i < label_indices.size (); i++)
        {
            if(label_indices[i].indices.size () > 10000) // Minimum Plane Size
            {
                plane_labels[i] = true;
            }
        }
        euclidean_cluster_comparator_->setInputCloud (cloud);
        euclidean_cluster_comparator_->setLabels (labels);
        euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
        euclidean_cluster_comparator_->setDistanceThreshold (this->cluster_tolerance, false);

        pcl::PointCloud<pcl::Label> euclidean_labels;
        std::vector<pcl::PointIndices> euclidean_label_indices;
        pcl::OrganizedConnectedComponentSegmentation<PointT,pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);

        euclidean_segmentation.setInputCloud (cloud);
        euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

        for (size_t i = 0; i < euclidean_label_indices.size (); i++)
        {
            if ((euclidean_label_indices[i].indices.size () > this->min_cluster_size)
                        && (euclidean_label_indices[i].indices.size () < this->max_cluster_size))
            {

                pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());

                pcl::ExtractIndices<PointT> extract;
                extract.setInputCloud (cloud);
                pcl::IndicesPtr indices_ptr(new std::vector<int>(euclidean_label_indices[i].indices));
                //*indices_ptr = euclidean_label_indices[i].indices;
                extract.setIndices(indices_ptr);
                extract.setKeepOrganized(true);
                //extract.setNegative(false);
                extract.filter (*cluster);
                this->changeNaN2Black(cluster);
                clusters.push_back(*cluster);

                if((find_only_max_clusters) && (euclidean_label_indices[i].indices.size() > max_size_cluster))
                {
                    max_size_cluster = euclidean_label_indices[i].indices.size();
                    max_size_index = total_clusters++;
                }
            }
        }

        PCL_INFO ("Got %d euclidean clusters!\n", clusters.size ());

    }
    else
        PCL_WARN("Cannot Find Clusters on normals");

    if(!clusters.empty())
    {
        if(find_only_max_clusters)
        {
            pcl::PCLImage tmp;
            pcl::PointCloud<PointT>::Ptr ptr(new pcl::PointCloud<PointT>(clusters[max_size_index]));
            this->changeNaN2Black(ptr);
            this->extractRGBFromCloud(clusters[max_size_index], tmp);
            output.push_back(tmp);
        }
        else
        {
            for(int i =  0; i < clusters.size() ; i++)
            {
                pcl::PCLImage tmp;
                pcl::PointCloud<PointT>::Ptr ptr(new pcl::PointCloud<PointT>(clusters[i]));
                this->changeNaN2Black(ptr);
                this->extractRGBFromCloud(clusters[i], tmp);
                output.push_back(tmp);
            }
        }
    }
}

int ClothesDetector::getEgbisSegmentVisualize(cv::Mat &input, cv::Mat& output)
{
    int num_ccs;
    output = egbis::runEgbisOnMat(input, this->egbis_sigma, this->egbis_k, this->egbis_min_size, &num_ccs);
    return num_ccs;
}


int ClothesDetector::getEgbisSegment(cv::Mat &input, std::vector<cv::Mat>& output, std::vector<double>& out_percent, double percent_th)
{
    int num_ccs;
    egbis::getEgbisSegment(input, output, this->egbis_sigma,
                           this->egbis_k, this->egbis_min_size, &num_ccs, out_percent, percent_th);
    return num_ccs;
}

void ClothesDetector::detectClothesObjects(std::vector<cv::Mat> &images_th, ClothesContainer& out, bool check_table, bool crop_original)
{
    int max_rect_area = 0;
    int idx = 0;
    if(!out.empty())
    {
        std::cout << "ClothesDetector::detectClothesObjects => Output vector is not empty" << std::endl;
        return;
    }
    out.resize(images_th.size());
    for(int i = 0; i < images_th.size(); i++)
    {

        DetectorDescriptors  temp_desc;
        cv::Mat temp_mat;
        images_th[i].copyTo(temp_mat);
        this->computeDescriptors(temp_mat, temp_desc);

        int area = temp_desc.rect.area();
        images_th[i].copyTo(temp_desc.mask);
        if(area > max_rect_area)
        {
            max_rect_area = area;
            idx = i;
        }
        out[i] = temp_desc;
    }

    if(check_table)
    {
        for(int i=0 ; i < out.size(); i++)
        {
            if( i == idx)
                out[i].type = TABLE;
                //out[i].type = CLOTHES;
            else
                out[i].type = CLOTHES;
        }
    }
    else
    {
        for(int i=0 ; i < out.size(); i++)
        {
            out[i].type = CLOTHES;
        }
    }

    if(crop_original)
    {
        this->cropOriginal(out);
    }

}


void ClothesDetector::saveOutputImages(ClothesContainer& images, std::string filename,
                                       bool draw_descriptors)
{
    for(int i = 0; i < images.size(); i++)
    {
        std::stringstream tempss;
        tempss << filename <<  '_' << i << ".jpg";
        imwrite(tempss.str().c_str(), images[i].cropped);
        if(draw_descriptors)
        {
            cv::Mat drawing;
            this->drawDescriptors(images[i], drawing);
            std::stringstream tempss_2;
            tempss_2 << filename << '_'  << i << "_desc.jpg";
            imwrite(tempss_2.str().c_str(), drawing);
        }
    }
}


void ClothesDetector::cropOriginal(ClothesContainer& out)
{
    if(this->original.empty())
    {
        std::cout << "Abort ClothesDetector::cropOriginal :: the original image has not been set." << std::endl;
    }
    for(int i=0 ; i < out.size(); i++)
    {
        cv::Mat temp;
        this->original.copyTo(temp, out[i].mask);
        out[i].cropped = cv::Mat(temp, out[i].rect);
    }
}

void ClothesDetector::getBinaryImage(std::vector<cv::Mat>& images, std::vector<cv::Mat>& image_th, int threshold_value,
                                     int closing_window_size, int opening_window_size, int kernel_type)
{
    if(!image_th.empty())
    {
        std::cout << " ClothesDetector::getBinaryImage => Output vector is not empty" << std::endl;
        return;
    }
    std::vector<cv::Mat> images_gray(images.size());
    const int OPERATION_OPENING = 2;
    const int OPERATION_CLOSING = 3;
    image_th.resize(images.size());


    //Morphological Kernel MORPH_RECT,MORPH_CROSS,MORPH_ELLIPSE
    Mat closing_element = getStructuringElement( kernel_type,
                                         Size( 2*closing_window_size+ 1, 2*closing_window_size+1 ),
                                         Point( closing_window_size, closing_window_size ) );

    Mat opening_element = getStructuringElement( kernel_type,
                                         Size( 2*opening_window_size + 1, 2*opening_window_size+1 ),
                                         Point( opening_window_size, opening_window_size ) );

    for(int i = 0; i< images.size(); i++)
    {
        try
        {
            cvtColor( images[i], images_gray[i], CV_BGR2GRAY );
            threshold( images_gray[i], image_th[i], threshold_value, 255, THRESH_BINARY);
            morphologyEx(image_th[i], image_th[i], OPERATION_CLOSING, closing_element);
            morphologyEx(image_th[i], image_th[i], OPERATION_OPENING, opening_element);
        }
        catch ( cv::Exception & e )
        {
            std::cout << e.what() << std::endl;
            exit(0);
        }
    }
}

bool ClothesDetector::findDominantColor(DetectorDescriptors &input, int cluster_number)
{
    if(input.cropped.empty())
        return false;

    Mat kmeans_data = Mat::zeros(input.cropped.cols*input.cropped.rows, 3, CV_32F);
    std::cout << "Cropped Image Size = " <<  input.cropped.size() << std::endl;
    std::cout << "Total input pixel = " <<  input.cropped.rows*input.cropped.cols << std::endl;
    std::vector<Mat> bgr;
    cv::split(input.cropped, bgr);

    for(int i=0; i<input.cropped.cols*input.cropped.rows; i++)
    {

        kmeans_data.at<float>(i,0) = (float)bgr[0].data[i];
        kmeans_data.at<float>(i,1) = (float)bgr[1].data[i];
        kmeans_data.at<float>(i,2) = (float)bgr[2].data[i];
    }

    cv::Mat labels_;
    cv::Mat centers;
    std::vector<int> histogram(cluster_number);
    cv::TermCriteria termcrit( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0);
    cv::kmeans(kmeans_data, cluster_number, labels_, termcrit, 3, KMEANS_PP_CENTERS, centers);
    std::cout << "labels_ size = " << labels_.size() << std::endl;
    std::cout << "centers_ size " << centers.size() << std::endl;
    for(int i=0; i < centers.rows ; i++)
        for(int j=0; j < centers.cols ; j++)
        {
            std::cout << "centers.at (" <<i << ',' << j << ") = " << centers.at<float>(i,j) << std::endl;
        }

    if(centers.empty())
        return false;

    //Counting Labels
    for(int i=0; i < labels_.rows ; i++)
        for(int j=0; j < labels_.cols ; j++)
        {
            for(int k = 0 ; k < histogram.size() ; k++)
            {
                if(k == labels_.at<int>(i,j))
                    histogram[k]++;
            }
        }

    for(int i=0; i < histogram.size() ; i++)
        std::cout << "hist.[" << i << "] = " << histogram[i] << std::endl;

    //get argmax
    int idx = 0;
    int value = 0;
    for(int i=0; i < histogram.size() ; i++)
    {
        if(histogram[i] > value)
        {
            value = histogram[i];
            idx = i;
        }
    }


    cv::Mat color(1,1, CV_8UC3);
    int fix_code_index = 0;
    float max_not_black = 0;
    for(int i=0; i < centers.rows ; i++ )
    {
        float i0 = centers.at<float>(i,0);
        float i1 = centers.at<float>(i,1);
        float i2 = centers.at<float>(i,2);
        float tmp = sqrt(i0*i0 + i1*i1 + i2*i2);
        if(max_not_black < tmp)
        {
            max_not_black = tmp;
            fix_code_index = i;
        }
    }
    idx = fix_code_index;

    //get_center from argmax
    //-------------------------------------- END FIX CODE
    color.at<cv::Vec3b>(0,0) = cv::Vec3b((uchar)centers.at<float>(idx, 0), (uchar)centers.at<float>(idx, 1),
                                         (uchar)centers.at<float>(idx, 2));
    std::cout << "BGR = " << (int)color.at<cv::Vec3b>(0,0)[0] << ", " << (int)color.at<cv::Vec3b>(0,0)[1]  << ", "
    << (int)color.at<cv::Vec3b>(0,0)[2] << std::endl;
    cvtColor(color, color, CV_BGR2HSV);

    std::cout << "HSV = " << (int)color.at<cv::Vec3b>(0,0)[0] << ", " << (int)color.at<cv::Vec3b>(0,0)[1]  << ", "
    << (int)color.at<cv::Vec3b>(0,0)[2] << std::endl;

    //Thresholding
    input.dominant_color = ( ( (uint8_t)color.at<cv::Vec3b>(0,0)[1] >= this->sat_lower_th)
                             && ( (uint8_t)color.at<cv::Vec3b>(0,0)[1] <= this->sat_upper_th )
                             && ( (uint8_t)color.at<cv::Vec3b>(0,0)[2] >= this->value_lower_th)
                             && ( (uint8_t)color.at<cv::Vec3b>(0,0)[2] <= this->value_upper_th) )
                           ?(WHITE):(NON_WHITE);

    if(input.dominant_color == WHITE)
        std::cout << "Dominant Color = WHITE" << std::endl;
    else if(input.dominant_color == NON_WHITE)
        std::cout << "Dominant Color = NON_WHITE" << std::endl;

    return true;
}

void ClothesDetector::map2DPointToPointCloud(pcl::PointCloud<PointT>::Ptr cloud, DetectorDescriptors& input, int window)
{
    if((!cloud->isOrganized()) || (input.dominant_color == UNKNOWN)) 
        return;

    int center_x = (int)input.centroid.x;
    int center_y = (int)input.centroid.y;
    //pcl::PointXYZRGBA this_point = cloud->at(center_x, center_y);
    PointT this_point = cloud->at(center_x, center_y);

    if( !(isnan(this_point.x)|| isnan(this_point.y) || isnan(this_point.z)) )
    {
        input.position.x = this_point.x;
        input.position.y = this_point.y;
        input.position.z = this_point.z;
        return;
    }
    else
    {
        for(int i = 1 - window; i < window ; i++)
            for(int j = 1 - window; j < window ; j++)
            {
                //pcl::PointXYZRGBA pts = cloud->at(center_x + i, center_y + j);
                PointT pts = cloud->at(center_x + i, center_y + j);
                if( !(isnan(pts.x)|| isnan(pts.y) || isnan(pts.z)) )
                {
                    input.position.x = pts.x;
                    input.position.y = pts.y;
                    input.position.z = pts.z;
                    return;
                }

            }
    }
}



void ClothesDetector::computeDescriptors(cv::Mat images_th, DetectorDescriptors &out)
{
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    /// Find contours
    findContours( images_th, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    /// Approximate contours to polygons + get bounding rects
    std::vector< std::vector<Point> > contours_poly( contours.size() );
    std::vector<Rect> boundRect( contours.size() );

    int max_area_index = 0;
    int max_area = 0;
    if(contours.empty())
    {
        std::cout << "Cannot Detect Contour: Abort Computing Descriptors" << std::endl;
        return;
    }

    for( int j = 0; j < contours.size(); j++ )
    {
        approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
        boundRect[j] = boundingRect( Mat(contours_poly[j]) );
        //Get the contour which has maximum rect area
        int area = boundRect[j].area();
        if(area > max_area)
        {
            max_area = area;
            max_area_index = j;
        }
    }

    out.rect = boundRect[max_area_index];
    out.contour = contours[max_area_index];
    out.contour_area = contourArea(contours[max_area_index]);
    out.type = UNKNOWN;

    //Compute Centroid (Central Moments of Shape)
    Moments mu= moments( contours[max_area_index], false );
    out.centroid = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );
}

void ClothesDetector::drawDescriptors(DetectorDescriptors& input, cv::Mat& output)
{
    this->original.copyTo(output);
    Scalar color = Scalar( 0, 0, 255);
    std::vector<std::vector<Point> > contours_poly(input.contour.size());
    try
    {
        approxPolyDP( Mat(input.contour), contours_poly, 3, true );
        drawContours( output, contours_poly, 0, color, 1, 8, std::vector<Vec4i>(), 0, Point() );
    }
    catch (cv::Exception &e)
    {
        PCL_WARN("Drawing Contour Error in Output Descriptor Image\n");
        //std::cout << "Error drawing Contour : " << e.what() <<std::endl;
    }
    circle(output, input.centroid, 3, color,4); //Marking Centroid of main segment
    rectangle( output, input.rect.tl(), input.rect.br(), color, 2, 8, 0 );
    std::string print = "Type: ";
    if(input.type == TABLE)
        print += "Table";
    else if(input.type == CLOTHES)
        print += "Clothes";
    else
        print += "Unknown";

    print += ", Color: ";

    if(input.dominant_color == WHITE)
        print += "White";
    else if(input.dominant_color == NON_WHITE)
        print += "Non_White";
    else
        print += "Unknown";

    std::cout << print << std::endl;
    putText(output, print, Point(0,25), FONT_HERSHEY_PLAIN, 2, color, 3);
}

//Credit Frank
pcl::PointCloud<PointT>::Ptr ClothesDetector::removeNormalPlane(const pcl::PointCloud<PointT>::Ptr &cloud) {
    //ROS_INFO("ClusterExtraction FIND NORMAL_PLANE");
    pcl::SACSegmentationFromNormals <PointT, pcl::Normal> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
    pcl::ExtractIndices <PointT> extract;
    pcl::NormalEstimation <PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud <pcl::Normal>);

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);


    seg.setNormalDistanceWeight(0.1);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setProbability(0.99);
    seg.setInputCloud(cloud);
    seg.setInputNormals(cloud_normals);

    seg.segment(*inliers_plane, *coefficients_plane);

//        delete seg;

    extract.setInputCloud(cloud);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);

    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_plane);


    for (int i = 0; i < cloud_plane->points.size(); i++) {
        float x = std::abs(cloud_plane->points[i].x);
        float y = std::abs(cloud_plane->points[i].y);
        float z = std::abs(cloud_plane->points[i].z);

        left_x_plane = std::min(left_x_plane, cloud_plane->points[i].x);
        down_y_plane = std::min(down_y_plane, cloud_plane->points[i].y);
        neg_z_plane = std::min(neg_z_plane, cloud_plane->points[i].z);

        right_x_plane = std::max(right_x_plane, cloud_plane->points[i].x);
        up_y_plane = std::max(up_y_plane, cloud_plane->points[i].y);
        pos_z_plane = std::max(pos_z_plane, cloud_plane->points[i].z);
    }
    std::cout << "width_plane  " << std::abs(left_x_plane - right_x_plane) << std::endl;
    std::cout << "height_plane   " << std::abs(down_y_plane - up_y_plane) << std::endl;
    std::cout << "depth_plane   " << std::abs(neg_z_plane - pos_z_plane) << std::endl;
    std::cout << "down_y_plane  " << down_y_plane << std::endl;
    std::cout << "up_y_plane  " << up_y_plane << std::endl;
    std::cout << "right_x_plane  " << right_x_plane << std::endl;
    std::cout << "left_x_plane  " << left_x_plane << std::endl;
    std::cout << "pos_z_plane  " << pos_z_plane << std::endl;
    std::cout << "neg_z_plane  " << neg_z_plane << std::endl;

    if (not cloud_plane->empty()) {
        //writer.write<pcl::PointXYZ>(this->path.str() + "cloud_plane.pcd", *cloud_plane, false);
        //ROS_INFO("Saved: %s%s", this->path.str().c_str(), "cloud_plane.pcd");
    }
   // ROS_INFO("ClusterExtraction CUT_NORMAL_PLANE");
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud <PointT>);
    extract.setNegative(true);
    extract.filter(*cloud_filtered);

    if (not cloud_filtered->empty()) {
        //writer.write<pcl::PointXYZ>(this->path.str() + "cloud_remove_plane.pcd", *cloud_filtered, false);
        //ROS_INFO("Saved: %s%s", this->path.str().c_str(), "cloud_remove_plane.pcd");
    }

    return cloud_filtered;
}

pcl::PointIndices::Ptr ClothesDetector::getNormalPlaneInliers(const pcl::PointCloud<PointT>::Ptr &cloud) {

    //Find Plane Indices and its Boundary
    pcl::SACSegmentationFromNormals <PointT, pcl::Normal> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
    pcl::ExtractIndices <PointT> extract;
    pcl::NormalEstimation <PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud <pcl::Normal>);

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);


    seg.setNormalDistanceWeight(0.1);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setProbability(0.99);
    seg.setInputCloud(cloud);
    seg.setInputNormals(cloud_normals);

    seg.segment(*inliers_plane, *coefficients_plane);

//        delete seg;

    extract.setInputCloud(cloud);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);

    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_plane);


    for (int i = 0; i < cloud_plane->points.size(); i++) {
        float x = std::abs(cloud_plane->points[i].x);
        float y = std::abs(cloud_plane->points[i].y);
        float z = std::abs(cloud_plane->points[i].z);

        left_x_plane = std::min(left_x_plane, cloud_plane->points[i].x);
        down_y_plane = std::min(down_y_plane, cloud_plane->points[i].y);
        neg_z_plane = std::min(neg_z_plane, cloud_plane->points[i].z);

        right_x_plane = std::max(right_x_plane, cloud_plane->points[i].x);
        up_y_plane = std::max(up_y_plane, cloud_plane->points[i].y);
        pos_z_plane = std::max(pos_z_plane, cloud_plane->points[i].z);
    }
    std::cout << "width_plane  " << std::abs(left_x_plane - right_x_plane) << std::endl;
    std::cout << "height_plane   " << std::abs(down_y_plane - up_y_plane) << std::endl;
    std::cout << "depth_plane   " << std::abs(neg_z_plane - pos_z_plane) << std::endl;
    std::cout << "down_y_plane  " << down_y_plane << std::endl;
    std::cout << "up_y_plane  " << up_y_plane << std::endl;
    std::cout << "right_x_plane  " << right_x_plane << std::endl;
    std::cout << "left_x_plane  " << left_x_plane << std::endl;
    std::cout << "pos_z_plane  " << pos_z_plane << std::endl;
    std::cout << "neg_z_plane  " << neg_z_plane << std::endl;


    return inliers_plane;
}



pcl::PointCloud<PointT>::Ptr ClothesDetector::cropCloudInArea(const pcl::PointCloud<PointT>::Ptr &cloud, bool keep_organized)
{
    pcl::PassThrough<PointT>::Ptr pass(new pcl::PassThrough<PointT>());
    pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>());
    pass->setKeepOrganized(keep_organized);
    pass->setInputCloud (cloud);
    pass->setFilterFieldName ("z");
    pass->setFilterLimits (this->neg_z_plane, this->pos_z_plane);
    pass->filter (*temp);
    pass->setInputCloud (temp);
    pass->setFilterFieldName ("y");
    pass->setFilterLimits (this->down_y_plane, this->up_y_plane);
    pass->filter (*temp);
    pass->setInputCloud (temp);
    pass->setFilterFieldName ("x");
    pass->setFilterLimits (this->left_x_plane, this->right_x_plane);
    pass->filter (*temp);
    return temp;
}

void ClothesDetector::findCroppedAreaFromCloud(const pcl::PointCloud<PointT>::Ptr &cloud )
{
    float Max_x = -9999, Max_y = -9999, Max_z = -9999;
    float Min_x = 9999, Min_y = 9999, Min_z = 9999;
    for (float i = 0; i < cloud->points.size (); ++i){
        if(Max_x < cloud->points[i].x){
            Max_x = cloud->points[i].x;
        }
        if(Max_y < cloud->points[i].y){
            Max_y = cloud->points[i].y;
        }

        if(Max_z < cloud->points[i].z){
            Max_z = cloud->points[i].z;
        }
        if(Min_x > cloud->points[i].x){
            Min_x = cloud->points[i].x;
        }
        if(Min_y > cloud->points[i].y){
            Min_y = cloud->points[i].y;
        }
        if(Min_z > cloud->points[i].z){
            Min_z = cloud->points[i].z;
        }
    }

    this->down_y_plane = Min_y;
    this->up_y_plane = Max_x;
    this->right_x_plane = Max_x;
    this->left_x_plane = Min_x;
    this->pos_z_plane = Max_z;
    this->neg_z_plane = Min_z;
}

void ClothesDetector::changeNaN2Black(const pcl::PointCloud<PointT>::Ptr &cloud )
{
    for (int i = 0; i < cloud->width; i++)
        for (int j = 0; j < cloud->height; j++)
        {
            if(isnan(cloud->at(i,j).x) || isnan(cloud->at(i,j).y) || isnan(cloud->at(i,j).z) )
            {
                //Change Nan's RGB to Black
                cloud->at(i,j).r = 0;
                cloud->at(i,j).g = 0;
                cloud->at(i,j).b = 0;
            }
        }
}


pcl::PointCloud<PointT>::Ptr ClothesDetector::filterScene(const pcl::PointCloud<PointT>::Ptr &cloud)
{
    pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>());
    this->pass_scene->setKeepOrganized(true);
    this->pass_scene->setInputCloud (cloud);
    this->pass_scene->setFilterFieldName ("z");
    this->pass_scene->setFilterLimits (this->min_scene_z, this->max_scene_z);
    this->pass_scene->filter (*temp);

    if(this->is_cloud_transform)
    {
        this->pass_scene->setInputCloud (temp);
        this->pass_scene->setFilterFieldName ("x");
        this->pass_scene->setFilterLimits (this->min_scene_x, this->max_scene_x);
        this->pass_scene->filter (*temp);
    }
    else
    {
        if(this->scene_enable_y)
        {
            this->pass_scene->setInputCloud (temp);
            this->pass_scene->setFilterFieldName ("y");
            this->pass_scene->setFilterLimits (this->min_scene_y, this->max_scene_y);
            this->pass_scene->filter (*temp);
        }
    }

    return temp;
}

bool ClothesDetector::extractRGBFromCloud(const pcl::PointCloud<PointT>& cloud, pcl::PCLImage& img)
{
    if (!cloud.isOrganized () || cloud.points.size () != cloud.width * cloud.height)
        return (false);

    std::vector<pcl::PCLPointField> fields;
    int field_idx = pcl::getFieldIndex (cloud, "rgba", fields);
    if (field_idx == -1)
            return (false);

    const size_t offset = fields[field_idx].offset;

    img.encoding = "rgb8";
    img.width = cloud.width;
    img.height = cloud.height;
    img.step = img.width * sizeof (unsigned char) * 3;
    img.data.resize (img.step * img.height);

    for (size_t i = 0; i < cloud.points.size (); ++i)
    {
        uint32_t val;
        pcl::getFieldValue<PointT, uint32_t> (cloud.points[i], offset, val);
        img.data[i * 3 + 0] = (val >> 16) & 0x0000ff;
        img.data[i * 3 + 1] = (val >> 8) & 0x0000ff;
        img.data[i * 3 + 2] = (val) & 0x0000ff;
    }

    return (true);
}


