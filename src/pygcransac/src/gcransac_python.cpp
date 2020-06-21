#include "gcransac_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "flann_neighborhood_graph.h"
#include "grid_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"
#include "solver_homography_four_point.h"
#include "solver_essential_matrix_five_point_stewenius.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&F,
                           int h1, int w1, int h2, int w2,
                           double threshold,
                           double conf,
                           int max_iters)
{
    
      int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 4, CV_64F);
    int iterations = 0;
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = dstPts[2*i];
        points.at<double>(i, 3) = dstPts[2*i + 1];
    }
    const size_t cell_number_in_neighborhood_graph_ = 8;
    neighborhood::GridNeighborhoodGraph neighborhood1(&points,
		w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h2 / static_cast<double>(cell_number_in_neighborhood_graph_),
//		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
//		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
//		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
//		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood1.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));
	
	// Apply Graph-cut RANSAC
	utils::DefaultFundamentalMatrixEstimator estimator;
	FundamentalMatrix model;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(w1), // The width of the source image
		static_cast<double>(h1), // The height of the source image
		static_cast<double>(w2), // The width of the destination image
		static_cast<double>(h2));  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}
	
	GCRANSAC<utils::DefaultFundamentalMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = 0.0005 * threshold * max_image_diagonal; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.14; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	//printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	
    const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();

  
    inliers.resize(num_tents);
   
    const int num_inliers = statistics.inliers.size();
    for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
        inliers[pt_idx] = 0;
    
    }
    for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
        inliers[statistics.inliers[pt_idx]] = 1;
    }
    
    
    F.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            F[i*3+j] = (double)model.descriptor(i,j);
        }
    }
    return num_inliers;
}

int findEssentialMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&E,
                         std::vector<double>& src_K,
                           std::vector<double>& dst_K,
                         
                           int h1, int w1, int h2, int w2,
                           double threshold,
                           double conf,
                           int max_iters)
{
    
      int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 4, CV_64F);
    int iterations = 0;
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = dstPts[2*i];
        points.at<double>(i, 3) = dstPts[2*i + 1];
    }
    const size_t cell_number_in_neighborhood_graph_ = 8;
    neighborhood::GridNeighborhoodGraph neighborhood1(&points,
		w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h2 / static_cast<double>(cell_number_in_neighborhood_graph_),
//		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
//		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
//		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
//		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood1.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));
	
        
        	Eigen::Matrix3d intrinsics_src,
		intrinsics_dst;

            
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            intrinsics_src(i,j) = src_K[i*3+j];
        }
    }
    
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            intrinsics_dst(i,j) = dst_K[i*3+j];
        }
    }
        
        	cv::Mat normalized_points(points.size(), CV_64F);
	utils::normalizeCorrespondences(points,
		intrinsics_src,
		intrinsics_dst,
		normalized_points);
    
	// Apply Graph-cut RANSAC
	utils::DefaultEssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	EssentialMatrix model;
	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
	static_cast<double>(w1), // The width of the source image
		static_cast<double>(h1), // The height of the source image
		static_cast<double>(w2), // The width of the destination image
		static_cast<double>(h2));  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}
	
	GCRANSAC<utils::DefaultEssentialMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.14; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(normalized_points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	// Get the statistics of the results
	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();

  
    inliers.resize(num_tents);
   
    const int num_inliers = statistics.inliers.size();
    for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
        inliers[pt_idx] = 0;
    
    }
    for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
        inliers[statistics.inliers[pt_idx]] = 1;
    }
    
    
    E.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            E[i*3+j] = (double)model.descriptor(i,j);
        }
    }
    return num_inliers;
}


int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    int h1, int w1, int h2, int w2,
                           double threshold,
                           double conf,
                           int max_iters)

{
     const size_t cell_number_in_neighborhood_graph_ = 8;
    
      int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 4, CV_64F);
    int iterations = 0;
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = dstPts[2*i];
        points.at<double>(i, 3) = dstPts[2*i + 1];
    }
    
    neighborhood::GridNeighborhoodGraph neighborhood1(&points,
		w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h2 / static_cast<double>(cell_number_in_neighborhood_graph_),
//		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
//		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
//		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
//		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood1.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));
	
    utils::DefaultHomographyEstimator estimator;
	Homography model;

	GCRANSAC<utils::DefaultHomographyEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	
    gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.14; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

    	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(w1), // The width of the source image
		static_cast<double>(h1), // The height of the source image
		static_cast<double>(w2), // The width of the destination image
		static_cast<double>(h2),  // The height of the destination image
		0.5); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

        
    const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
    
    
    H.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            H[i*3+j] = model.descriptor(i,j);
        }
    }	

  
    inliers.resize(num_tents);
   
    const int num_inliers = statistics.inliers.size();
    for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
        inliers[pt_idx] = 0;
    
    }
    for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
        inliers[statistics.inliers[pt_idx]] = 1;
    }

    return num_inliers;
    
}
