//
// (c) Bit Parallel Ltd, January 2023
//

#ifndef ALLOTHETIC_CUDA_CROP_HPP
#define ALLOTHETIC_CUDA_CROP_HPP

#include <driver_types.h>
#include <opencv2/core/types.hpp>

cudaError_t cudaCrop(uchar3* input, uchar3* output, const cv::Rect inputRect, const cv::Rect outputRect, const cudaStream_t stream);

#endif
