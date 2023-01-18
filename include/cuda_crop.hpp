//
// (c) Bit Parallel Ltd, January 2023
//

#ifndef ALLOTHETIC_CUDA_CROP_HPP
#define ALLOTHETIC_CUDA_CROP_HPP

#include <cstdint>
#include <cuda_runtime_api.h>
#include <opencv2/core/types.hpp>

template <class PixelType>
class CudaCrop
{
    private:
        const cv::Size inputSize;
        const cv::Rect outputRect;
        const dim3 blockDim, gridDim;
        PixelType* input;
        PixelType* output;
        cudaStream_t stream;
        cudaEvent_t timerStart, timerStop;

    // notes 1, allocate managed GPU memory based on the input / output sizes, create and associate a stream
    //       2, integrate with an existing stream
    //       3, crop and existing Mat, this must be backed by managed GPU memory
    //
    public:
        CudaCrop(const cv::Size& inputSize, const cv::Rect& outputRect);
        CudaCrop(const cv::Size& inputSize, const cv::Rect& outputRect, const cudaStream_t stream);
        CudaCrop(const cv::Mat& inputMat, const cv::Rect& outputRect, const cudaStream_t stream);
        PixelType* getInputBuffer() const;
        PixelType* getOutputBuffer() const;
        float execute() const;
        cudaStream_t getStream() const;
        ~CudaCrop();

    private:
        void initDeviceMemory();
        void initDeviceEvents();
        int32_t iDivRoundUp(const int32_t x, const int32_t y) const;
};

#endif
