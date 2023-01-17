//
// (c) Bit Parallel Ltd, January 2023
//

#ifndef ALLOTHETIC_CUDA_CROP_HPP
#define ALLOTHETIC_CUDA_CROP_HPP

#include <cstdint>
#include <driver_types.h>
#include <opencv2/core/types.hpp>

template <class PixelType>
class CudaCrop
{
    private:
        const cv::Rect inputRect, outputRect;
        const dim3 blockDim, gridDim;
        PixelType* input;
        PixelType* output;
        cudaStream_t stream;
        cudaEvent_t timerStart, timerStop;

    public:
        CudaCrop(const cv::Rect inputRect, const cv::Rect outputRect);
        CudaCrop(const cv::Rect inputRect, const cv::Rect outputRect, const cudaStream_t stream);
        PixelType* getInputBuffer() const;
        PixelType* getOutputBuffer() const;
        float execute() const;
        cudaStream_t getStream() const;
        ~CudaCrop();

    private:
        void init();
        int32_t iDivRoundUp(const int32_t x, const int32_t y) const;
};

#endif
