//
// (c) Bit Parallel Ltd, January 2023
//

#include "cuda_crop.hpp"

inline int32_t iDivRoundUp(int32_t x, int32_t y)
{
    return ((x % y) != 0) ? ((x / y) + 1) : (x / y);
}

template<typename T>
__global__ void gpuCrop(T* input, T* output, const int32_t offsetX, const int32_t offsetY, const int32_t inputWidth, const int32_t outputWidth, const int32_t outputHeight)
{
    const int32_t outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t outputY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outputX >= outputWidth || outputY >= outputHeight) return;

    const int32_t inputX = outputX + offsetX;
    const int32_t inputY = outputY + offsetY;
    output[(outputY * outputWidth) + outputX] = input[(inputY * inputWidth) + inputX];
}


template<typename T>
cudaError_t launchCrop(T* input, T* output, const cv::Rect inputRect, const cv::Rect outputRect, const cudaStream_t stream)
{
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivRoundUp(outputRect.width, blockDim.x), iDivRoundUp(outputRect.height, blockDim.y));
    gpuCrop<T><<<gridDim, blockDim, 0, stream>>>(input, output, outputRect.x, outputRect.y, inputRect.width, outputRect.width, outputRect.height);

    return cudaGetLastError();
}

cudaError_t cudaCrop(uchar3* input, uchar3* output, const cv::Rect inputRect, const cv::Rect outputRect, const cudaStream_t stream)
{
	return launchCrop<uchar3>(input, output, inputRect, outputRect, stream);
}
