//
// (c) Bit Parallel Ltd, January 2023
//

#include "cuda_utils.hpp"
#include "cuda_crop.hpp"

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

template <class PixelType>
CudaCrop<PixelType>::CudaCrop(const cv::Rect inputRect, const cv::Rect outputRect):
    inputRect(inputRect), outputRect(outputRect),
    blockDim(dim3(8, 8)), gridDim(dim3(iDivRoundUp(outputRect.width, blockDim.x), iDivRoundUp(outputRect.height, blockDim.y))) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        init();
}

template <class PixelType>
CudaCrop<PixelType>::CudaCrop(const cv::Rect inputRect, const cv::Rect outputRect, const cudaStream_t stream):
    inputRect(inputRect), outputRect(outputRect),
    blockDim(dim3(8, 8)), gridDim(dim3(iDivRoundUp(outputRect.width, blockDim.x), iDivRoundUp(outputRect.height, blockDim.y))),
    stream(stream) {
        init();
}

template <class PixelType>
void CudaCrop<PixelType>::init()
{
    CUDA_CHECK(cudaMallocManaged((void**)&input, inputRect.width * inputRect.height * sizeof(PixelType)));
    CUDA_CHECK(cudaMallocManaged((void**)&output, outputRect.width * outputRect.height * sizeof(PixelType)));
    CUDA_CHECK(cudaEventCreate(&timerStart));
    CUDA_CHECK(cudaEventCreate(&timerStop));
}

template <class PixelType>
PixelType* CudaCrop<PixelType>::getInputBuffer() const
{
    return input;
}

template <class PixelType>
PixelType* CudaCrop<PixelType>::getOutputBuffer() const
{
    return output;
}

template <class PixelType>
float CudaCrop<PixelType>::execute() const
{
    CUDA_CHECK(cudaEventRecord(timerStart, stream));
    gpuCrop<PixelType><<<gridDim, blockDim, 0, stream>>>(input, output, outputRect.x, outputRect.y, inputRect.width, outputRect.width, outputRect.height);
    CUDA_CHECK(cudaEventRecord(timerStop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, timerStart, timerStop));
    CUDA_CHECK(cudaGetLastError());

    return elapsedTime;
}

template <class PixelType>
cudaStream_t CudaCrop<PixelType>::getStream() const
{
    return stream;
}

template <class PixelType>
CudaCrop<PixelType>::~CudaCrop()
{
    // FIXME! not checking the return values
    //        problems should be reported, but there's not much that can be done
    //
    cudaStreamDestroy(stream);
    cudaFree(input);
    cudaFree(output);

    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
}

template <class PixelType>
int32_t CudaCrop<PixelType>::iDivRoundUp(const int32_t x, const int32_t y) const
{
    return ((x % y) != 0) ? ((x / y) + 1) : (x / y);
}

// forward references as required by non-header template classes
// add others as required
//
template class CudaCrop<uchar3>;
