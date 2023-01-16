//
// (c) Bit Parallel Ltd, January 2023
//

#include <cstdint>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "cuda_crop.hpp"
#include "cuda_utils.hpp"

int32_t main(int32_t argc, char** argv)
{
    // set to enable CUDA timing of the crop kernel
    //
    constexpr auto time = false;

    if (argc != 2)
    {
        std::cout << "Invalid arguments, please use:\n";
        std::cout << "./centre-crop ../video/some_video.mp4\n";
        return -1;
    }

    // centre crop 1080p to 720p
    //
    const auto inputRect = cv::Rect2i(0, 0, 1920, 1080);
    const auto outputRect = cv::Rect2i(319, 179, 1280, 720);

    cudaStream_t stream;
    cudaEvent_t timerStart, timerStop;
    uchar3* input = nullptr;
    uchar3* output = nullptr;
    try
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMallocManaged((void**)&input, inputRect.width * inputRect.height * sizeof(uchar3)));
        CUDA_CHECK(cudaMallocManaged((void**)&output, outputRect.width * outputRect.height * sizeof(uchar3)));

        float elapsedTime;
        if constexpr (time)
        {
            CUDA_CHECK(cudaEventCreate(&timerStart));
            CUDA_CHECK(cudaEventCreate(&timerStop));
        }

        const auto videoFileName = std::string(argv[1]);;
        std::cout << "Centre cropping video: " << videoFileName << "\n";

        auto video = cv::VideoCapture(videoFileName);
        auto frame = cv::Mat(inputRect.size(), CV_8UC3, input);
        while (video.read(frame))
        {
            if constexpr (time) CUDA_CHECK(cudaEventRecord(timerStart, stream));
            CUDA_CHECK(cudaCrop(input, output, inputRect, outputRect, stream));
            if constexpr (time) CUDA_CHECK(cudaEventRecord(timerStop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if constexpr (time)
            {
                CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, timerStart, timerStop));
                std::cout << "Time: " << elapsedTime << " ms\n";
            }

            auto croppedFrame = cv::Mat(outputRect.size(), CV_8UC3, output);
            cv::imshow("Centre Cropped Video", croppedFrame);

            // note, assumes a 25 fps video, adjust accordingly
            //
            if (cv::waitKey(40) == 27) break;
        }

        cv::destroyAllWindows();
    }
    catch (const std::string& ex)
    {
        std::cout << "Error: " << ex << "\n";
    }

    // not checking the return values here as it's the end of the program...
    //
    cudaStreamDestroy(stream);
    cudaFree(input);
    cudaFree(output);
    if constexpr (time)
    {
        cudaEventDestroy(timerStart);
        cudaEventDestroy(timerStop);
    }

    return 0;
}
