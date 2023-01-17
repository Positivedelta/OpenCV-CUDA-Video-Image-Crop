//
// (c) Bit Parallel Ltd, January 2023
//

#include <cstdint>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "cuda_crop.hpp"

int32_t main(int32_t argc, char** argv)
{
    constexpr auto reportExecutionTime = true;

    if (argc != 2)
    {
        std::cout << "Invalid arguments, please use:\n";
        std::cout << "./centre-crop ../video/some_video.mp4\n";
        return -1;
    }

    // centre crop 1080p down to 720p
    //
    const auto inputRect = cv::Rect2i(0, 0, 1920, 1080);
    const auto outputRect = cv::Rect2i(319, 179, 1280, 720);
    const auto videoFileName = std::string(argv[1]);;
    auto video = cv::VideoCapture(videoFileName);

    try
    {
        // notes 1, images are BRG / RGB, i.e. each composite pixel can be represented using uchar3
        //       2, there is also a constructor also takes a cuda stream to allow integration with other cuda requirements
        //
        auto cudaCrop = CudaCrop<uchar3>(inputRect, outputRect);
        auto fullFrame = cv::Mat(inputRect.size(), CV_8UC3, cudaCrop.getInputBuffer());
        auto croppedFrame = cv::Mat(outputRect.size(), CV_8UC3, cudaCrop.getOutputBuffer());
        while (video.read(fullFrame))
        {
            auto elapsedTime = cudaCrop.execute();
            if constexpr (reportExecutionTime) std::cout << "Crop time: " << elapsedTime << " ms\n";

            // note, assumes a 25 fps video, adjust accordingly
            //
            cv::imshow("Centre Cropped Video", croppedFrame);
            if (cv::waitKey(40) == 27) break;
        }
    }
    catch (const std::string& ex)
    {
        std::cout << "Error: " << ex << "\n";
    }

    cv::destroyAllWindows();

    return 0;
}
