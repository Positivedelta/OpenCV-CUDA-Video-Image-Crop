//
// (c) Bit Parallel Ltd, January 2023
//

#ifndef ALLOTHETIC_CUDA_UTILS_HPP
#define ALLOTHETIC_CUDA_UTILS_HPP

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            throw "CUDA status #" + std::to_string(error_code) + " at " + __FILE__ + ":" + std::to_string(__LINE__);\
        }\
    }
#endif
#endif
