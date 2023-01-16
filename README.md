# Jetson Orin / Xavier CUDA Video Crop Example
#### An OpenCV / CUDA video crop example for the NVIDIA Orin / Xavier written in C++17

#### Build
```
mkdir build
cd build
cmake ..

#### Run
```
mkdir video
cp some-path/some-video.mp4 /video
./centre-crop video/some-video.mp4

#### Tested
```
JetPack 5.0.2
