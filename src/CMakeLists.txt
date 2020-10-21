CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(CV02)

# add libs you need
SET(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)
SET(SRC main.cpp)

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC} cli.h colors.h)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
