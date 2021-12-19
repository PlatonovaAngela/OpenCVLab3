TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

INCLUDEPATH += D:\opencv\build\include
INCLUDEPATH += D:\open_cv\opencv_contrib-master\modules\xfeatures2d\include

LIBS += D:\build_opencv\bin\libopencv_core454.dll
LIBS += D:\build_opencv\bin\libopencv_highgui454.dll
LIBS += D:\build_opencv\bin\libopencv_imgcodecs454.dll
LIBS += D:\build_opencv\bin\libopencv_imgproc454.dll
LIBS += D:\build_opencv\bin\libopencv_features2d454.dll
LIBS += D:\build_opencv\bin\libopencv_calib3d454.dll
LIBS += D:\build_opencv\bin\libopencv_features2d454.dll
LIBS += D:\build_opencv\bin\libopencv_xfeatures2d454.dll
