#include "stich_jni.h"
#include <exception>


#include<android/log.h>
featuredata temp;

extern "C"
{

JNIEXPORT jboolean JNICALL
Java_com_data100_taskmobile_ui_main_activity_MainActivity_getfeaturedata(JNIEnv *env,
                                                                 jobject obj,
                                                                 jobject image,
                                                                 jint direction,
                                                                 jdouble cutsize,
                                                                 jdouble compression_ratio)
{
    try
    {
        jclass jmat = (env)->FindClass("org/opencv/core/Mat");
        jmethodID getNativeObjAddr = (env)->GetMethodID(jmat, "getNativeObjAddr", "()J");
        jlong getimage = (env)->CallLongMethod(image, getNativeObjAddr, NULL);
        Mat myimage = Mat();
        myimage = *(Mat *)getimage;
        //LOGE("print %d %d", myimage.rows,myimage.cols);
        //LOGE("print %d", myimage.channels());

        if (myimage.empty())
        {
            jint a = 0;
            return (jboolean)a;
        }

        temp.keypoints.clear();
        temp.descriptors.release();
        temp.image.release();

        getfeaturedata(temp, myimage, (int)direction, (double)cutsize, (double)compression_ratio);
//        LOGE("print1: %d %d", temp.image.rows,temp.image.cols);
//        LOGE("print2: %d -- %lf -- %lf", direction,cutsize,compression_ratio);
        if (temp.descriptors.empty())
        {
            jint a = 0;
            return (jboolean)a;
        }
        else
        {
            jint a = 1;
            return (jboolean)a;
        }

    }

    catch(exception)
    {
        jint a = 0;
        return (jboolean)a;
    }
}

JNIEXPORT jint JNICALL
Java_com_data100_taskmobile_ui_main_activity_MainActivity_checkimage(JNIEnv *env,
                                                             jobject obj,
                                                             jobject image,
                                                             jint direction,
                                                             jdouble cutsize,
                                                             jdouble compression_ratio)
{
    Mat *myimage = new Mat;
    try
    {
        jclass jmat = (env)->FindClass("org/opencv/core/Mat");
        jmethodID getNativeObjAddr = (env)->GetMethodID(jmat, "getNativeObjAddr", "()J");
        jlong getimage = (env)->CallLongMethod(image, getNativeObjAddr, NULL);
        (*(Mat *)getimage).copyTo(*myimage);
        //LOGE("print %d %d", myimage.rows,myimage.cols);
        //LOGE("print %d", myimage.channels());
        if ((*myimage).empty())
        {
            jint a = -1;
            return a;
        }
        imagestatus *result = new imagestatus;
        cout<<temp.keypoints.size()<<endl;
//        LOGE("print %d", temp.keypoints.size());
//        LOGE("print %lf %lf", temp.keypoints[0].pt.x,temp.keypoints[0].pt.y);
//        LOGE("print %lf %lf", temp.keypoints[1].pt.x,temp.keypoints[1].pt.y);
//        LOGE("print %d %d", temp.image.rows,temp.image.cols);
        checkimage_v2(*result, temp, *myimage, (int)direction, (double)cutsize, (double)compression_ratio, 10, 20, 1.5, 0.5);
        (*myimage).release();
        delete myimage;
        jint k=(jint)(*result).direction_status;
        delete result;
        return k;
    }

    catch(exception)
    {
        LOGE("^&**&^^&*  print checkimage1 ERROR");
        delete myimage;
        jint a = -1;
        return a;
    }
    catch(Exception)
    {
        LOGE("^&**&^^&*  print checkimage2 ERROR");
        delete myimage;
        jint a = -1;
        return a;
    }
}
}
