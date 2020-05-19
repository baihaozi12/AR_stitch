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

JNIEXPORT jintArray JNICALL
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
        if ((*myimage).empty())
        {
            jintArray kk = env -> NewIntArray(9);
            jint p[9] = {-1, 0, 0, 0, 0, 0, 0, 0, 0};
            env->SetIntArrayRegion(kk, 0, 9, p);
            //delete []p;
            return kk;
        }
        stitch_status *result = new stitch_status;
        cout<<temp.keypoints.size()<<endl;
        check_image_v2(*result, temp, *myimage, (int)direction, (double)cutsize, (double)compression_ratio, 10, 20, 3, 1.5);
        (*myimage).release();
        delete myimage;
        jintArray kk = env -> NewIntArray(9);

        jint p[9];
        int count = 0;
        p[count++] = (jint)(*result).direction_status;
        for (size_t i = 0; i < result->corner.size(); i++) {
            Point2f pt = result->corner[i];
            p[count++] = (jint)pt.x;
            p[count++] = (jint)pt.y;
        }
        env->SetIntArrayRegion(kk, 0, 9, p);
        delete result;

        //delete []p;
        return kk;
    }

    catch(...)
    {
        LOGE("^&**&^^&*  print checkimage1 ERROR");
        delete myimage;
        jintArray kk = env -> NewIntArray(9);
        jint p[9] = {-1, 0, 0, 0, 0, 0, 0, 0, 0};
        env->SetIntArrayRegion(kk, 0, 9, p);
        //delete []p;
        return kk;
    }
}
}
