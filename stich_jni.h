#include "newstitchcheck.h"
#include "stitch.h"
#include <android/log.h>

#define LOG_TAG "CombinePicture"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

extern "C"
{

JNIEXPORT jboolean JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_getfeaturedata(JNIEnv *,
                                                                 jobject,
                                                                 jobject,
                                                                 jint,
                                                                 jdouble,
                                                                 jdouble);

JNIEXPORT jintArray JNICALL
Java_com_trax_jcall_AlgorithmNativeCarrier_checkimage(JNIEnv *,
                                                                     jobject,
                                                                     jobject,
                                                                     jint,
                                                                     jdouble,
                                                                     jdouble,
                                                                     jint,
                                                                     jint,
                                                                     jfloat,
                                                                     jfloat);
}