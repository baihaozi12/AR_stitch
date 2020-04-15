#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/affine.hpp>
#include <iostream>
#include "newstitchcheck.h"
#include<time.h>


using namespace cv;
using namespace std;


int main()
{
    Mat im0 = imread("/home/nolan/Images/left.jpg");
    Mat im1 = imread("/home/nolan/Images/right.jpg");

    if(! im0.data) {
        cout << "read image error" << endl;
    }
    featuredata *basedata = new featuredata();
    getfeaturedata(*basedata, im0, 1, 1, 0.5);

    clock_t start, finish;
    start = clock();

    stitch_status *result = new stitch_status();
    check_image_v3(*result, *basedata, im1, 0, 1, 0.5, 10, 20);
    cout << result->direction_status << endl;

    for (size_t i = 0; i < result->corner.size(); i++) {
        Point2f pt = result->corner[i];
        cout << (int)pt.x << ", " << (int)pt.y << ", ";
    }
    cout << endl;

    finish = clock();
    cout << "time cost: " << (double)(finish - start) / CLOCKS_PER_SEC << endl;


    return 0;
}
