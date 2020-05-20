#include "newstitchcheck.h"
#include "exception"
#include <stdlib.h>
#define GOODMATCHNUMBER 20
#define n_max 600;


class MyPoint
{
public:
    double x, y, z;
    MyPoint(double x=0.0, double y=0.0, double z=1.0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    void calculatenewpoint(Mat& H)
    {
        Mat point_old = (Mat_<double>(3, 1) << this->x, this->y, this->z);
        Mat point_new = H * point_old;
        point_new /= point_new.at<double>(2, 0);
        this->x = point_new.at<double>(0, 0);
        this->y = point_new.at<double>(1, 0);
        this->z = point_new.at<double>(2, 0);
    }
};


class Corner
{
public:
    MyPoint ltop;
    MyPoint lbottom;
    MyPoint rtop;
    MyPoint rbottom;

    void calculatefromimage(Mat& img)
    {
        int rows = img.rows;
        int cols = img.cols;
        this->ltop.x = 0.0;
        this->ltop.y = 0.0;
        this->lbottom.x = 0.0;
        this->lbottom.y = float(rows);
        this->rtop.x = float(cols);
        this->rtop.y = 0.0;
        this->rbottom.x = float(cols);
        this->rbottom.y = float(rows);
    }

    void calculatefromhomo(Mat& H)
    {
        this->ltop.calculatenewpoint(H);
        this->lbottom.calculatenewpoint(H);
        this->rtop.calculatenewpoint(H);
        this->rbottom.calculatenewpoint(H);
    }

};


int calculatecorners(Corner& c, Mat& img, Mat& H) {
    c.calculatefromimage(img);
    c.calculatefromhomo(H);
    return 0;
}


int gethomoandmask_v3(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints, int direction, int h_, int w_, double cutsize, int match_num)
{
    result.mask.clear();
    vector<Point2f> imagePoints1, imagePoints2;
    double ratio;
    double delta = 0;
    if (direction == 0 or direction == 1) {
        ratio = (double)(h_) / n_max;
        int w = (int)(w_ / ratio);
        delta = w * (1 - cutsize);
    } else {
        ratio = (double)(w_) / n_max;
        int h = (int)(h_ / ratio);
        delta = h * (1 - cutsize);
    }
    if (GoodMatchePoints.size() < match_num) { return 1; }
    for (auto & GoodMatchePoint : GoodMatchePoints) {
        Point2f pt1 = keyPts1[GoodMatchePoint.queryIdx].pt;
        Point2f pt2 = keyPts2[GoodMatchePoint.trainIdx].pt;
        if (direction == 0) {
            pt1.x += delta;
        } else if (direction == 1) {
            pt2.x += delta;
        } else if (direction == 2) {
            pt1.y += delta;
        } else if (direction == 3) {
            pt2.y += delta;
        }
        pt1.x = pt1.x * ratio;
        pt1.y = pt1.y * ratio;
        pt2.x = pt2.x * ratio;
        pt2.y = pt2.y * ratio;

        imagePoints1.push_back(pt1);
        imagePoints2.push_back(pt2);
    }
    if (imagePoints1.size() != imagePoints2.size() && imagePoints1.size() < match_num && imagePoints2.size() < match_num) {
        return 1;
    }
    vector<uchar> mask;
    Mat homo = (Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    homo = findHomography(imagePoints1, imagePoints2, RANSAC, 5.0, mask);
    if (!homo.empty() && homo.rows == 3 && homo.cols == 3) {
        result.homo = homo;
    }
    result.mask = mask;
    return 0;
}

int check_image_v2(stitch_status &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio, int match_num1, int match_num2, double threshold1, double threshold2)
{

    result.direction_status = 0;
    result.homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    try {
        featuredata checkdata;
        vector<KeyPoint> keypoints;
        Mat descriptors;
//        Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
//        Ptr<AKAZE> f2d = AKAZE::create();

        Size target_size;
        target_size.width = 1080;
        target_size.height = 1920;
        resize(image, image, target_size);

        
        Ptr<cv::xfeatures2d::SiftFeatureDetector> f2d = cv::xfeatures2d::SiftFeatureDetector::create(1000);

        int step = 10;
//        vector<KeyPoint> kps;
        for (int i=step; i<image.rows-step; i+=step)
        {
            for (int j=step; j<image.cols-step; j+=step)
            {
                // x,y,radius
//                kps.push_back(KeyPoint(float(j), float(i), float(step)));
//                keypoints.push_back(KeyPoint(int(j), int(i), int(step)))
                keypoints.push_back(KeyPoint(float(j), float(i), float(step)));
            }
        }

        LoadImage(checkdata.image, image, direction, cutsize, compression_ratio);
        f2d->detectAndCompute(checkdata.image, noArray(), checkdata.keypoints, checkdata.descriptors);
        if (checkdata.keypoints.size() < match_num1) {
            return 0;
        }

//        BFMatcher matcher;
        FlannBasedMatcher matcher;

        vector<vector<DMatch>> matchePoints12;
        vector<DMatch> goodmatchpoints;
        if (basedata.descriptors.rows < 1 || checkdata.descriptors.rows < 1) {
            return 0;
        }
        matcher.knnMatch(basedata.descriptors, checkdata.descriptors, matchePoints12, 2);
        for (size_t i = 0; i < matchePoints12.size(); i++) {
            if (matchePoints12[i][0].distance < 0.75 * matchePoints12[i][1].distance) {
                goodmatchpoints.push_back(matchePoints12[i][0]);
            }
        }

        homoandmask hmdata;
        gethomoandmask_v3(hmdata, basedata.keypoints, checkdata.keypoints, goodmatchpoints, direction, image.rows,
                          image.cols, cutsize, match_num1);//计算单应性矩阵

        vector<DMatch> lastmatchpoints;
        for (size_t i = 0; i < hmdata.mask.size(); i++) {
            if (hmdata.mask[i] != (uchar) 0) {
                lastmatchpoints.push_back(goodmatchpoints[i]);
            }
        }

        if (lastmatchpoints.size() < match_num1) {
            return 0;
        }

        Corner c;
        calculatecorners(c, basedata.image, hmdata.homo);
        result.homo = hmdata.homo;
        result.corner = vector<Point2f>({Point2f(c.ltop.x, c.ltop.y), Point2f(c.lbottom.x, c.lbottom.y),
                                         Point2f(c.rbottom.x, c.rbottom.y), Point2f(c.rtop.x, c.rtop.y)});

        if (lastmatchpoints.size() >= match_num1 && lastmatchpoints.size() < match_num2) {
            result.direction_status = 1;
            return 0;
        }

        if (lastmatchpoints.size() >= match_num2) {
            result.direction_status = 2;

            if (abs(result.corner[0].x - result.corner[2].x) > threshold1 * image.cols ||
                abs(result.corner[0].y - result.corner[2].y) > threshold1 * image.rows ||
                abs(result.corner[1].x - result.corner[3].x) > threshold1 * image.cols ||
                abs(result.corner[1].y - result.corner[3].y) > threshold1* image.rows) {
                result.direction_status = 1;
            }


            switch (direction) {
                case 0:
                    if (abs(result.corner[2].y - float(image.rows)) + abs(result.corner[3].y) > threshold2 * image.rows ||
                    abs(result.corner[3].x) > float(image.cols) ||
                    abs(result.corner[2].x) > float(image.cols)) {
                        result.direction_status = -2;
                    }
                    break;
                case 1:
                    if (abs(result.corner[1].y - float(image.rows)) + abs(result.corner[0].y) > threshold2 * image.rows ||
                    result.corner[0].x < 0 ||
                    result.corner[1].x < 0) {
                        result.direction_status = -2;
                    }
                    break;
                case 2:
                    if (abs(result.corner[1].x) + abs(result.corner[2].x - float(image.cols)) > threshold2 * image.cols ||
                    abs(result.corner[1].y) > float(image.rows) ||
                    abs(result.corner[2].y) > float(image.rows)) {
                        result.direction_status = -2;
                    }
                    break;
                case 3:
                    if (abs(result.corner[0].x) + abs(result.corner[3].x - float(image.cols)) > threshold2 * image.cols ||
                    result.corner[0].y < 0 ||
                    result.corner[3].y < 0) {
                        result.direction_status = -2;
                    }
                    break;
                default:
                    break;
            }

            return 0;
        }
    }
    catch (...) {
        result.direction_status = -1;
        return 0;
    }
    return 0;
}

int getfeaturedata(featuredata &result, Mat &image, int direction, double cutsize, double compression_ratio)
{
    Mat *image_ = new Mat();
    try
    {
        if (image.empty())
        {
            (*image_).release();
            delete image_;
            return 0;
        }
        Size target_size;
        target_size.width = 1080;
        target_size.height = 1920;
        resize(image, image, target_size);
        LoadImage(*image_, image, direction, cutsize, compression_ratio);
        get_keypoints_and_descriptors(result, *image_);
        result.image = image;

        (*image_).release();
        delete image_;
        return 1;
    }
    catch (...)
    {
        (*image_).release();
        delete image_;
        return 0;
    }
}


int get_keypoints_and_descriptors(featuredata &result, Mat &image)
{
    try {
        Mat *M = new Mat();
        image.copyTo(*M);
        if (image.channels() == 3) {
            cvtColor(*M, *M, COLOR_RGB2GRAY);
        }
        vector<KeyPoint> *keypoints = new vector<KeyPoint>;
        Mat *descriptors = new Mat();

//        Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
//        Ptr<Feature2D> f2d = xfeatures2d::SURF::create(100, 1, 1, false, true);
//        Ptr<AKAZE> f2d = AKAZE::create();
        Ptr<cv::xfeatures2d::SiftFeatureDetector> f2d = cv::xfeatures2d::SiftFeatureDetector::create(1000);
        int step = 10;
//        vector<KeyPoint> kps;
        for (int i=step; i<image.rows-step; i+=step)
        {
            for (int j=step; j<image.cols-step; j+=step)
            {
                // x,y,radius
//                kps.push_back(KeyPoint(float(j), float(i), float(step)));
//                keypoints.push_back(KeyPoint(int(j), int(i), int(step)))
                keypoints->push_back(KeyPoint(float(j), float(i), float(step)));
            }
        }
        f2d->detectAndCompute(*M, noArray(), *keypoints, *descriptors);

        for (size_t i = 0; i < (*keypoints).size(); i++) {
            result.keypoints.push_back((*keypoints)[i]);
        }
        (*descriptors).copyTo(result.descriptors);
        (*M).copyTo(result.image);
        (*M).release();
        delete M;
        (*keypoints).clear();
        delete keypoints;
        (*descriptors).release();
        delete descriptors;
        f2d->clear();
        return 1;
    }
    catch (...) { return 0; }
}


int checkimage(imagestatus &result, featuredata& basedata, Mat& image, int direction, double cutsize, double compression_ratio)
{
    Mat *image_ = new Mat();
    image.copyTo(*image_);
    try {
        featuredata *checkdata = new featuredata();
        getfeaturedata(*checkdata, *image_, direction, cutsize, compression_ratio);//获取检测图特征信息

        if ((*checkdata).keypoints.size() < GOODMATCHNUMBER) {
            result.direction_status = -1;
            result.homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
            checkdata->descriptors.release();
            checkdata->image.release();
            checkdata->keypoints.clear();
            delete checkdata;
            (*image_).release();
            delete image_;
            return 0;
        }

        vector<DMatch> *goodmatchpoints = new vector<DMatch>;
        get_good_match_point(*goodmatchpoints, basedata.descriptors, (*checkdata).descriptors);//筛选匹配点

        homoandmask *hmdata = new homoandmask;
        gethomoandmask(*hmdata, basedata.keypoints, (*checkdata).keypoints, *goodmatchpoints);//计算单应性矩阵

        vector<DMatch> *lastmatchpoints = new vector<DMatch>;
        //        vector<Point2f> *ImagePoints1 = new vector<Point2f>, *ImagePoints2 = new vector<Point2f>;
        //计算最后匹配上的点的数量
        for (size_t i = 0; i < (*hmdata).mask.size(); i++) {
            if ((*hmdata).mask[i] != (uchar)0) {
                (*lastmatchpoints).push_back((*goodmatchpoints)[i]);
                //                (*ImagePoints1).push_back(basedata.keypoints[(*goodmatchpoints)[i].queryIdx].pt);
                //                (*ImagePoints2).push_back((*checkdata).keypoints[(*goodmatchpoints)[i].trainIdx].pt);
            }
        }

        //        cout << "Good Match: " << (*goodmatchpoints).size() << endl;
        //        cout << "Last Match: " << (*lastmatchpoints).size() << endl;;
        //        Mat img_matches;
        //        drawMatches( basedata.image, basedata.keypoints, checkdata->image, checkdata->keypoints, *goodmatchpoints, img_matches, Scalar::all(-1),
        //                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //        imwrite("good_matches.jpg", img_matches );
        //
        //        Mat img_last_matches;
        //        drawMatches( basedata.image, basedata.keypoints, checkdata->image, checkdata->keypoints, *lastmatchpoints, img_last_matches, Scalar::all(-1),
        //                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //        imwrite("last_matches.jpg", img_last_matches );

        //        result.direction_status = checkimagestatus((*checkdata).image, (*hmdata).homo, direction, 1 - cutsize);
        result.direction_status = 0;
        if ((*lastmatchpoints).size() < GOODMATCHNUMBER) {
            result.direction_status = 0;;
        }

        if ((*lastmatchpoints).size() >= GOODMATCHNUMBER) {
            result.direction_status = 1;
        }

        (*hmdata).homo.copyTo(result.homo);
        if (result.direction_status < 1) { result.homo = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); }
        checkdata->descriptors.release();
        checkdata->image.release();
        checkdata->keypoints.clear();
        delete checkdata;
        hmdata->mask.clear();
        hmdata->homo.release();
        delete hmdata;
        (*image_).release();
        delete image_;
        (*goodmatchpoints).clear();
        delete goodmatchpoints;
        return 1;
    }
    catch (exception) {
        result.direction_status = result.direction_status = -1;;
        (*image_).release();
        delete image_;
        return 0;
    }
    catch (Exception) {
        result.direction_status = result.direction_status = -1;;
        (*image_).release();
        delete image_;
        return 0;
    }
}


int computepoint(Point2f &dstpoint, Point2f &oripoint, Mat &homo)
{
    try {
        Mat *point = new Mat();
        *point = (Mat_<double>(3, 1) << oripoint.x, oripoint.y, 1.0);
        Mat *point_ = new Mat();
        *point_ = homo * (*point);
        *point_ /= abs((*point_).at<double>(2, 0)) + tmmin;
        dstpoint.x = (*point_).at<double>(0, 0);
        dstpoint.y = (*point_).at<double>(1, 0);
        (*point).release();
        delete point;
        (*point_).release();
        delete point_;
        return 1;
    }
    catch (exception) { return 0; }
    catch (Exception) { return 0; }
}


int forback(vector<Point2f> &vp, vector<Point2f> &vp_)
{
    boxdata *box = new boxdata;
    get_boxdata(*box, vp);
    double S = (box->xmax - box->xmin) * (box->ymax - box->ymin);
    boxdata *box_ = new boxdata;
    get_boxdata(*box_, vp_);
    double S_ = (box_->xmax - box_->xmin) * (box_->ymax - box_->ymin);
    delete box;
    delete box_;
    if ((S_ - S) > (0.2*S)) {
        //cout << "FORWORD" << endl;
        return 1000;
    }
    if ((S_ - S) < (-0.2*S)) {
        //cout << "BACKWORD" << endl;
        return 2000;
    }
    return 0;
}


int updown(Point2f &p, Point2f &p_, double d)
{
    double dy = p.y - p_.y;
    int result = 0;
    if (dy < -d) {
        result += 100;
    }
    if (dy > d) {
        result += 200;
    }
    return result;
}


int leftright(Point2f &p, Point2f &p_, double d)
{
    double dx = p.x - p_.x;
    int result = 0;
    if (dx < -d) {
        result += 10;
    }
    if (dx > d) {
        result += 20;
    }
    return result;
}


int rotate(Point2f &p1, Point2f &p2, double d)
{
    double dy = p1.y - p2.y;
    //cout << dy << endl;
    int result = 0;
    if (dy < -d) {
        //cout << "CLOCKWISE" << endl;
        result += 10000;
    }
    if (dy > d) {
        //cout << "COUNTERCLOCKWISE" << endl;
        result += 20000;
    }
    return result;
}


int checkimagestatus(Mat& Image, Mat& homo, int direction, double cutsize)
{
    int result = 0;
    try {
        double R = Image.rows;
        double C = Image.cols;
        Point2f  p1(0.0, 0.0), p2(C, 0.0), p3(0.0, R), p4(C, R), p5(C / 2, R / 2), p6(0, R / 2), p7(C, R / 2);
        //Point2f p1_ = computepoint(p1, homo);
        //Point2f p2_ = computepoint(p2, homo);
        //Point2f p3_ = computepoint(p3, homo);
        //Point2f p4_ = computepoint(p4, homo);
        Point2f p5_;
        computepoint(p5_, p5, homo);
        //Point2f p6_ = computepoint(p6, homo);
        //Point2f p7_ = computepoint(p7, homo);
        double PX = 0.01 * C, PY = 0.01 * R;
        if (direction == 0 || direction == 1) {
            PX = 0.06 * C * cutsize;
            PY = 0.04 * R * cutsize;
        }
        else if (direction == 2 || direction == 3) {
            PX = 0.04 * C * cutsize;
            PY = 0.06 * R * cutsize;
        }
        result += leftright(p5, p5_, PX);
        result += updown(p5, p5_, PY);
        if (result == 0) {
            result = 1;
        }
        //result += rotate(p6_, p7_, 2.0);
        //if (result == 1) {
        //    vector<Point2f> vp, vp_;
        //    vp.push_back(p1);
        //    vp.push_back(p2);
        //    vp.push_back(p3);
        //    vp.push_back(p4);
        //    vp_.push_back(p1_);
        //    vp_.push_back(p2_);
        //    vp_.push_back(p3_);
        //    vp_.push_back(p4_);
        //    result = forback(vp, vp_);
        //}
        //if (result == 0) {
        //    result = 1;
        //}
        return result;
    }
    catch (exception) { result = -1; }
    catch (Exception) { result = -1; }
    return result;
}


int cutimage(Mat& result, Mat& image, int xmin, int ymin, int xmax, int ymax)
{
    try {
        Rect rect(xmin, ymin, xmax - xmin, ymax - ymin);
        (image(rect)).copyTo(result);
        return 1;
    }
    catch (...) { return 0; }
}


int LoadImage(Mat& result, Mat& image, int direction, double cutsize, double compression_ratio)
{
    try {
        int c = image.cols;
        int r = image.rows;
        int cols;
        int rows;
        if (direction == 0 || direction == 1)
        {
//            if (r > c)
//            {
//                cutimage(result, image, 0, (int)(r * 0.2), c, (int)(r * 0.8));
//            }
//            else
//            {
//                result = image;
//            }
            result = image;
            double ratio = double(result.rows) / n_max;
            cols = (int)(result.cols / ratio);
            rows = n_max;
        }
        else if (direction == 2 || direction == 3)
        {
//            if (r < c)
//            {
//                cutimage(result, image, (int)(c * 0.2), 0, (int)(c * 0.8), r);
//            }
//            else
//            {
//                result = image;
//            }
            result = image;
            double ratio = double(result.cols) / n_max;
            cols = n_max;
            rows = (int)(result.rows / ratio);
        }
        else
        {
            return 0;
        }

        Size size = Size(cols, rows);
        resize(result, result, size, 0, 0, INTER_AREA);

        switch (direction) {
            case 0:
                cutimage(result, result, 0, 0, (int)(cols * cutsize), rows);
                break;
            case 1:
                cutimage(result, result, (int)(cols * (1 - cutsize)), 0, cols, rows);
                break;
            case 2:
                cutimage(result, result, 0, 0, cols, (int)(rows * cutsize));
                break;
            case 3:
                cutimage(result, result, 0, (int)(rows * (1 - cutsize)), cols, rows);
                break;
            default:
                break;
        }
        return 1;
    }
    catch (...) { return 0; }
}


int get_good_match_point(vector<DMatch> &result, Mat& descriptors1, Mat& descriptors2)
{
    result.clear();
        try {
        BFMatcher *matcher = new BFMatcher;
        //        FlannBasedMatcher *matcher = new FlannBasedMatcher();
        vector<vector<DMatch>> matchePoints12;
        if (descriptors1.rows < 1 || descriptors2.rows < 1) { return 0; }
        (*matcher).knnMatch(descriptors1, descriptors2, matchePoints12, 2);
        //        double mindist = matchePoints12[0][0].distance;
        for (size_t i = 0; i < matchePoints12.size(); i++) {
            //            mindist = MIN(matchePoints12[i][0].distance, mindist);
            if (matchePoints12[i][0].distance < 0.75 * matchePoints12[i][1].distance) {
                result.push_back(matchePoints12[i][0]);
            }
        }
        (*matcher).clear();
        delete matcher;
        matchePoints12.clear();
        return 1;
    }
    catch (...) { return 0; }
}


int gethomoandmask_v2(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints, int direction, Mat& image, double cutsize, int match_num)
{
    result.mask.clear();
    try {
        vector<Point2f> *imagePoints1 = new vector<Point2f>, *imagePoints2 = new vector<Point2f>;
        int h_ = image.rows;
        int w_ = image.cols;
        double ratio;
        double delta = 0;
        if (direction == 0 or direction == 1) {
            ratio = (double)(h_) / n_max;
            int w = (int)(w_ / ratio);
            delta = w * (1 - cutsize);
        } else {
            ratio = (double)(w_) / n_max;
            int h = (int)(h_ / ratio);
            delta = h * (1 - cutsize);
        }

        if (GoodMatchePoints.size() < match_num) { return 0; }
        for (auto & GoodMatchePoint : GoodMatchePoints) {
            Point2f pt1 = keyPts1[GoodMatchePoint.queryIdx].pt;
            Point2f pt2 = keyPts2[GoodMatchePoint.trainIdx].pt;


            if (direction == 0) {
                pt1.x += delta;
            } else if (direction == 1) {
                pt2.x += delta;
            } else if (direction == 2) {
                pt1.y += delta;
            } else if (direction == 3) {
                pt2.y += delta;
            }

            pt1.x = pt1.x * ratio;
            pt1.y = pt1.y * ratio;
            pt2.x = pt2.x * ratio;
            pt2.y = pt2.y * ratio;


//            cout << "@@@@" << pt1 << " " << pt2 << endl;

            (*imagePoints1).push_back(pt1);
            (*imagePoints2).push_back(pt2);
        }

        if ((*imagePoints1).size() != (*imagePoints2).size()
            && (*imagePoints1).size() < match_num
            && (*imagePoints2).size() < match_num) {
            (*imagePoints1).clear();
            delete imagePoints1;
            (*imagePoints2).clear();
            delete imagePoints2;
            return 0;
        }
        vector<uchar> mask;

        Mat homo = (Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        mask.clear();
        try {
            homo = findHomography(*imagePoints1, *imagePoints2, RANSAC, 5.0, mask);
            if (!homo.empty() && homo.rows == 3 && homo.cols == 3) {
                Rect rect(0, 0, homo.cols, homo.rows);
                homo.copyTo(result.homo(rect));

//                cout << "####" << result.homo << endl;

            }
            else { mask.clear(); }
        }
        catch (...) { mask.clear(); }
        homo.release();

        for (size_t i = 0; i < mask.size(); i++) {
            result.mask.push_back(mask[i]);
        }
        (*imagePoints1).clear();
        delete imagePoints1;
        (*imagePoints2).clear();
        delete imagePoints2;
    }
    catch (...) { return 0; }
    return 1;
}


int gethomoandmask(homoandmask &result, vector<KeyPoint> &keyPts1, vector<KeyPoint> &keyPts2, vector<DMatch> &GoodMatchePoints)
{
    result.mask.clear();
    try {
        vector<Point2f> *imagePoints1 = new vector<Point2f>, *imagePoints2 = new vector<Point2f>;
        if (GoodMatchePoints.size() < GOODMATCHNUMBER) { return 0; }
        for (size_t i = 0; i < GoodMatchePoints.size(); i++) {
            (*imagePoints1).push_back(keyPts1[GoodMatchePoints[i].queryIdx].pt);
            (*imagePoints2).push_back(keyPts2[GoodMatchePoints[i].trainIdx].pt);
        }
        if ((*imagePoints1).size() != (*imagePoints2).size()
            && (*imagePoints1).size() < GOODMATCHNUMBER
            && (*imagePoints2).size() < GOODMATCHNUMBER) {
            (*imagePoints1).clear();
            delete imagePoints1;
            (*imagePoints2).clear();
            delete imagePoints2;
            return 0;
        }
        vector<uchar> mask;

        Mat homo = (Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        mask.clear();
        try {

            // 仿射变换
            //            homo = estimateAffine2D(*imagePoints2, *imagePoints1, mask);
            //            if (!homo.empty() && homo.rows == 2 && homo.cols == 3) {
            //                Rect rect(0, 0, homo.cols, homo.rows);
            //                homo.copyTo(result.homo(rect));
            //            }

            // 透视变换
            homo = findHomography(*imagePoints2, *imagePoints1, RANSAC, 5.0, mask);
            if (!homo.empty() && homo.rows == 3 && homo.cols == 3) {
                Rect rect(0, 0, homo.cols, homo.rows);
                homo.copyTo(result.homo(rect));
            }


            else { mask.clear(); }
        }
        catch (Exception) { mask.clear(); }
        catch (exception) { mask.clear(); }
        homo.release();

        for (size_t i = 0; i < mask.size(); i++) {
            result.mask.push_back(mask[i]);
        }
        (*imagePoints1).clear();
        delete imagePoints1;
        (*imagePoints2).clear();
        delete imagePoints2;
        return 1;
    }
    catch (...) { return 0; }
}


int get_boxdata(boxdata &result, vector<Point2f>& points)
{
    if (points.size() < 1) {
        return 0;
    }
    result.xmin = points[0].x;
    result.ymin = points[0].y;
    result.xmax = points[0].x;
    result.ymax = points[0].y;
    for (size_t i = 0; i < points.size(); i++) {
        result.xmin = MIN(result.xmin, points[i].x);
        result.ymin = MIN(result.ymin, points[i].y);
        result.xmax = MAX(result.xmax, points[i].x);
        result.ymax = MAX(result.ymax, points[i].y);
    }
    return 1;
}
