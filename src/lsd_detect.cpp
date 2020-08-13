//
// Created by qzj on 2020/8/13.
//

#include "lsd_detect.h"
#include "util.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void RemoveValue(vector<uint32_t> vec, uint32_t value) {
    vector<uint32_t>::iterator it;
    for (it = vec.begin(); it != vec.end();) {
        if (*it == value)
            it = vec.erase(it);
        else
            it++;
    }
}

struct sort_lines_by_response {
    inline bool operator()(const line_descriptor::KeyLine &a, const line_descriptor::KeyLine &b) {
        return (a.response > b.response);
    }
};

void ExtractLineSegment(const Mat &img, std::vector<line_descriptor::KeyLine> &keylines, Mat &ldesc, int scale,
                        int numOctaves) {
    uint16_t width = img.cols;
    uint16_t height = img.rows;

    // detect line features
    Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
    lsd->detect(img, keylines, scale, numOctaves);

    //去掉图像边缘的线，可以在lsd提取文件里修改
    for (std::vector<line_descriptor::KeyLine>::iterator it = keylines.begin(); it != keylines.end();) {
        Eigen::Vector3d sp_l;
        sp_l << it->startPointX, it->startPointY, 1.0;
        Eigen::Vector3d ep_l;
        ep_l << it->endPointX, it->endPointY, 1.0;

        if ((sp_l[0] > (width - 5) || sp_l[0] < 5 || sp_l[1] > (height - 5) || sp_l[1] < 5) &&
            (ep_l[0] > (width - 5) || ep_l[0] < 5 || ep_l[1] > (height - 5) || ep_l[1] < 5)) {
            it = keylines.erase(it);
        } else {
            it++;
        }
    }
    unsigned int lsdNFeatures = 50;
    // filter lines
    if (keylines.size() > lsdNFeatures) {
        sort(keylines.begin(), keylines.end(), sort_lines_by_response());
        keylines.resize(lsdNFeatures);
        for (unsigned int i = 0; i < lsdNFeatures; i++) {
            keylines[i].class_id = i;
        }
    }

    lbd->compute(img, keylines, ldesc);     //计算特征线段的描述子
}


void LoadImages(std::string &strPath, vector<string> &vstrImageLeft) {
    cerr << "Start LoadImages." << endl;
    vstrImageLeft.reserve(10000);

    unsigned int iSize = strPath.size();
    if (strPath.at(iSize - 1) != '/')
        strPath.push_back('/');

    //load image 法一：
    getSortedImages(strPath, vstrImageLeft);

    cout << "Finish LoadImages: " << vstrImageLeft.size() << endl;
}

