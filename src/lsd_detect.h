//
// Created by qzj on 2020/8/13.
//

#ifndef RELOCAL_LSD_DETECT_H
#define RELOCAL_LSD_DETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <eigen3/Eigen/Core>
#include <vector>
#include <string>

void LoadImages(std::string &strPath, std::vector<std::string> &vstrImageLeft);
void ExtractLineSegment(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylines, cv::Mat &ldesc, int scale = 1.2, int numOctaves = 1);
void RemoveValue(std::vector<uint32_t> vec, uint32_t value);

static inline bool scoreTopMax(const std::pair<uint32_t, double> &a, const std::pair<uint32_t, double> &b)
{
    return a.second > b.second;
}

#endif //RELOCAL_LSD_DETECT_H
