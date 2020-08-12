#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "util.h"

using namespace cv;
using namespace std;
void LoadImages(string &strPath, vector<string> &vstrImageLeft);
/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/
#define TRAIN_INDEXES_NUM 2

int main( int argc, char** argv ) {
    // read the image 
    cout<<"reading images... "<<endl;
    vector<Mat> images;
    string path = "/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/";
    int trainIdx[TRAIN_INDEXES_NUM] = { 3, 6 };

    //载入所有图片的路径
    vector<string> vImgPaths;
    for ( int i=0; i<TRAIN_INDEXES_NUM; i++ )
    {
        string seq = to_string(trainIdx[i]);
        while (seq.size()<2)
            seq="0"+seq;
        string img_path = path + seq + "/left/";
        vector<string> vstrImageLeft;
        LoadImages(img_path, vstrImageLeft);
        vImgPaths.insert(vImgPaths.end(),vstrImageLeft.begin(),vstrImageLeft.end());
    }
    //载入图片
    for(auto path:vImgPaths)
        images.push_back(imread(path));
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
    // create vocabulary 
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    //vocabulary info: Vocabulary: k = 10, L = 5, Weighting = tf-idf, Scoring = L1-norm, Number of words = 99725
    //cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "self_robot_vocab.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}


void LoadImages(string &strPath, vector<string> &vstrImageLeft)
{
    cerr << "Start LoadImages." << endl;
    vstrImageLeft.reserve(10000);

    unsigned int iSize = strPath.size();
    if(strPath.at(iSize-1)!='/')
        strPath.push_back('/');

    //load image 法一：
    getSortedImages(strPath, vstrImageLeft);

    assert(vTimeStamps.size()==vstrImageLeft.size() && vTimeStamps.size()==vstrImageRight.size());

    cout<<"Finish LoadImages: "<<vstrImageLeft.size()<<endl;
}