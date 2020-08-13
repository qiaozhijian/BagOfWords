#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "util.h"
#include "lsd_detect.h"

using namespace cv;
using namespace std;
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
    vector<Mat> pdescs;
    vector<Mat> ldescs;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat pdesc, ldesc;
        vector<line_descriptor::KeyLine> keylines;
        detector->detectAndCompute( image, Mat(), keypoints, pdesc );
        pdescs.push_back( pdesc );

        ExtractLineSegment(image, keylines, ldesc);
        ldescs.push_back( ldesc);
    }

    // create point vocabulary
    cout<<"creating vocabulary point... "<<endl;
    DBoW3::Vocabulary vocab_pt;
    vocab_pt.create( pdescs );
    //vocabulary info: Vocabulary: k = 10, L = 5, Weighting = tf-idf, Scoring = L1-norm, Number of words = 99725
    //cout<<"vocabulary info: "<<vocab_pt<<endl;
    vocab_pt.save( "robot_vocab_pt.yml.gz" );
    cout<<"done"<<endl;

    // create point vocabulary
    cout<<"creating vocabulary line... "<<endl;
    DBoW3::Vocabulary vocab_line;
    vocab_line.create( ldescs );
    //vocabulary info: Vocabulary: k = 10, L = 5, Weighting = tf-idf, Scoring = L1-norm, Number of words = 99725
    //cout<<"vocabulary info: "<<vocab_line<<endl;
    vocab_line.save( "robot_vocab_line.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}
