#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <cmath>
#include <vector>
#include <map>
#include <set>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int search_coord(vector<Point2f>,vector<Point2f>, vector<pair<uint,uint>>&);
bool is_near(double, double ,int koef = 20);
double metric(Point2f,Point2f);

int main()
{

    Mat test1 = imread("test2.jpg");
    namedWindow("test1_original");
    imshow("test1_original",test1);
    waitKey();
    destroyWindow("test1_oroginal");

    Mat gaus;
    GaussianBlur(test1, gaus,{5,5},5);

    Mat draw, canny_output;
    Canny(gaus,draw,50,130);
    draw.convertTo(canny_output,CV_8U);


    Mat kernel = getStructuringElement(MORPH_RECT,Size(8,8));
    Mat closed;
    morphologyEx(canny_output, closed, MORPH_CLOSE, kernel);


    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(closed,contours, hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE );

    vector<vector<Point2f>> all_corners;
    vector<Mat> figures;
    for(size_t i = 0; i< contours.size(); i++)
    {
        Mat picha;
        picha.create(test1.rows,test1.cols,test1.type());
        drawContours(picha,contours,(int)i,Scalar(255,255,255));

        vector<Point2f> corners;
        double qualityLevel = 0.01;
        double minDistance = 150;
        int blockSize = 10, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        Mat gray;
        cvtColor(picha, gray, COLOR_BGR2GRAY);
        goodFeaturesToTrack( gray,
                             corners,
                             20,
                             qualityLevel,
                             minDistance,
                             Mat(),
                             blockSize,
                             gradientSize,
                             useHarrisDetector,
                             k );


        figures.push_back(picha);

        map<uint,Point2f> obx;
        for(Point2f t: corners)
        {
            double min = 100000;
            uint i_min = 0;
            for(uint j = 0; j < contours[i].size(); ++j)
            {
                if(abs(metric(contours[i][j],t)) < min)
                {
                    min = metric(contours[i][j],t);
                    i_min = j;
                }
            }
            obx.emplace(make_pair(i_min,t));
        }

        auto it = obx.cbegin();
        for(uint j = 0; j < corners.size(); ++j)
        {
           corners[j] = it->second;
           ++it;
        }
        all_corners.push_back(corners);

    }

    set<uint> keys;
    keys.emplace(0);
    vector<pair<Point,vector<pair<uint,uint>>>> perehod;
    while(keys.size() != figures.size())
    {
        for(uint i = 0; i < all_corners.size(); ++i)
        {
            for(uint j = 1; j < all_corners.size(); ++j)
            {
                if(keys.find(j) != keys.cend())
                {
                    continue;
                }

                vector<pair<uint,uint>> t;
                if(search_coord(all_corners[i],all_corners[j],t)>= 4)
                {

                    perehod.push_back(make_pair(Point(i,j),t));
                    keys.emplace(j);

                }
            }

        }
        break;
    }

    for(auto dat: perehod)
    {
        int i = dat.first.x;
        int j = dat.first.y;
        auto t = dat.second;
        Point2f key11,key21,key31;
        key11 = all_corners[i][t[0].first];
        key21 = all_corners[i][t[1].first];
        key31 = all_corners[i][t[2].first];

        Point2f key12,key22,key32;
        key12 = all_corners[j][t[0].second];
        key22 = all_corners[j][t[1].second];
        key32 = all_corners[j][t[2].second];


        for(int k = 0; k < all_corners[j].size();++k)
        {
            Point now = all_corners[j][k];
            double l1,l2,l3;
            double max_metr = 8;

            l1 = metric(key12,now);
            l2 = metric(key22,now);
            l3 = metric(key32,now);

            for(int i1 = -1000; i1 < 1000 + test1.rows; ++i1)
            {
                for(int j1 = -1000; j1 < 1000 + test1.cols; ++j1)
                {
                    double m1 = metric(key11,Point(i1,j1));
                    if(!is_near(l1,m1,max_metr))
                        continue;

                    double m2 = metric(key21,Point(i1,j1));
                    if(!is_near(l2,m2,max_metr))
                        continue;

                    double m3 = metric(key31,Point(i1,j1));
                    if(!is_near(l3,m3,max_metr))
                        continue;

                    all_corners[j][k] = Point(i1,j1);
                    break;

                }
            }
        }
    }


    Mat ans;
    ans.create(2000,3000,figures[0].type());
    for(uint i = 0; i < all_corners.size(); ++i)
    {
        for(uint j = 0; j < all_corners[i].size(); j++)
        {
            if (j == all_corners[i].size()-1)
            {
                line(ans,all_corners[i][j],all_corners[i][0],Scalar(256,256,256));
                break;
            }

            line(ans,all_corners[i][j],all_corners[i][j+1],Scalar(256,256,256));
        }
    }
    namedWindow("Answer");
    imshow("Answer",ans);
    waitKey();
    destroyWindow("test1_oroginal");

    return 0;
}

int search_coord(vector<Point2f> fig1,vector<Point2f> fig2, vector<pair<uint,uint>>& t)
{
    int koef_max = 4;
    int vrem;

    for(int i = 0; i < fig1.size();++i)
    {
        for(int j = 0; j< fig2.size(); j++)
        {
            int ll = 1;
            int lr = 1;
            double m1, m2;
            vrem = 0;

            while(true)
            {
                if((i + ll) >= fig1.size())
                    m1 = metric(fig1[(i+ll) % fig1.size()],fig1[i]);
                else
                    m1 = metric(fig1[i+ll],fig1[i]);
                if((j - ll) < 0)
                    m2 = metric(fig2[fig2.size() + (j - ll)],fig2[j]);
                 else
                    m2 = metric(fig2[j - ll],fig2[j]);


                if(is_near(m1,m2))
                {
                    ++vrem;
                    ++ll;

                }
                else
                {
                    break;
                }
            }

            while(true)
            {
                if((i - lr) < 0)
                    m1 = metric(fig1[fig1.size() + (i - lr)],fig1[i]);
                 else
                    m1 = metric(fig1[i - lr],fig1[i]);

                if((j + lr) >= fig2.size())
                    m2 = metric(fig2[(j+lr) % fig2.size()],fig2[j]);
                 else
                    m2 = metric(fig2[j+lr],fig2[j]);

                if(is_near(m1,m2))
                {
                    ++vrem;
                    ++lr;
                }
                else
                {
                    break;
                }
            }
            if(vrem >= koef_max)
            {
                t.push_back({i,j});
                continue;
            }


        }
    }

    return t.size();

}

bool is_near(double a, double b,int koef)
{
    return abs(a - b) < koef;
}
double metric(Point2f a,Point2f b)
{
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y - b.y)*(a.y - b.y));
}
