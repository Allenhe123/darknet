#include "../include/darknet.h"

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include<opencv2/opencv.hpp>



//////////////////////////////////////////////////////////////////////////////////////////////////
void resizeInner(float *src, float* dst,int srcWidth,int srcHeight,int dstWidth,int dstHeight)
{
    float* part;
    size_t sizePa=dstWidth*srcHeight*3*sizeof(float);
    part=(float*)malloc(sizePa);
 
    float w_scale = (float)(srcWidth - 1) / (dstWidth - 1);
    float h_scale = (float)(srcHeight - 1) / (dstHeight - 1);
 
    for(int k = 0; k < 3; ++k){
        for(int r = 0; r < srcHeight; ++r){
            for(int c = 0; c < dstWidth; ++c){
                float val = 0;
                if(c == dstWidth-1 || srcWidth == 1){
                    val=src[k*srcWidth*srcHeight+r*srcWidth+srcWidth-1];
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val=(1 - dx) * src[k*srcWidth*srcHeight+r*srcWidth+ix] + dx * src[k*srcWidth*srcHeight+r*srcWidth+ix+1];
                }
                part[k*srcHeight*dstWidth + r*dstWidth + c]=val;
            }
        }
    }
 
    for(int k = 0; k < 3; ++k){
        for(int r = 0; r < dstHeight; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(int c = 0; c < dstWidth; ++c){
                float val = (1-dy) * part[k*dstWidth*srcHeight+iy*dstWidth+c];
                dst[k*dstWidth*dstHeight + r*dstWidth + c]=val;
            }
            if(r == dstHeight-1 || srcHeight == 1)
                continue;
            for(int c = 0; c < dstWidth; ++c){
                float val = dy * part[k*dstWidth*srcHeight+(iy+1)*dstWidth+c];
                dst[k*dstWidth*dstHeight + r*dstWidth + c]+=val;
            }
        }
    }
    free(part);
}

// 将OpenCV的图像由RGBRGBRGB...转化为yolo的RRRGGGBBB...格式, 并将像素值归一化到0~1.0
void imgConvert(const cv::Mat& img, float* dst)
{
    uchar *data = img.data;
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
 
    for(int k= 0; k < c; ++k)
    {
        for(int i = 0; i < h; ++i)
        {
            for(int j = 0; j < w; ++j)
            {
                dst[k * w * h + i * w + j] = data[(i * w + j) * c + k] / 255.;
            }
        }
    }
}
 
 // 将图像缩放到cfg指定的网络输入的大小
void imgResize(float *src, float* dst,int srcWidth,int srcHeight,int dstWidth,int dstHeight)
{
    int new_w = srcWidth;
    int new_h = srcHeight;
    if (((float)dstWidth/srcWidth) < ((float)dstHeight/srcHeight)) 
    {
        new_w = dstWidth;
        new_h = (srcHeight * dstWidth)/srcWidth;
    } 
    else 
    {
        new_h = dstHeight;
        new_w = (srcWidth * dstHeight)/srcHeight;
    }
 
    float* ImgReInner;
    size_t sizeInner=new_w*new_h*3*sizeof(float);
    ImgReInner=(float*)malloc(sizeInner);
    resizeInner(src,ImgReInner,srcWidth,srcHeight,new_w,new_h);
 
    for(int i=0;i<dstWidth*dstHeight*3;i++){
        dst[i]=0.5;
    }
 
    for(int k = 0; k < 3; ++k)
    {
        for(int y = 0; y < new_h; ++y)
        {
            for(int x = 0; x < new_w; ++x)
            {
                float val = ImgReInner[k*new_w*new_h+y*new_w+x];
                dst[k*dstHeight*dstWidth + ((dstHeight-new_h)/2+y)*dstWidth + (dstWidth-new_w)/2+x]=val;
            }
        }
    }
    free(ImgReInner);
}
 
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

void allen_detector()
{
    std::string cfgfile = "/home/allen/myproject/darknet/cfg/yolov3.cfg";        // 读取模型文件，请自行修改相应路径
    std::string weightfile = "/home/allen/myproject/darknet/yolov3.weights";
    float thresh = 0.5;                                                          //参数设置
    float nms = 0.35;
    int classes = 80;

    network* net = load_network((char*)cfgfile.c_str(), (char*)weightfile.c_str(), 0);
    set_batch_network(net, 1);
    cv::VideoCapture capture("/home/allen/myproject/dms/BlinkDetect/14-MaleNoGlasses.avi");
    cv::Mat frame;
    cv::Mat rgbImg;
 
    std::vector<std::string> classNamesVec;
    std::ifstream classNamesFile("/home/allen/myproject/darknet/data/coco.names");  //标签文件coco有80类
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
 
    while(capture.read(frame))
    {
        cv::cvtColor(frame, rgbImg, cv::COLOR_BGR2RGB);
        size_t srcSize = rgbImg.rows * rgbImg.cols * 3 * sizeof(float);
        float* srcImg = new float[rgbImg.rows * rgbImg.cols * 3];
        imgConvert(rgbImg, srcImg);                                         //将图像转为yolo形式
 
        float* resizeImg =  new float[net->w * net->h * 3 ];
        imgResize(srcImg, resizeImg, frame.cols, frame.rows, net->w, net->h);    //缩放图像
 
        network_predict(net, resizeImg);                                     //网络推理
        
        int nboxes = 0;
        detection* dets = get_network_boxes(net, rgbImg.cols, rgbImg.rows, thresh, 0.5, 0, 1, &nboxes);
        if(nms)
        {
            do_nms_sort(dets, nboxes, classes, nms);
        }
 
        std::vector<cv::Rect>boxes;
        std::vector<int>classNames;
        for (int i = 0; i < nboxes; i++)
        {
            bool flag = 0;
            int className;
            for(int j=0; j<classes; j++)
            {
                if(dets[i].prob[j] > thresh)
                {
                    if(!flag)
                    {
                        flag = 1;
                        className = j;
                    }
                }
            }
            if(flag)
            {
                int left = (dets[i].bbox.x - dets[i].bbox.w / 2.) * frame.cols;
                int right = (dets[i].bbox.x + dets[i].bbox.w / 2.) * frame.cols;
                int top = (dets[i].bbox.y - dets[i].bbox.h / 2.) * frame.rows;
                int bot = (dets[i].bbox.y + dets[i].bbox.h / 2.) * frame.rows;
 
                if (left < 0) left = 0;
                if (right > frame.cols - 1)  right = frame.cols - 1;
                if (top < 0) top = 0;
                if (bot > frame.rows - 1)  bot = frame.rows - 1;
 
                cv::Rect box(left, top, fabs(left - right), fabs(top - bot));
                boxes.push_back(box);
                classNames.push_back(className);
            }
        }
        free_detections(dets, nboxes);
 
        for(int i=0; i<boxes.size(); i++)
        {
            int offset = classNames[i] * 123457 % 80;
            float red = 255 * get_color(2, offset, 80);
            float green = 255 * get_color(1, offset, 80);
            float blue = 255 * get_color(0, offset, 80);
 
            cv::rectangle(frame, boxes[i], cv::Scalar(blue, green, red), 2);
 
            cv::String label = cv::String(classNamesVec[classNames[i]]);
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(frame, label, cv::Point(boxes[i].x, boxes[i].y + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(red, blue, green),2);
        }
        imshow("video",frame);
  
        delete []srcImg;
        delete []resizeImg;
    }
    free_network(net);
    capture.release();
}
//////////////////////////////////////////////////////////////////////////////

int main()
{
#ifdef OPENCV
    std::cout << "use opencv..." << std::endl;
#else
    std::cout << "not use opencv..." << std::endl;
#endif
    std::cout << "hello darknet" << std::endl;

    allen_detector();

    return 0;
}
