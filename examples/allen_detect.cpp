#include "../include/darknet.h"
#include "../src/image.h"

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <string>
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

/////////////////////////////////////////////////////////////////////////////////////////////////////

void allen_detect_image(const std::string filename)
{
    float thresh = 0.5;
    float nms = 0.45;

    list *options = read_data_cfg("cfg/coco.data");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);   // namelist = data/coco.names

    image **alphabet = load_alphabet();
    network* net = load_network("cfg/yolov3.cfg", "yolov3.weights", 0);
    set_batch_network(net, 1);

    image img = load_image_cv(const_cast<char*>(filename.c_str()), 0);

    // scale the image by yolov3.cfg
    image sized = letterbox_image(img, net->w, net->h);

    network_predict(net, sized.data);
    
    int nboxes = 0;
    detection* dets = get_network_boxes(net, img.w, img.h, thresh, 0.5, 0, 1, &nboxes);
    layer l = net->layers[net->n-1];
    if(nms)  do_nms_sort(dets, nboxes, l.classes, nms);

    draw_detections(img, dets, nboxes, thresh, names, alphabet, l.classes);
    free_detections(dets, nboxes);

    save_image(img, "predictions");
#ifdef OPENCV
    make_window("predictions", 512, 512, 0);
    show_image(img, "predictions", 0);
#endif

    // free_list_contents(options);
    free_list(options);
    free(names);
    free_image(img);
    free_image(sized);
    free_network(net);
}

void allen_detect_video(const std::string filename)
{
    float thresh = 0.5;
    float nms = 0.45;

    network* net = load_network("cfg/yolov3.cfg", "yolov3.weights", 0);
    set_batch_network(net, 1);

    std::vector<std::string> classNamesVec;
    std::ifstream classNamesFile("data/coco.names");  //标签文件coco有80类
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    srand(2222222);

    cv::Mat frame;
    cv::VideoCapture capture(filename);
    while(capture.read(frame))
    {
        image srcImage = mat_to_image(frame);
        std::cout << "mat_to_image done" << std::endl;

        // scale the image by yolov3.cfg
        image sized = letterbox_image(srcImage, net->w, net->h);
        std::cout << "letterbox_image done" << std::endl;
 
        network_predict(net, sized.data);
        std::cout << "predict done" << std::endl;
        
        int nboxes = 0;
        detection* dets = get_network_boxes(net, srcImage.w, srcImage.h, thresh, 0.5, 0, 1, &nboxes);
        std::cout << "get_network_boxes done" << std::endl;
        
        layer l = net->layers[net->n-1];
        if(nms)  do_nms_sort(dets, nboxes, l.classes, nms);
        std::cout << "do_nms_sort done" << std::endl;

        std::vector<cv::Rect>boxes;
        std::vector<std::string>classNames;
        std::vector<int> classname_idx_vec;
        std::multimap<float, int, std::greater<int>> mark_map;
        for (int i = 0; i < nboxes; i++)
        {
            mark_map.clear();

            bool flag = false;
            // int className;
            for(int j=0; j<l.classes; j++)
            {
                if (dets[i].prob[j] > thresh)
                {
                    mark_map.emplace(dets[i].prob[j], j);
                    if (!flag)
                    {
                        classname_idx_vec.push_back(j);
                        flag = true;
                    }
                }
                // if(dets[i].prob[j] > thresh)
                // {
                //     if(!flag)
                //     {
                //         flag = 1;
                //         className = j;
                //     } 
                // }
            }

            if (mark_map.empty())
            {
                std::cout << "box: " << i << " empty!!!" << std::endl;
            }
            else
            {
                std::cout << "box: " << i;
                for (auto v : mark_map)
                    std::cout << " val:" << v.first << " idx:" << v.second;
                std::cout << std::endl;
            }
            
            if(!mark_map.empty())
            {
                // yolov3的box输出x和y是归一化的矩形框的中心点，w和h是归一化矩形框的宽度和高度
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

                std::string classname;
                for (auto v : mark_map)
                {
                    classname += classNamesVec[v.second];
                    classname += ";";
                }
                classNames.push_back(classname);
            }
        }
        free_detections(dets, nboxes);
 
        for(int i=0; i<boxes.size(); i++)
        {
            int offset = classname_idx_vec[i] * 123457 % l.classes;
            float red = 255 * get_color(2, offset, l.classes);
            float green = 255 * get_color(1, offset, l.classes);
            float blue = 255 * get_color(0, offset, l.classes);
 
            cv::rectangle(frame, boxes[i], cv::Scalar(blue, green, red), 2);
 
            cv::String label = classNames[i];
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(frame, label, cv::Point(boxes[i].x, boxes[i].y + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(red, blue, green),2);
        }

        std::cout << "show img" << std::endl;
        imshow("video",frame);
        int c = cv::waitKey(10);
  
        free_image(srcImage);
        free_image(sized);
    }
    free_network(net);
    capture.release();
}

void allen_detector(int img_or_video, const std::string filename)
{
    if (img_or_video == 0) allen_detect_image(filename);
    else if (img_or_video == 1) allen_detect_video(filename);
    else std::cout << "invalid arg" << std::endl;
}
//////////////////////////////////////////////////////////////////////////////
// ./allen_detect 0 filepath
// ./allen_detect 1 filepath
int main(int argc, char**argv)
{
#ifdef OPENCV
    std::cout << "use opencv..." << std::endl;
#else
    std::cout << "not use opencv..." << std::endl;
#endif
    std::cout << "hello darknet" << std::endl;

    int img_video = 0; // 0 - image, 1 - video
    if (argc >= 3)  
    {
        img_video = std::atoi(argv[1]);
        allen_detector(img_video, argv[2]);
    }
    else std::cout << "invalid number of parameters" << std::endl;
    return 0;
}
