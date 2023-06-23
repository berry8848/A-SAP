//�ړI�F�_�T�����C�Ђ��ݕ␳�֐��쐬�̏��� ��l�����̈敪�����m�C�Y���������x�����Ƃɒ��S���W�����߂遨���W�ϊ����\�[�g

#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //�摜���o�́�GUI����p
#include <opencv2/core/core.hpp> //���ʎqPoint�p
#include <string> //csv�t�@�C���������ݗp
#include <fstream> //csv�t�@�C���������ݗp
#include <algorithm> //sort�֐��p
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";
#define THRESHOLD 128
#define NUMBER_OF_DOTS 1


//�w�i�̈���������̈�̂����C�ł��傫���̈�̃��x���ԍ��𒊏o
int circleLabels(Mat& matrix) {
    int label_num = 0; //n�Ԗڂɑ傫���v�f�̃��x���ԍ��i�[�p�D
    int max = 0; //�ő�ʐϊi�[�p

    //�w�i�̈�i���x��0�j���̂�����i�Ԗڂɑ傫���v�f��T��
    for (int j = 1; j < matrix.rows; j++) {
        if (max < matrix.at<int>(j, 4))
        {
            max = matrix.at<int>(j, 4);
            label_num = j;
        }
    }    
    return label_num;
}

//�����̍��W�l����͂Ƃ��C�����̒��S���W���o�͂Ƃ���
Point2f calculateCenter(const vector<Point>& points) {
    float centerX = 0.0;
    float centerY = 0.0;

    // ���W�̍��v���v�Z
    for (const auto& point : points) {
        centerX += point.x;
        centerY += point.y;
    }

    // ���W�̐��Ŋ����Ē��S���W�����߂�
    centerX /= points.size();
    centerY /= points.size();

    Point2f center;
    center.x = centerX;
    center.y = centerY;

    return center;
}


//���o�������x���ԍ����ƂɁC���̃��x���ԍ��𖞂������W�𒊏o���C���o�������W�l�̒��S���W��Ԃ�
Point2f circleCenter(int circle_labels, Mat& labels) {
    Point2f center; //���S���W�i�[�p
    vector<Point> target_label_points; //���x���ԍ��𖞂������W�i�[�p

    //�w�肵�����x���ԍ��̍��W�l��T��
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) == circle_labels) {
                target_label_points.push_back({ x, y });
                //cout << "(x, y) = " << x << " , " << y << "  " << target_label_points.back() << endl;
            }
        }
    }
    center = calculateCenter(target_label_points); //���S���W���v�Z
    return center;
}


int main()
{
    //�t�@�C����������
    string output_csv_file_path = "Output/result.csv";
    // ��������csv�t�@�C�����J��(std::ofstream�̃R���X�g���N�^�ŊJ��)
    ofstream ofs_csv_file(output_csv_file_path);

    Mat img_src;

    ////�J�����g�p��
    //VideoCapture capture(0);//�J�����I�[�v��
    //if (!capture.isOpened()) {
    //    cout << "error" << endl;
    //    return -1;
    //}
    //// �B�e�摜�T�C�Y�̐ݒ�
    //bool bres = capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    //if (bres != true) {
    //    return -1;
    //}
    //bres = capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    //if (bres != true) {
    //    return -1;
    //}
    //// �B�e�摜�擾�������̍H�v
    //bres = capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    //if (bres != true) {
    //    return -1;
    //}
    ////�R�[�i�[���o
    //// �Q�l�Fhttp://opencv.jp/opencv2-x-samples/corner_detection/
    ////�P�������ʐ^���B�� �����݁C�摜�̓ǂݍ��݂ɕύX
    //capture >> img_src; //�J�����f���̓ǂݍ���
    //Mat result_img = img_src.clone(); //�o�͉摜�p 

    // �摜�t�@�C���g�p��
    string filename = "red_circle.jpg";
    // �摜��ǂݍ���
    img_src = imread(filename);
    if (img_src.empty()) {
        cout << "Failed to load the image: " << filename << endl;
        return -1;
    }
    Mat result_img = img_src.clone(); //�o�͉摜�p 

     // �摜��HSV�F��Ԃɕϊ�
    Mat hsvImage;
    cvtColor(img_src, hsvImage, cv::COLOR_BGR2HSV);

    // �ԐF�͈̔͂��`
    Scalar lowerRed(0, 100, 100);
    Scalar upperRed(10, 255, 255);
    Scalar lowerRed2(170, 100, 100);
    Scalar upperRed2(180, 255, 255);

    // �ԐF�̈�𒊏o
    Mat redMask1, redMask2, redMask, redImage;
    inRange(hsvImage, lowerRed, upperRed, redMask1);
    inRange(hsvImage, lowerRed2, upperRed2, redMask2);
    add(redMask1, redMask2, redMask);

    //�̈敪��
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(redMask, labels, stats, centroids);
    cout << "labels.at<int>(200, 100): " << labels.at<int>(200, 100) << endl;
    cout << "labels.at<int>(0, 0): " << labels.at<int>(0, 0) << endl;
    cout << "stats: " << stats << endl;

    int circle_labels = circleLabels(stats); //�w�i�̈���������̈�̂����C�ł��傫���̈�̃��x���ԍ��𒊏o

    cout << "circle_labels : " << circle_labels << endl;

    Point2f center;
    center = circleCenter(circle_labels, labels); //���o�������x���ԍ����ƂɁC���̃��x���ԍ��𖞂������W�𒊏o���C���o�������W�l�̒��S���W��Ԃ��D

    ////���W�ϊ�((u, v)��(u, height - v))
    //float height = img_src.rows;
    //cout << "height: " << height << endl;
    //vector<Point2f> trans_centers(centers.size());
    //for (int i = 0; i < centers.size(); i++) {
    //    Point2f origin_pnt = centers[i];
    //    double u = origin_pnt.x;
    //    double v = origin_pnt.y;
    //    double trans_u = u;
    //    double trans_v = height - v;
    //    Point2f trans_pnt;
    //    trans_pnt.x = trans_u;
    //    trans_pnt.y = trans_v;
    //    trans_centers[i] = trans_pnt;
    //}
    //cout << "trans_centers: " << trans_centers << endl;

    // �o�͉摜�̍쐬
    circle(result_img, Point(center.x, center.y), 1, Scalar(0, 255, 0), -1); //�֐��̐��� http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
    circle(result_img, Point(center.x, center.y), 8, Scalar(0, 255, 0));
    ofs_csv_file << center.x << ", " << center.y << endl; //csv�t�@�C���o��

    // ���ʕ\��
    imshow(win_src, img_src); //���͉摜��\��
    imshow("Red Image", redMask); // �ԐF�̈�̕\��
    imshow("result_img", result_img); //��_���o�摜��\��

    //// �摜��ۑ�����
    //std::string filename1 = "gray_img.jpg";
    //std::string filename2 = "gaussian_img.jpg";
    //std::string filename3 = "laplacian_img_abs.jpg";
    //std::string filename4 = "result_img.jpg";
    //bool success1 = cv::imwrite(filename1, gray_img);
    //bool success2 = cv::imwrite(filename2, gaussian_img);
    //bool success3 = cv::imwrite(filename3, laplacian_img_abs);
    //bool success4 = cv::imwrite(filename4, result_img);

    waitKey(0);

    //capture.release();
    return 0;
}