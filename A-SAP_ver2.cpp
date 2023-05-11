#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //�摜���o�́�GUI����p
#include <string> //csv�t�@�C���������ݗp
#include <fstream> //csv�t�@�C���������ݗp
#include <algorithm> //sort�֐��p
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";



int main()
{
    Mat img_src;
    VideoCapture capture(0);//�J�����I�[�v��
    if (!capture.isOpened()) {
        cout << "error" << endl;
        return -1;
    }

    //�t�@�C����������
    string output_csv_file_path = "Output/result.csv";
    // ��������csv�t�@�C�����J��(std::ofstream�̃R���X�g���N�^�ŊJ��)
    ofstream ofs_csv_file(output_csv_file_path);

    //�R�[�i�[���o
    // �Q�l�Fhttp://opencv.jp/opencv2-x-samples/corner_detection/
    //�P�������ʐ^���B��
    capture >> img_src; //�J�����f���̓ǂݍ���
    Mat result_img = img_src.clone(); //�o�͉摜�p 
    
    //�O���[�X�P�[���ϊ�
    Mat gray_img;
    cvtColor(img_src, gray_img, COLOR_BGR2GRAY); 

    //�K�E�V�A���t�B���^�̓K�p
    Mat gaussian_img;
    GaussianBlur(gray_img, gaussian_img, Size(9, 9), 0, 0);
    
    // ���v���V�A���t�B���^�̓K�p
    Mat laplacian_img_raw;
    Laplacian(gaussian_img, laplacian_img_raw, CV_16S, 5);
    //���v���V�A���̌��ʂ�ABS
    Mat laplacian_img_abs;
    cv::convertScaleAbs(laplacian_img_raw, laplacian_img_abs);

    //�R�[�i�[�̌��o
    vector<Point2f> corners;
    goodFeaturesToTrack(laplacian_img_abs, corners, 80, 0.01, 30, Mat(), 3, true);

    //// y���W�����������Ƀ\�[�g
    //sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    //    return a.y < b.y;
    //    });
    //// x���W�����������Ƀ\�[�g
    //sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    //    return (a.y == b.y) ? (a.x < b.x) : false;
    //    });
    //cout << corners << "\n";

    //// �o�͉摜�̍쐬
    vector<Point2f>::iterator it_corner = corners.begin();
    it_corner = corners.begin();
    for (; it_corner != corners.end(); ++it_corner) {
        circle(result_img, Point(it_corner->x, it_corner->y), 1, Scalar(0, 255, 0), -1); //�֐��̐��� http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
        ofs_csv_file << it_corner->x << ", " << it_corner->y << endl;
        circle(result_img, Point(it_corner->x, it_corner->y), 8, Scalar(0, 255, 0));
    }
    

    // ���ʕ\��
    //�E�C���h�E����
    namedWindow(win_src, WINDOW_AUTOSIZE);
    namedWindow("gray_img", WINDOW_AUTOSIZE);
    namedWindow("gaussian_img", WINDOW_AUTOSIZE);
    namedWindow("laplacian_img_raw", WINDOW_AUTOSIZE);
    namedWindow("laplacian_img_abs", WINDOW_AUTOSIZE);
    namedWindow("result_img", WINDOW_AUTOSIZE);

    imshow(win_src, img_src); //���͉摜��\��
    imshow("gray_img", gray_img); //�O���[�X�P�[���摜��\��
    imshow("gaussian_img", gaussian_img); //���R���摜��\��
    imshow("laplacian_img_raw", laplacian_img_raw); //���v���V�A���t�B���^�̌��ʁi0�`255�͈̔͂Ɏ��܂�Ȃ�)��\��
    imshow("laplacian_img_abs", laplacian_img_abs); //���v���V�A���t�B���^�̌��ʁi0�`255�͈̔͂Ɏ��܂�)��\��
    imshow("result_img", result_img); //��_���o�摜��\��

    waitKey(0);

    capture.release();
    return 0;
}