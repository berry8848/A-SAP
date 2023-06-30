#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //�摜���o�́�GUI����p
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";


int main()
{
    Mat img_src;
    //VideoCapture capture(0);//�J�����I�[�v��
    //if (!capture.isOpened()) {
    //    cout << "error" << endl;
    //    return -1;
    //}
    ////�R�[�i�[���o
    //// �Q�l�Fhttp://opencv.jp/opencv2-x-samples/corner_detection/
    ////�P�������ʐ^���B�� �����݁C�摜�̓ǂݍ��݂ɕύX
    //capture >> img_src; //�J�����f���̓ǂݍ���
    //Mat result_img = img_src.clone(); //�o�͉摜�p 

    // �摜�t�@�C���̃p�X
    string filename = "master.jpg";
    // �摜��ǂݍ���
    img_src = imread(filename);
    if (img_src.empty()) {
        cout << "Failed to load the image: " << filename << endl;
        return -1;
    }
    Mat result_img = img_src.clone(); //�o�͉摜�p 

    //�O���[�X�P�[���ϊ�
    Mat gray_img;
    cvtColor(img_src, gray_img, COLOR_BGR2GRAY);

    //�K�E�V�A���t�B���^�̓K�p
    Mat gaussian_img;
    GaussianBlur(gray_img, gaussian_img, Size(3, 3), 0, 0);

    //�O���[�X�P�[���摜�ɑ΂���2�l��
    Mat binary_img;
    threshold(gaussian_img, binary_img, 128, 255, THRESH_BINARY);

    //��f�l�𔽓]�D���]���Ȃ��ƃE�C���h�E�S�̂��֊s�F�������D
    Mat inverted_binary_img = 255 - binary_img;

    // �֊s���i�[����x�N�g��
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy; //Vec4i�͊K�w����\�����邽�߂Ɏg�p�����f�[�^�^�D�e�v�f�͊K�w���̈قȂ鑤�ʂ�\���D

    // �֊s���o
    findContours(inverted_binary_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1); //RETR_EXTERNAL:��ԊO���̔��֊s�����o��

    // contours�̗v�f��\��
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            cout << "(" << point.x << ", " << point.y << ") " << endl;
        }
    }

    // hierarchy��\��
    for (int i = 0; i < contours.size(); i++) {
        cout << "Contour " << i << ", Hierarchy: "
            << ", Next: " << hierarchy[i][0]
            << ", Previous: " << hierarchy[i][1]
            << ", Child: " << hierarchy[i][2] 
            << "Parent: " << hierarchy[i][3] 
            << endl;
    }

    // �֊s�̕`��
    drawContours(result_img, contours, -1, Scalar(0, 0, 255), 2);


    // ���ʕ\��
    imshow(win_src, img_src); //���͉摜��\��
    //imshow("gray_img", gray_img); //�O���[�X�P�[���摜��\��
    //imshow("gaussian_img", gaussian_img); //���R���摜��\��
    imshow("inverted_binary_img", inverted_binary_img); //��l�����]�摜��\��
    imshow("result_img", result_img); //�֊s���o���ʂ�\��


    waitKey(0);

    //capture.release();
    return 0;
}