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

//reference_point�Ɉ�ԋ߂��_��Ԃ��֐�
Point2f getClosestPoint(const vector<Point2f>& points, Point2f& reference_point) {
    Point2f closestPoint; //��_�Ɉ�ԋ߂��_���i�[
    float minDistance = numeric_limits<float>::max();
    // �e���W�ƌ��_�Ƃ̋������v�Z���A�ŏ������̍��W���擾
    for (const Point2f& point : points) {
        float distance = sqrt(pow(point.x - reference_point.x, 2) + pow(point.y - reference_point.y, 2));
        if (distance < minDistance) {
            minDistance = distance;
            closestPoint = point;
        }
    }
    return closestPoint;
}

//reference_point����Ƃ��Arange�Ɋ܂܂��_���points���璊�o���Ax���W�̏���������sort�������ʂ�Ԃ�
vector<Point2f> sortWithinRange(const vector<Point2f>& points, Point2f& reference_point, double range) {
    vector<Point2f> range_points; //range���Ɋ܂܂��_��i�[�p
    // range�Ɋ܂܂��_��̒��o
    for (const Point2f& point : points) {
        if (reference_point.y - range / 2 < point.y && point.y < reference_point.y + range / 2) {
            range_points.push_back(point);
        }
    }
    // x���W�����������Ƀ\�[�g
    sort(range_points.begin(), range_points.end(), [](const Point2f& a, const Point2f& b) {
        return a.x < b.x;
        });

    return range_points;
}

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
    GaussianBlur(gray_img, gaussian_img, Size(3, 3), 0, 0);

    // ���v���V�A���t�B���^�̓K�p
    Mat laplacian_img_raw;
    Laplacian(gaussian_img, laplacian_img_raw, CV_16S, 5);
    //convertScaleAbs��alpha,beta�̒l�����肷��
    double minValue, maxValue;
    double alpha, beta;
    minMaxLoc(laplacian_img_raw, &minValue, &maxValue); //�ő�ŏ��̉�f�l�̎擾
    alpha = 255 / (maxValue - minValue);
    beta = -alpha / minValue;
    cout << "Minimum value: " << minValue << endl;
    cout << "Maximum value: " << maxValue << endl;
    cout << "alpha: " << alpha << endl;
    cout << "beta: " << beta << endl;

    //���v���V�A���̌��ʂ�ABS
    Mat laplacian_img_abs;
    convertScaleAbs(laplacian_img_raw, laplacian_img_abs, alpha, beta);

    //�R�[�i�[�̌��o
    vector<Point2f> corners;
    goodFeaturesToTrack(laplacian_img_abs, corners, 80, 0.01, 30, Mat(), 3, true);
    cout << "corners: " << corners << endl;
    cout << "corners: " << corners.at(0) << endl;

    //�O���[�X�P�[���摜�ɑ΂���2�l��
    Mat binary_img;
    threshold(gray_img, binary_img, 128, 255, THRESH_BINARY);

    //��f�l�𔽓]
    Mat inverted_binary_img = 255 - binary_img;


    //�Z�O�����e�[�V����
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(inverted_binary_img, labels, stats, centroids);
    cout << "labels: " << labels.at<int>(20, 10) << endl;
    cout << "stats: " << stats << endl;


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
    namedWindow("inverted_binary_img", WINDOW_AUTOSIZE);
    namedWindow("labels", WINDOW_AUTOSIZE);
    namedWindow("result_img", WINDOW_AUTOSIZE);

    imshow(win_src, img_src); //���͉摜��\��
    imshow("gray_img", gray_img); //�O���[�X�P�[���摜��\��
    imshow("gaussian_img", gaussian_img); //���R���摜��\��
    imshow("laplacian_img_raw", laplacian_img_raw); //���v���V�A���t�B���^�̌��ʁi0�`255�͈̔͂Ɏ��܂�Ȃ�)��\��
    imshow("laplacian_img_abs", laplacian_img_abs); //���v���V�A���t�B���^�̌��ʁi0�`255�͈̔͂Ɏ��܂�)��\��
    imshow("inverted_binary_img", inverted_binary_img); //��_���o�摜��\��
    //imshow("labels", labels); //��_���o�摜��\��
    imshow("result_img", result_img); //��_���o�摜��\��


    waitKey(0);

    capture.release();
    return 0;
}