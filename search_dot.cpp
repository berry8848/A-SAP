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
    //�t�@�C����������
    string output_csv_file_path = "Output/result.csv";
    // ��������csv�t�@�C�����J��(std::ofstream�̃R���X�g���N�^�ŊJ��)
    ofstream ofs_csv_file(output_csv_file_path);

    Mat img_src;
    VideoCapture capture(0);//�J�����I�[�v��
    if (!capture.isOpened()) {
        cout << "error" << endl;
        return -1;
    }
    //�R�[�i�[���o
    // �Q�l�Fhttp://opencv.jp/opencv2-x-samples/corner_detection/
    //�P�������ʐ^���B�� �����݁C�摜�̓ǂݍ��݂ɕύX
    capture >> img_src; //�J�����f���̓ǂݍ���
    Mat result_img = img_src.clone(); //�o�͉摜�p 

    //// �摜�t�@�C���̃p�X
    //string filename = "image.jpg";
    //// �摜��ǂݍ���
    //img_src = imread(filename);
    //if (img_src.empty()) {
    //    cout << "Failed to load the image: " << filename << endl;
    //    return -1;
    //}
    //Mat result_img = img_src.clone(); //�o�͉摜�p 

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
    cv::convertScaleAbs(laplacian_img_raw, laplacian_img_abs, alpha, beta);

    //�R�[�i�[�̌��o
    vector<Point2f> corners;
    goodFeaturesToTrack(laplacian_img_abs, corners, 80, 0.01, 30, Mat(), 3, true);

    //���W�ϊ�((u, v)��(u, height - v))
    float height = laplacian_img_abs.rows;
    cout << "height: " << height << endl;
    vector<Point2f> trans_corners(corners.size());
    for (int i = 0; i < corners.size(); i++) {
        Point2f origin_pnt = corners[i];
        double u = origin_pnt.x;
        double v = origin_pnt.y;
        double trans_u = u;
        double trans_v = height - v;
        Point2f trans_pnt;
        trans_pnt.x = trans_u;
        trans_pnt.y = trans_v;
        trans_corners[i] = trans_pnt;
    }

    //// y���W�����������Ƀ\�[�g
    //sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    //    return a.y < b.y;
    //    });
    //// x���W�����������Ƀ\�[�g
    //sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    //    return (a.y == b.y) ? (a.x < b.x) : false;
    //    });
    //cout << corners << endl;;

    // y���W�����������Ƀ\�[�g
    sort(trans_corners.begin(), trans_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
        });
    // x���W�����������Ƀ\�[�g
    sort(trans_corners.begin(), trans_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y == b.y) ? (a.x < b.x) : false;
        });
    cout << "trans_corners: " << trans_corners << endl;


    //(0, 0)�ɋ߂��_����E���т�sort
    vector<Point2f> sort_trans_corners; //sort�����_��p
    vector<Point2f> range_corners; //range���Ɋ܂܂��_��p
    double range = sqrt(pow(trans_corners[0].x - trans_corners[1].x, 2) + pow(trans_corners[0].y - trans_corners[1].y, 2)); //x���W�̏���������sort����ۂ̓_��͈̔͂̌���D���̒l�͈ꎞ�I
    Point2f reference_point(0.0f, 0.0f); //��_�i�[�p
    cout << "int(sqrt(trans_corners.size())): " << int(sqrt(trans_corners.size())) << endl;

    for (int i = 0; i < int(sqrt(trans_corners.size())); i++) {
        reference_point = getClosestPoint(trans_corners, reference_point); //��_�Ɉ�ԋ߂��_��V���Ȋ�_�Ƃ���
        range_corners = sortWithinRange(trans_corners, reference_point, range); //reference_point����Ƃ��Crange�Ɋ܂܂��_���trans_corners���璊�o���Cx���W�̏���������sort�������ʂ�Ԃ�
        //sort_trans_corners�Ɋi�[
        for (const Point2f& corner : range_corners) {
            sort_trans_corners.push_back(corner);
        }
        reference_point = Point2f(reference_point.x, reference_point.y + range); //��_��y���W��range�������������W�Ɉ�ԋ߂��_��V���Ȋ�_�Ƃ���D
    }
    cout << "sort_trans_corners: " << sort_trans_corners << endl;

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

    // �摜��ۑ�����
    std::string filename1 = "gray_img.jpg";
    std::string filename2 = "gaussian_img.jpg";
    std::string filename3 = "laplacian_img_abs.jpg";
    std::string filename4 = "result_img.jpg";
    bool success1 = cv::imwrite(filename1, gray_img);
    bool success2 = cv::imwrite(filename2, gaussian_img);
    bool success3 = cv::imwrite(filename3, laplacian_img_abs);
    bool success4 = cv::imwrite(filename4, result_img);


    waitKey(0);

    capture.release();
    return 0;
}