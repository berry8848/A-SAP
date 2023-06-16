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
#define SEARCH_RANGE 2
#define NUMBER_OF_CORNERS 24
#define NUMBER_OF_MASTERS 6


//�͈͓���0�ȊO�̒l����郉�x����Ԃ�
int searchLabelInRange(const Point2f& corner, Mat& labels, vector<vector<Point2f>>& master_labels) {
    // �摜�̕��ƍ������擾
    int width = labels.cols;
    int height = labels.rows;

    for (int x = corner.x - SEARCH_RANGE; x <= corner.x + SEARCH_RANGE; ++x) {
        for (int y = corner.y - SEARCH_RANGE; y <= corner.y + SEARCH_RANGE; ++y) {
            if (x < 0 || width <= x || y < 0 || height <= y) continue;
            int label_num = labels.at<int>(y, x);
            if (label_num != 0) {
                master_labels[label_num - 1.0].push_back(corner);
                return label_num;
            }
        }
    }
    cout << "ERROR�F�R�[�i�[�̃��x���t�����ł��܂���ł����D" << endl;
    return 0;
}

//�e�s�̗�̃T�C�Y���m�F���C�w�肵����̃T�C�Y�ȊO�̎��G���[���b�Z�[�W��\��
void checkColumnSize(const vector<vector<Point2f>>& matrix) {
    int expectedSize = NUMBER_OF_CORNERS / NUMBER_OF_MASTERS;
    for (int targetRow = 0; targetRow < matrix.size(); ++targetRow) {
        int actualColumnSize = matrix[targetRow].size();  // ��̃T�C�Y�͍ŏ��̍s�̗v�f���Ƃ���
        if (actualColumnSize != expectedSize) {
            cout << "ERROR: Row " << targetRow << " �͊��҂��Ă���T�C�Y���ł͂���܂���D" << endl;
        }
    }
}

int main()
{
    //�t�@�C����������
    string output_csv_file_path = "Output/result.csv";
    // ��������csv�t�@�C�����J��(std::ofstream�̃R���X�g���N�^�ŊJ��)
    ofstream ofs_csv_file(output_csv_file_path);

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
    string filename = "image.jpg";
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

    // ���v���V�A���t�B���^�̓K�p
    Mat laplacian_img_raw;
    Laplacian(gaussian_img, laplacian_img_raw, CV_16S, 5);
    //convertScaleAbs��alpha,beta�̒l�����肷��
    double minValue, maxValue;
    double alpha, beta;
    minMaxLoc(laplacian_img_raw, &minValue, &maxValue); //�ő�ŏ��̉�f�l�̎擾
    alpha = 255 / (maxValue - minValue);
    beta = -alpha / minValue;

    //���v���V�A���̌��ʂ�ABS
    Mat laplacian_img_abs;
    convertScaleAbs(laplacian_img_raw, laplacian_img_abs, alpha, beta);

    //�R�[�i�[�̌��o
    vector<Point2f> corners;
    goodFeaturesToTrack(laplacian_img_abs, corners, 80, 0.01, 30, Mat(), 3, true);
    if (corners.size() != NUMBER_OF_CORNERS) {
        cout << "ERROR�F�w�肳�ꂽ�R�[�i�[�_�̐��ƌ��m�����R�[�i�[�_�̐����قȂ�܂��D" << endl;
    }
    cout << "corners: " << corners << endl;
    cout << "corners.size: " << corners.size() << endl;

    //�O���[�X�P�[���摜�ɑ΂���2�l��
    Mat binary_img;
    threshold(gray_img, binary_img, 128, 255, THRESH_BINARY);

    //��f�l�𔽓]
    Mat inverted_binary_img = 255 - binary_img;


    //�̈敪��
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(inverted_binary_img, labels, stats, centroids);
    cout << "labels.at<int>(200, 100): " << labels.at<int>(200, 100) << endl;
    cout << "labels.at<int>(0, 0): " << labels.at<int>(0, 0) << endl;
    cout << "stats: " << stats << endl;

    //�R�[�i�[�̃��x���t��
    int corner_labels[NUMBER_OF_CORNERS];
    vector<vector<Point2f>> master_labels(NUMBER_OF_MASTERS, vector<Point2f>(0));
    for (int i = 0; i < NUMBER_OF_CORNERS; i++) {
        corner_labels[i] = searchLabelInRange(corners.at(i), labels, master_labels);
        cout << "corner_labels: " << corner_labels[i] << endl;
    }

    //����Ƀ��x���t���ł��Ă��邩���m�F
    checkColumnSize(master_labels);

    // �l�̕\��
    for (int i = 0; i < master_labels.size(); ++i) {
        for (int j = 0; j < master_labels[i].size(); ++j) {
            cout << "master_labels[" << i << "][" << j << "]: (" << master_labels[i][j].x << ", " << master_labels[i][j].y << ")" << endl;
        }
    }

    // �o�͉摜�̍쐬
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

    //capture.release();
    return 0;
}