#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //画像入出力＆GUI操作用
#include <string> //csvファイル書き込み用
#include <fstream> //csvファイル書き込み用
#include <algorithm> //sort関数用
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";
#define SEACH_VALUE 40
#define NUMBER_OF_CORNERS 24

//reference_pointに一番近い点を返す関数
Point2f getClosestPoint(const vector<Point2f>& points, Point2f& reference_point) {
    Point2f closestPoint; //基準点に一番近い点を格納
    float minDistance = numeric_limits<float>::max();
    // 各座標と原点との距離を計算し、最小距離の座標を取得
    for (const Point2f& point : points) {
        float distance = sqrt(pow(point.x - reference_point.x, 2) + pow(point.y - reference_point.y, 2));
        if (distance < minDistance) {
            minDistance = distance;
            closestPoint = point;
        }
    }
    return closestPoint;
}

//reference_pointを基準とし、rangeに含まれる点列をpointsから抽出し、x座標の小さい順にsortした結果を返す
vector<Point2f> sortWithinRange(const vector<Point2f>& points, Point2f& reference_point, double range) {
    vector<Point2f> range_points; //range内に含まれる点列格納用
    // rangeに含まれる点列の抽出
    for (const Point2f& point : points) {
        if (reference_point.y - range / 2 < point.y && point.y < reference_point.y + range / 2) {
            range_points.push_back(point);
        }
    }
    // x座標が小さい順にソート
    sort(range_points.begin(), range_points.end(), [](const Point2f& a, const Point2f& b) {
        return a.x < b.x;
        });

    return range_points;
}

//範囲内で0以外の値を取るラベルを返す
int searchLabelInRange(const Point2f& center, Mat& labels) {
    for (int x = center.x - SEACH_VALUE; x <= center.x + SEACH_VALUE; ++x) {
        for (int y = center.y - SEACH_VALUE; y <= center.y + SEACH_VALUE; ++y) {
            if (labels.at<int>(x, y) != 0) {
                return labels.at<int>(x, y);
            }
        }
    }
    cout << "ERROR：コーナーのラベル付けができませんでした．" << endl;
    return 0;
}


int main()
{
    Mat img_src;
    VideoCapture capture(0);//カメラオープン
    if (!capture.isOpened()) {
        cout << "error" << endl;
        return -1;
    }

    //ファイル書き込み
    string output_csv_file_path = "Output/result.csv";
    // 書き込むcsvファイルを開く(std::ofstreamのコンストラクタで開く)
    ofstream ofs_csv_file(output_csv_file_path);

    //コーナー検出
    // 参考：http://opencv.jp/opencv2-x-samples/corner_detection/
    //１枚だけ写真を撮る
    capture >> img_src; //カメラ映像の読み込み
    Mat result_img = img_src.clone(); //出力画像用 

    //グレースケール変換
    Mat gray_img;
    cvtColor(img_src, gray_img, COLOR_BGR2GRAY);

    //ガウシアンフィルタの適用
    Mat gaussian_img;
    GaussianBlur(gray_img, gaussian_img, Size(3, 3), 0, 0);

    // ラプラシアンフィルタの適用
    Mat laplacian_img_raw;
    Laplacian(gaussian_img, laplacian_img_raw, CV_16S, 5);
    //convertScaleAbsのalpha,betaの値を決定する
    double minValue, maxValue;
    double alpha, beta;
    minMaxLoc(laplacian_img_raw, &minValue, &maxValue); //最大最小の画素値の取得
    alpha = 255 / (maxValue - minValue);
    beta = -alpha / minValue;
    cout << "Minimum value: " << minValue << endl;
    cout << "Maximum value: " << maxValue << endl;
    cout << "alpha: " << alpha << endl;
    cout << "beta: " << beta << endl;

    //ラプラシアンの結果にABS
    Mat laplacian_img_abs;
    convertScaleAbs(laplacian_img_raw, laplacian_img_abs, alpha, beta);

    //コーナーの検出
    vector<Point2f> corners;
    goodFeaturesToTrack(laplacian_img_abs, corners, 80, 0.01, 30, Mat(), 3, true);
    if (corners.size() != NUMBER_OF_CORNERS) {
        cout << "ERROR：指定されたコーナー点の数と検知したコーナー点の数が異なります．" << endl;
    }
    cout << "corners: " << corners << endl;
    cout << "corners.size: " << corners.size() << endl;

    //グレースケール画像に対して2値化
    Mat binary_img;
    threshold(gray_img, binary_img, 128, 255, THRESH_BINARY);

    //画素値を反転
    Mat inverted_binary_img = 255 - binary_img;


    //領域分割
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(inverted_binary_img, labels, stats, centroids);
    cout << "labels: " << labels.at<int>(20, 10) << endl;
    cout << "stats: " << stats << endl;

    //コーナーのラベル付け
    int corner_labels[NUMBER_OF_CORNERS];
    for (int i = 0; i < NUMBER_OF_CORNERS; i++) {
        //cout << "corners: " << corners.at(i) << endl;
        corner_labels[i] = searchLabelInRange(corners.at(i), labels);
        cout << "corner_labels: " << corner_labels[i] << endl;
    }

    //// 出力画像の作成
    vector<Point2f>::iterator it_corner = corners.begin();
    it_corner = corners.begin();
    for (; it_corner != corners.end(); ++it_corner) {
        circle(result_img, Point(it_corner->x, it_corner->y), 1, Scalar(0, 255, 0), -1); //関数の説明 http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
        ofs_csv_file << it_corner->x << ", " << it_corner->y << endl;
        circle(result_img, Point(it_corner->x, it_corner->y), 8, Scalar(0, 255, 0));
    }


    // 結果表示
    //ウインドウ生成
    namedWindow(win_src, WINDOW_AUTOSIZE);
    namedWindow("gray_img", WINDOW_AUTOSIZE);
    namedWindow("gaussian_img", WINDOW_AUTOSIZE);
    namedWindow("laplacian_img_raw", WINDOW_AUTOSIZE);
    namedWindow("laplacian_img_abs", WINDOW_AUTOSIZE);
    namedWindow("inverted_binary_img", WINDOW_AUTOSIZE);
    namedWindow("labels", WINDOW_AUTOSIZE);
    namedWindow("result_img", WINDOW_AUTOSIZE);

    imshow(win_src, img_src); //入力画像を表示
    imshow("gray_img", gray_img); //グレースケール画像を表示
    imshow("gaussian_img", gaussian_img); //平坦化画像を表示
    imshow("laplacian_img_raw", laplacian_img_raw); //ラプラシアンフィルタの結果（0～255の範囲に収まらない)を表示
    imshow("laplacian_img_abs", laplacian_img_abs); //ラプラシアンフィルタの結果（0～255の範囲に収まる)を表示
    imshow("inverted_binary_img", inverted_binary_img); //交点検出画像を表示
    //imshow("labels", labels); //交点検出画像を表示
    imshow("result_img", result_img); //交点検出画像を表示


    waitKey(0);

    capture.release();
    return 0;
}
