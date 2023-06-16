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
#define SEARCH_RANGE 2
#define NUMBER_OF_CORNERS 24
#define NUMBER_OF_MASTERS 6


//範囲内で0以外の値を取るラベルを返す
int searchLabelInRange(const Point2f& corner, Mat& labels, vector<vector<Point2f>>& master_labels) {
    // 画像の幅と高さを取得
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
    cout << "ERROR：コーナーのラベル付けができませんでした．" << endl;
    return 0;
}

//各行の列のサイズを確認し，指定した列のサイズ以外の時エラーメッセージを表示
void checkColumnSize(const vector<vector<Point2f>>& matrix) {
    int expectedSize = NUMBER_OF_CORNERS / NUMBER_OF_MASTERS;
    for (int targetRow = 0; targetRow < matrix.size(); ++targetRow) {
        int actualColumnSize = matrix[targetRow].size();  // 列のサイズは最初の行の要素数とする
        if (actualColumnSize != expectedSize) {
            cout << "ERROR: Row " << targetRow << " は期待しているサイズ数ではありません．" << endl;
        }
    }
}

int main()
{
    //ファイル書き込み
    string output_csv_file_path = "Output/result.csv";
    // 書き込むcsvファイルを開く(std::ofstreamのコンストラクタで開く)
    ofstream ofs_csv_file(output_csv_file_path);

    Mat img_src;
    //VideoCapture capture(0);//カメラオープン
    //if (!capture.isOpened()) {
    //    cout << "error" << endl;
    //    return -1;
    //}
    ////コーナー検出
    //// 参考：http://opencv.jp/opencv2-x-samples/corner_detection/
    ////１枚だけ写真を撮る ※現在，画像の読み込みに変更
    //capture >> img_src; //カメラ映像の読み込み
    //Mat result_img = img_src.clone(); //出力画像用 

    // 画像ファイルのパス
    string filename = "master.jpg";
    // 画像を読み込む
    img_src = imread(filename);
    if (img_src.empty()) {
        cout << "Failed to load the image: " << filename << endl;
        return -1;
    }
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
    cout << "labels.at<int>(200, 100): " << labels.at<int>(200, 100) << endl;
    cout << "labels.at<int>(0, 0): " << labels.at<int>(0, 0) << endl;
    cout << "stats: " << stats << endl;

    //コーナーのラベル付け
    int corner_labels[NUMBER_OF_CORNERS];
    vector<vector<Point2f>> master_labels(NUMBER_OF_MASTERS, vector<Point2f>(0)); //マスター配置の四隅の座標値を格納．例：master_labels[][] = (x, y)
    for (int i = 0; i < NUMBER_OF_CORNERS; i++) {
        corner_labels[i] = searchLabelInRange(corners.at(i), labels, master_labels);
        cout << "corner_labels: " << corner_labels[i] << endl;
    }

    //正常にラベル付けできているかを確認
    checkColumnSize(master_labels);

    // 値の表示
    for (int i = 0; i < master_labels.size(); ++i) {
        for (int j = 0; j < master_labels[i].size(); ++j) {
            cout << "master_labels[" << i << "][" << j << "]: (" << master_labels[i][j].x << ", " << master_labels[i][j].y << ")" << endl;
        }
    }

    //// 出力画像の作成（本番用）
    //vector<Point2f>::iterator it_corner = corners.begin();
    //it_corner = corners.begin();
    //for (; it_corner != corners.end(); ++it_corner) {
    //    circle(result_img, Point(it_corner->x, it_corner->y), 1, Scalar(0, 255, 0), -1); //関数の説明 http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
    //    ofs_csv_file << it_corner->x << ", " << it_corner->y << endl;
    //    circle(result_img, Point(it_corner->x, it_corner->y), 8, Scalar(0, 255, 0));
    //}

    // 出力画像の作成（確認用）
    for (int i = 0; i < NUMBER_OF_MASTERS; i++) {
        if (i % NUMBER_OF_MASTERS == 0) {
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 1, Scalar(0, 255, 0), -1);
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 8, Scalar(0, 255, 0));
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 1, Scalar(0, 255, 0), -1);
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 8, Scalar(0, 255, 0));
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 1, Scalar(0, 255, 0), -1);
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 8, Scalar(0, 255, 0));
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 1, Scalar(0, 255, 0), -1);
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 8, Scalar(0, 255, 0));
        }
        if (i % NUMBER_OF_MASTERS == 1) {
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 1, Scalar(255, 0, 0), -1);
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 8, Scalar(255, 0, 0));
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 1, Scalar(255, 0, 0), -1);
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 8, Scalar(255, 0, 0));
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 1, Scalar(255, 0, 0), -1);
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 8, Scalar(255, 0, 0));
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 1, Scalar(255, 0, 0), -1);
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 8, Scalar(255, 0, 0));

        }
        if (i % NUMBER_OF_MASTERS == 2) {
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 1, Scalar(0, 0, 255), -1);
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 8, Scalar(0, 0, 255));
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 1, Scalar(0, 0, 255), -1);
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 8, Scalar(0, 0, 255));
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 1, Scalar(0, 0, 255), -1);
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 8, Scalar(0, 0, 255));
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 1, Scalar(0, 0, 255), -1);
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 8, Scalar(0, 0, 255));
        }
        if (i % NUMBER_OF_MASTERS == 3) {
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 1, Scalar(0, 255, 255), -1);
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 8, Scalar(0, 255, 255));
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 1, Scalar(0, 255, 255), -1);
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 8, Scalar(0, 255, 255));
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 1, Scalar(0, 255, 255), -1);
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 8, Scalar(0, 255, 255));
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 1, Scalar(0, 255, 255), -1);
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 8, Scalar(0, 255, 255));
        }
        if (i % NUMBER_OF_MASTERS == 4) {
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 1, Scalar(255, 0, 255), -1);
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 8, Scalar(255, 0, 255));
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 1, Scalar(255, 0, 255), -1);
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 8, Scalar(255, 0, 255));
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 1, Scalar(255, 0, 255), -1);
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 8, Scalar(255, 0, 255));
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 1, Scalar(255, 0, 255), -1);
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 8, Scalar(255, 0, 255));
        }
        if (i % NUMBER_OF_MASTERS == 5) {
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 1, Scalar(255, 255, 0), -1);
            circle(result_img, Point(master_labels[i][0].x, master_labels[i][0].y), 8, Scalar(255, 255, 0));
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 1, Scalar(255, 255, 0), -1);
            circle(result_img, Point(master_labels[i][1].x, master_labels[i][1].y), 8, Scalar(255, 255, 0));
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 1, Scalar(255, 255, 0), -1);
            circle(result_img, Point(master_labels[i][2].x, master_labels[i][2].y), 8, Scalar(255, 255, 0));
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 1, Scalar(255, 255, 0), -1);
            circle(result_img, Point(master_labels[i][3].x, master_labels[i][3].y), 8, Scalar(255, 255, 0));
        }
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

    //capture.release();
    return 0;
}
