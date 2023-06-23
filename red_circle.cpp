//目的：点探索し，ひずみ補正関数作成の準備 二値化→領域分割→ノイズ除去→ラベルごとに中心座標を求める→座標変換→ソート

#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //画像入出力＆GUI操作用
#include <opencv2/core/core.hpp> //識別子Point用
#include <string> //csvファイル書き込み用
#include <fstream> //csvファイル書き込み用
#include <algorithm> //sort関数用
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";
#define THRESHOLD 128
#define NUMBER_OF_DOTS 1


//背景領域を除いた領域のうち，最も大きい領域のラベル番号を抽出
int circleLabels(Mat& matrix) {
    int label_num = 0; //n番目に大きい要素のラベル番号格納用．
    int max = 0; //最大面積格納用

    //背景領域（ラベル0）をのぞいたi番目に大きい要素を探索
    for (int j = 1; j < matrix.rows; j++) {
        if (max < matrix.at<int>(j, 4))
        {
            max = matrix.at<int>(j, 4);
            label_num = j;
        }
    }    
    return label_num;
}

//複数の座標値を入力とし，それらの中心座標を出力とする
Point2f calculateCenter(const vector<Point>& points) {
    float centerX = 0.0;
    float centerY = 0.0;

    // 座標の合計を計算
    for (const auto& point : points) {
        centerX += point.x;
        centerY += point.y;
    }

    // 座標の数で割って中心座標を求める
    centerX /= points.size();
    centerY /= points.size();

    Point2f center;
    center.x = centerX;
    center.y = centerY;

    return center;
}


//抽出したラベル番号ごとに，そのラベル番号を満たす座標を抽出し，抽出した座標値の中心座標を返す
Point2f circleCenter(int circle_labels, Mat& labels) {
    Point2f center; //中心座標格納用
    vector<Point> target_label_points; //ラベル番号を満たす座標格納用

    //指定したラベル番号の座標値を探索
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            if (labels.at<int>(y, x) == circle_labels) {
                target_label_points.push_back({ x, y });
                //cout << "(x, y) = " << x << " , " << y << "  " << target_label_points.back() << endl;
            }
        }
    }
    center = calculateCenter(target_label_points); //中心座標を計算
    return center;
}


int main()
{
    //ファイル書き込み
    string output_csv_file_path = "Output/result.csv";
    // 書き込むcsvファイルを開く(std::ofstreamのコンストラクタで開く)
    ofstream ofs_csv_file(output_csv_file_path);

    Mat img_src;

    ////カメラ使用時
    //VideoCapture capture(0);//カメラオープン
    //if (!capture.isOpened()) {
    //    cout << "error" << endl;
    //    return -1;
    //}
    //// 撮影画像サイズの設定
    //bool bres = capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    //if (bres != true) {
    //    return -1;
    //}
    //bres = capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    //if (bres != true) {
    //    return -1;
    //}
    //// 撮影画像取得高速化の工夫
    //bres = capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    //if (bres != true) {
    //    return -1;
    //}
    ////コーナー検出
    //// 参考：http://opencv.jp/opencv2-x-samples/corner_detection/
    ////１枚だけ写真を撮る ※現在，画像の読み込みに変更
    //capture >> img_src; //カメラ映像の読み込み
    //Mat result_img = img_src.clone(); //出力画像用 

    // 画像ファイル使用時
    string filename = "red_circle.jpg";
    // 画像を読み込む
    img_src = imread(filename);
    if (img_src.empty()) {
        cout << "Failed to load the image: " << filename << endl;
        return -1;
    }
    Mat result_img = img_src.clone(); //出力画像用 

     // 画像をHSV色空間に変換
    Mat hsvImage;
    cvtColor(img_src, hsvImage, cv::COLOR_BGR2HSV);

    // 赤色の範囲を定義
    Scalar lowerRed(0, 100, 100);
    Scalar upperRed(10, 255, 255);
    Scalar lowerRed2(170, 100, 100);
    Scalar upperRed2(180, 255, 255);

    // 赤色領域を抽出
    Mat redMask1, redMask2, redMask, redImage;
    inRange(hsvImage, lowerRed, upperRed, redMask1);
    inRange(hsvImage, lowerRed2, upperRed2, redMask2);
    add(redMask1, redMask2, redMask);

    //領域分割
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(redMask, labels, stats, centroids);
    cout << "labels.at<int>(200, 100): " << labels.at<int>(200, 100) << endl;
    cout << "labels.at<int>(0, 0): " << labels.at<int>(0, 0) << endl;
    cout << "stats: " << stats << endl;

    int circle_labels = circleLabels(stats); //背景領域を除いた領域のうち，最も大きい領域のラベル番号を抽出

    cout << "circle_labels : " << circle_labels << endl;

    Point2f center;
    center = circleCenter(circle_labels, labels); //抽出したラベル番号ごとに，そのラベル番号を満たす座標を抽出し，抽出した座標値の中心座標を返す．

    ////座標変換((u, v)→(u, height - v))
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

    // 出力画像の作成
    circle(result_img, Point(center.x, center.y), 1, Scalar(0, 255, 0), -1); //関数の説明 http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
    circle(result_img, Point(center.x, center.y), 8, Scalar(0, 255, 0));
    ofs_csv_file << center.x << ", " << center.y << endl; //csvファイル出力

    // 結果表示
    imshow(win_src, img_src); //入力画像を表示
    imshow("Red Image", redMask); // 赤色領域の表示
    imshow("result_img", result_img); //交点検出画像を表示

    //// 画像を保存する
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