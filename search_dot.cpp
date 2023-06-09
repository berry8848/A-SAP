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
#define THRESHOLD 128
#define NUMBER_OF_DOTS 25


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


//背景領域を除いた領域のうち，上からNUMBER_OF_DOTS個だけ，大きい領域のラベル番号を抽出
 vector<int> extractLabels(Mat& matrix){
    vector<int> extract_labels(NUMBER_OF_DOTS); //抽出したラベル番号用．ドットの数だけ箱を用意．
    int max_current = matrix.at<int>(1, 4);
    int dot_label_num = 0; //n番目に大きい要素のラベル番号格納用．

    for (int i = 0; i < NUMBER_OF_DOTS; i++){

        //背景領域（ラベル0）をのぞいたi番目に大きい要素を探索
        for (int j = 1; j < matrix.rows; j++) {
            if (max_current < matrix.at<int>(j, 4))
            {  
                max_current = matrix.at<int>(j, 4);
                dot_label_num = j;
            }
        }
        extract_labels[i] = dot_label_num;        
    }
    
    return extract_labels;
}

int main()
{
    //ファイル書き込み
    string output_csv_file_path = "Output/result.csv";
    // 書き込むcsvファイルを開く(std::ofstreamのコンストラクタで開く)
    ofstream ofs_csv_file(output_csv_file_path);

    Mat img_src;


    //カメラ使用時
    VideoCapture capture(0);//カメラオープン
    if (!capture.isOpened()) {
        cout << "error" << endl;
        return -1;
    }
    // 撮影画像サイズの設定
    bool bres = capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    if (bres != true) {
        return -1;
    }
    bres = capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    if (bres != true) {
        return -1;
    }
    // 撮影画像取得高速化の工夫
    bres = capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (bres != true) {
        return -1;
    }
    //コーナー検出
    // 参考：http://opencv.jp/opencv2-x-samples/corner_detection/
    //１枚だけ写真を撮る ※現在，画像の読み込みに変更
    capture >> img_src; //カメラ映像の読み込み
    Mat result_img = img_src.clone(); //出力画像用 

    //// 画像ファイル使用時
    //string filename = "image.jpg";
    //// 画像を読み込む
    //img_src = imread(filename);
    //if (img_src.empty()) {
    //    cout << "Failed to load the image: " << filename << endl;
    //    return -1;
    //}
    //Mat result_img = img_src.clone(); //出力画像用 

    //グレースケール変換
    Mat gray_img;
    cvtColor(img_src, gray_img, COLOR_BGR2GRAY);

    //// 膨張処理
    //Mat dilated_img;
    //dilate(gray_img, dilated_img, Mat(), Point(-1, -1), 3);

    //// 収縮処理
    //Mat eroded_img;
    //erode(dilated_img, eroded_img, Mat(), Point(-1, -1), 3);


    // 二値化
    Mat binary_img;
    threshold(gray_img, binary_img, THRESHOLD, 255, THRESH_BINARY);




    //// 二値化画像の画素値を表示
    //for (int row = 0; row < binaryImage.rows; ++row) {
    //    for (int col = 0; col < binaryImage.cols; ++col) {
    //        int pixelValue = binaryImage.at<uchar>(row, col);
    //        std::cout << "Pixel value at (" << row << ", " << col << "): " << pixelValue << std::endl;
    //    }
    //}

    //画素値を反転
    Mat inverted_binary_img = 255 - binary_img;

    //領域分割
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(inverted_binary_img, labels, stats, centroids);
    cout << "labels.at<int>(200, 100): " << labels.at<int>(200, 100) << endl;
    cout << "labels.at<int>(0, 0): " << labels.at<int>(0, 0) << endl;
    cout << "stats: " << stats.rows << endl;

    vector<int> extract_labels = extractLabels(stats); //背景領域を除いた領域のうち，上からNUMBER_OF_DOTS個だけ，大きい領域のラベル番号を抽出

    for (int i = 0; i < extract_labels.size(); i++) {
        cout << "extract_labels[i]: " << extract_labels[i] << endl;
    }

    //// Mat型の各要素の値を表示
    //for (int i = 0; i < labels.rows; i++)
    //{
    //    for (int j = 0; j < labels.cols; j++)
    //    {
    //        //std::cout << "Pixel (" << i << ", " << j << "): ";
    //        std::cout << static_cast<int>(labels.at<int>(i, j)) << ", ";
    //    }
    //}
    
    ////座標変換((u, v)→(u, height - v))
    //float height = laplacian_img_abs.rows;
    //cout << "height: " << height << endl;
    //vector<Point2f> trans_corners(corners.size());
    //for (int i = 0; i < corners.size(); i++) {
    //    Point2f origin_pnt = corners[i];
    //    double u = origin_pnt.x;
    //    double v = origin_pnt.y;
    //    double trans_u = u;
    //    double trans_v = height - v;
    //    Point2f trans_pnt;
    //    trans_pnt.x = trans_u;
    //    trans_pnt.y = trans_v;
    //    trans_corners[i] = trans_pnt;
    //}

    ////// y座標が小さい順にソート
    ////sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    ////    return a.y < b.y;
    ////    });
    ////// x座標が小さい順にソート
    ////sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    ////    return (a.y == b.y) ? (a.x < b.x) : false;
    ////    });
    ////cout << corners << endl;;

    //// y座標が小さい順にソート
    //sort(trans_corners.begin(), trans_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    //    return a.y < b.y;
    //    });
    //// x座標が小さい順にソート
    //sort(trans_corners.begin(), trans_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    //    return (a.y == b.y) ? (a.x < b.x) : false;
    //    });
    //cout << "trans_corners: " << trans_corners << endl;


    ////(0, 0)に近い点から右並びにsort
    //vector<Point2f> sort_trans_corners; //sortした点列用
    //vector<Point2f> range_corners; //range内に含まれる点列用
    //double range = sqrt(pow(trans_corners[0].x - trans_corners[1].x, 2) + pow(trans_corners[0].y - trans_corners[1].y, 2)); //x座標の小さい順にsortする際の点列の範囲の決定．この値は一時的
    //Point2f reference_point(0.0f, 0.0f); //基準点格納用
    //cout << "int(sqrt(trans_corners.size())): " << int(sqrt(trans_corners.size())) << endl;

    //for (int i = 0; i < int(sqrt(trans_corners.size())); i++) {
    //    reference_point = getClosestPoint(trans_corners, reference_point); //基準点に一番近い点を新たな基準点とする
    //    range_corners = sortWithinRange(trans_corners, reference_point, range); //reference_pointを基準とし，rangeに含まれる点列をtrans_cornersから抽出し，x座標の小さい順にsortした結果を返す
    //    //sort_trans_cornersに格納
    //    for (const Point2f& corner : range_corners) {
    //        sort_trans_corners.push_back(corner);
    //    }
    //    reference_point = Point2f(reference_point.x, reference_point.y + range); //基準点のy座標にrangeだけ足した座標に一番近い点を新たな基準点とする．
    //}
    //cout << "sort_trans_corners: " << sort_trans_corners << endl;

    ////// 出力画像の作成
    //vector<Point2f>::iterator it_corner = corners.begin();
    //it_corner = corners.begin();
    //for (; it_corner != corners.end(); ++it_corner) {
    //    circle(result_img, Point(it_corner->x, it_corner->y), 1, Scalar(0, 255, 0), -1); //関数の説明 http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
    //    ofs_csv_file << it_corner->x << ", " << it_corner->y << endl;
    //    circle(result_img, Point(it_corner->x, it_corner->y), 8, Scalar(0, 255, 0));
    //}


    //// 結果表示
    ////ウインドウ生成
    namedWindow(win_src, WINDOW_AUTOSIZE);
    namedWindow("gray_img", WINDOW_AUTOSIZE);
    namedWindow("dilated_img", WINDOW_AUTOSIZE);
    namedWindow("eroded_img", WINDOW_AUTOSIZE);
    namedWindow("binary_img", WINDOW_AUTOSIZE);
    //namedWindow("result_img", WINDOW_AUTOSIZE);

    imshow(win_src, img_src); //入力画像を表示
    imshow("gray_img", gray_img); //グレースケール画像を表示
    //imshow("dilated_img", dilated_img); //膨張処理画像を表示
    //imshow("eroded_img", eroded_img); //収縮処理画像を表示
    imshow("binary_img", binary_img); //2値化画像を表示
    //imshow("result_img", result_img); //交点検出画像を表示

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

    capture.release();
    return 0;
}
