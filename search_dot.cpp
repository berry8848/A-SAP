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
    vector<int> used(matrix.rows, 0); // extract_labelsに追加する要素番目を1にする．これをしないと，面積が同じ値の領域を抽出することができない．
    int max_current = 1000000; //この値以下の面積のうち最大の面積の領域を探索する．
    int dot_label_num = 0; //n番目に大きい要素のラベル番号格納用．


    for (int i = 0; i < NUMBER_OF_DOTS; i++){
        int max = 0;
        //背景領域（ラベル0）をのぞいたi番目に大きい要素を探索
        for (int j = 1; j < matrix.rows; j++) {
            if (max < matrix.at<int>(j, 4) && matrix.at<int>(j, 4) <= max_current && used[j] == 0)
            {  
                max = matrix.at<int>(j, 4);
                dot_label_num = j;
            }
        }
        extract_labels[i] = dot_label_num;
        used[dot_label_num] = 1;
        max_current = matrix.at<int>(dot_label_num, 4);
    }
    for (int i = 0; i < used.size(); i++) {
        std::cout << used[i] << " "; // 配列の要素を出力
    }
    std::cout << std::endl;

    return extract_labels;
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
 vector<Point2f> extractCenter(vector<int>& extract_labels, Mat& labels) {
     Point2f center; //中心座標格納用
     vector<Point2f> centers; //各ラベル番号の中心座標格納用
     vector<Point> target_label_points; //各ラベル番号の中心座標格納用

     for (int i = 0; i < extract_labels.size(); i++) {
         int search_num = extract_labels[i];
         //指定したラベル番号の座標値を探索
         for (int y = 0; y < labels.rows; y++) {
             for (int x = 0; x < labels.cols; x++) {
                 if (labels.at<int>(y, x) == search_num) {
                     target_label_points.push_back({ x, y });
                     //cout << "(x, y) = " << x << " , " << y << "  " << target_label_points.back() << endl;
                 }
             }
         }
         center = calculateCenter(target_label_points); //中心座標を計算
         centers.push_back(center);
         target_label_points.clear(); //格納した座標値を消去
     }
     return centers;
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
    string filename = "dot_image.jpg";
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

    // 二値化
    Mat binary_img;
    threshold(gray_img, binary_img, THRESHOLD, 255, THRESH_BINARY);

    //画素値を反転
    Mat inverted_binary_img = 255 - binary_img;

    //領域分割
    Mat labels, stats, centroids;
    int num_objects = connectedComponentsWithStats(inverted_binary_img, labels, stats, centroids);
    cout << "labels.at<int>(200, 100): " << labels.at<int>(200, 100) << endl;
    cout << "labels.at<int>(0, 0): " << labels.at<int>(0, 0) << endl;
    cout << "stats: " << stats << endl;

    vector<int> extract_labels = extractLabels(stats); //背景領域を除いた領域のうち，上からNUMBER_OF_DOTS個だけ，大きい領域のラベル番号を抽出

    for (int i = 0; i < extract_labels.size(); i++) {
        cout << "extract_labels[" << i << "] : " << extract_labels[i] << endl;
    }


    vector<Point2f> centers;
    centers = extractCenter(extract_labels, labels); //抽出したラベル番号ごとに，そのラベル番号を満たす座標を抽出し，抽出した座標値の中心座標を返す．

    
    //座標変換((u, v)→(u, height - v))
    float height = img_src.rows;
    cout << "height: " << height << endl;
    vector<Point2f> trans_centers(centers.size());
    for (int i = 0; i < centers.size(); i++) {
        Point2f origin_pnt = centers[i];
        double u = origin_pnt.x;
        double v = origin_pnt.y;
        double trans_u = u;
        double trans_v = height - v;
        Point2f trans_pnt;
        trans_pnt.x = trans_u;
        trans_pnt.y = trans_v;
        trans_centers[i] = trans_pnt;
    }

    //y座標が小さい順にソート
    sort(trans_centers.begin(), trans_centers.end(), [](const Point2f& a, const Point2f& b) {
        return a.y < b.y;
        });
    //x座標が小さい順にソート
    sort(trans_centers.begin(), trans_centers.end(), [](const Point2f& a, const Point2f& b) {
        return (a.y == b.y) ? (a.x < b.x) : false;
        });
    cout << "trans_centers: " << trans_centers << endl;


    //(0, 0)に近い点から右並びにソート
    vector<Point2f> sort_trans_centers; //ソートした点列用
    vector<Point2f> range_centers; //range内に含まれる点列用
    double range = sqrt(pow(trans_centers[0].x - trans_centers[1].x, 2) + pow(trans_centers[0].y - trans_centers[1].y, 2)); //x座標の小さい順にsortする際の点列の範囲の決定．この値は一時的
    Point2f reference_point(0.0f, 0.0f); //基準点格納用
    cout << "int(sqrt(trans_corners.size())): " << int(sqrt(trans_centers.size())) << endl;

    for (int i = 0; i < int(sqrt(trans_centers.size())); i++) {
        reference_point = getClosestPoint(trans_centers, reference_point); //基準点に一番近い点を新たな基準点とする
        range_centers = sortWithinRange(trans_centers, reference_point, range); //reference_pointを基準とし，rangeに含まれる点列をtrans_cornersから抽出し，x座標の小さい順にsortした結果を返す
        //sort_trans_cornersに格納
        for (const Point2f& corner : range_centers) {
            sort_trans_centers.push_back(corner);
        }
        reference_point = Point2f(reference_point.x, reference_point.y + range); //基準点のy座標にrangeだけ足した座標に一番近い点を新たな基準点とする．
    }
    cout << "sort_trans_centers: " << sort_trans_centers << endl;

    //// 出力画像の作成
    vector<Point2f>::iterator it_corner = centers.begin();
    it_corner = centers.begin();
    for (; it_corner != centers.end(); ++it_corner) {
        circle(result_img, Point(it_corner->x, it_corner->y), 1, Scalar(0, 255, 0), -1); //関数の説明 http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
        ofs_csv_file << it_corner->x << ", " << it_corner->y << endl;
        circle(result_img, Point(it_corner->x, it_corner->y), 8, Scalar(0, 255, 0));
    }


    //// 結果表示
    ////ウインドウ生成
    namedWindow(win_src, WINDOW_AUTOSIZE);
    namedWindow("gray_img", WINDOW_AUTOSIZE);
    namedWindow("dilated_img", WINDOW_AUTOSIZE);
    namedWindow("eroded_img", WINDOW_AUTOSIZE);
    namedWindow("binary_img", WINDOW_AUTOSIZE);
    namedWindow("result_img", WINDOW_AUTOSIZE);

    imshow(win_src, img_src); //入力画像を表示
    imshow("gray_img", gray_img); //グレースケール画像を表示
    //imshow("dilated_img", dilated_img); //膨張処理画像を表示
    //imshow("eroded_img", eroded_img); //収縮処理画像を表示
    imshow("binary_img", binary_img); //2値化画像を表示
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
