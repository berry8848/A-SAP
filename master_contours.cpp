#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> //画像入出力＆GUI操作用
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";


int main()
{
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

    //グレースケール画像に対して2値化
    Mat binary_img;
    threshold(gaussian_img, binary_img, 128, 255, THRESH_BINARY);

    //画素値を反転．反転しないとウインドウ全体が輪郭認識される．
    Mat inverted_binary_img = 255 - binary_img;

    // 輪郭を格納するベクトル
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy; //Vec4iは階層情報を表現するために使用されるデータ型．各要素は階層情報の異なる側面を表す．

    // 輪郭抽出
    findContours(inverted_binary_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1); //RETR_EXTERNAL:一番外側の白輪郭を取り出す

    // contoursの要素を表示
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            cout << "(" << point.x << ", " << point.y << ") " << endl;
        }
        cout << "    " << endl;
    }

    // hierarchyを表示
    for (int i = 0; i < contours.size(); i++) {
        cout << "Contour " << i << ", Hierarchy: "
            << ", Next: " << hierarchy[i][0]
            << ", Previous: " << hierarchy[i][1]
            << ", Child: " << hierarchy[i][2]
            << "Parent: " << hierarchy[i][3]
            << endl;
    }

    // 輪郭の描画
    drawContours(result_img, contours, -1, Scalar(0, 0, 255), 2);


    // 結果表示
    imshow(win_src, img_src); //入力画像を表示
    //imshow("gray_img", gray_img); //グレースケール画像を表示
    //imshow("gaussian_img", gaussian_img); //平坦化画像を表示
    imshow("inverted_binary_img", inverted_binary_img); //二値化反転画像を表示
    imshow("result_img", result_img); //輪郭抽出結果を表示

    // 画像を保存する
    std::string filename1 = "gray_img.jpg";
    std::string filename2 = "gaussian_img.jpg";
    std::string filename3 = "binary_img.jpg";
    std::string filename4 = "inverted_binary_img.jpg";
    std::string filename5 = "result_img.jpg";
    bool success1 = cv::imwrite(filename1, gray_img);
    bool success2 = cv::imwrite(filename2, gaussian_img);
    bool success3 = cv::imwrite(filename3, binary_img);
    bool success4 = cv::imwrite(filename4, inverted_binary_img);
    bool success5 = cv::imwrite(filename5, result_img);

    waitKey(0);

    //capture.release();
    return 0;
}
