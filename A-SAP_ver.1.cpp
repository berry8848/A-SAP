#define _CRT_SECURE_NO_WORNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <bitset> //細線化(Hilditchの方法)用
#include <opencv2/opencv.hpp>
//#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui/highgui.hpp> //画像入出力＆GUI操作用
#include <string> //csvファイル書き込み用
#include <fstream> //csvファイル書き込み用
#include <algorithm> //sort関数用
using namespace std;
using namespace cv;
string win_src = "src";
string win_dst = "dst";

//細線化(Zhang-Suenの方法)
void thinningIte(Mat& img, int pattern) {

    Mat del_marker = Mat::ones(img.size(), CV_8UC1);
    int x, y;

    for (y = 1; y < img.rows - 1; ++y) {

        for (x = 1; x < img.cols - 1; ++x) {

            int v9, v2, v3;
            int v8, v1, v4;
            int v7, v6, v5;

            v1 = img.data[y * img.step + x * img.elemSize()];
            v2 = img.data[(y - 1) * img.step + x * img.elemSize()];
            v3 = img.data[(y - 1) * img.step + (x + 1) * img.elemSize()];
            v4 = img.data[y * img.step + (x + 1) * img.elemSize()];
            v5 = img.data[(y + 1) * img.step + (x + 1) * img.elemSize()];
            v6 = img.data[(y + 1) * img.step + x * img.elemSize()];
            v7 = img.data[(y + 1) * img.step + (x - 1) * img.elemSize()];
            v8 = img.data[y * img.step + (x - 1) * img.elemSize()];
            v9 = img.data[(y - 1) * img.step + (x - 1) * img.elemSize()];

            int S = (v2 == 0 && v3 == 1) + (v3 == 0 && v4 == 1) +
                (v4 == 0 && v5 == 1) + (v5 == 0 && v6 == 1) +
                (v6 == 0 && v7 == 1) + (v7 == 0 && v8 == 1) +
                (v8 == 0 && v9 == 1) + (v9 == 0 && v2 == 1);

            int N = v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9;

            int m1 = 0, m2 = 0;

            if (pattern == 0) m1 = (v2 * v4 * v6);
            if (pattern == 1) m1 = (v2 * v4 * v8);

            if (pattern == 0) m2 = (v4 * v6 * v8);
            if (pattern == 1) m2 = (v2 * v6 * v8);

            if (S == 1 && (N >= 2 && N <= 6) && m1 == 0 && m2 == 0)
                del_marker.data[y * del_marker.step + x * del_marker.elemSize()] = 0;
        }
    }

    img &= del_marker;
}
//細線化(Zhang-Suenの方法の続き)
void thinning(const Mat& src, Mat& dst) {
    dst = src.clone();
    dst /= 255;         // 0は0 , 1以上は1に変換される

    Mat prev = Mat::zeros(dst.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIte(dst, 0);
        thinningIte(dst, 1);
        absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (countNonZero(diff) > 0);

    dst *= 255;
}

//細線化(Hilditchの方法)
void hilditchThinning(const unsigned char* src, unsigned char* dst, int w, int h)
{
    int offset[9][2] = { {0,0}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}, {0,1}, {1,1} };
    int nOdd[4] = { 1, 3, 5, 7 };
    int b[9];
    int px, py;
    bitset<6> condition;

    memcpy(dst, src, w * h);

    int path = 1;
    int counter;

    auto funcNc8 = [&nOdd](int* b)
    {
        array<int, 10> d;
        int j;
        int sum = 0;

        for (int i = 0; i <= 9; ++i)
        {
            j = i;
            if (i == 9) j = 1;
            if (abs(*(b + j)) == 1)
                d[i] = 1;
            else
                d[i] = 0;
        }

        for (int i = 0; i < 4; ++i)
        {
            j = nOdd[i];
            sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
        }

        return sum;
    };

    cout << "start thinning " << endl;
    clock_t beginTime = clock();

    do {
        cout << ".";
        counter = 0;

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                for (int i = 0; i < 9; ++i)
                {
                    b[i] = 0;
                    px = x + offset[i][0];
                    py = y + offset[i][1];
                    if (px >= 0 && px < w && py >= 0 && py < h)
                    {
                        int idx = w * py + px;
                        if (dst[idx] == 255)
                        {
                            b[i] = 1;
                        }
                        else if (dst[idx] == 127)
                        {
                            b[i] = -1;
                        }
                    }
                }

                condition.reset();

                // Condition 1
                if (b[0] == 1) condition.set(0, true);

                // Condition 2
                int sum = 0;
                for (int i = 0; i < 4; ++i)
                {
                    sum = sum + 1 - abs(b[nOdd[i]]);
                }
                if (sum >= 1) condition.set(1, true);

                // Condition 3
                sum = 0;
                for (int i = 1; i <= 8; ++i)
                {
                    sum = sum + abs(b[i]);
                }
                if (sum >= 2) condition.set(2, true);

                // Condition 4
                sum = 0;
                for (int i = 1; i <= 8; ++i)
                {
                    if (b[i] == 1) ++sum;
                }
                if (sum >= 1) condition.set(3, true);

                // Condition 5
                if (funcNc8(b) == 1) condition.set(4, true);

                // Condition 6
                sum = 0;
                for (int i = 1; i <= 8; ++i)
                {
                    if (b[i] != -1)
                    {
                        ++sum;
                    }
                    else {
                        int copy = b[i];
                        b[i] = 0;
                        if (funcNc8(b) == 1) ++sum;
                        b[i] = copy;
                    }
                }
                if (sum == 8) condition.set(5, true);

                // Final judgement
                if (condition.all())
                {
                    int idx = y * w + x;
                    dst[idx] = 127;
                    ++counter;
                }
            }
        }

        if (counter != 0)
        {
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    int idx = y * w + x;
                    if (dst[idx] == 127)
                    {
                        dst[idx] = 0;
                    }
                }
            }
        }

        ++path;
    } while (counter != 0);

    clock_t endTime = clock() - beginTime;
    cout << " Done! Time: " << (double)(endTime) / CLOCKS_PER_SEC << " sec, Num Path: " << path << endl;
}

//細線化(ChatGPT)
void thinning2(cv::Mat& image) {
    cv::Mat prevImage;
    cv::Mat diffImage;
    do {
        image.copyTo(prevImage);

        for (int y = 1; y < image.rows - 1; ++y) {
            for (int x = 1; x < image.cols - 1; ++x) {
                if (image.at<uchar>(y, x) == 255) {
                    int a = image.at<uchar>(y - 1, x);
                    int b = image.at<uchar>(y + 1, x);
                    int c = image.at<uchar>(y, x - 1);
                    int d = image.at<uchar>(y, x + 1);

                    int np = 0;
                    if (a == 0 && b == 255)
                        ++np;
                    if (b == 0 && c == 255)
                        ++np;
                    if (c == 0 && d == 255)
                        ++np;
                    if (d == 0 && a == 255)
                        ++np;

                    if (np == 1 && d == 0)
                        image.at<uchar>(y, x) = 0;
                }
            }
        }

        cv::absdiff(image, prevImage, diffImage);

    } while (cv::countNonZero(diffImage) > 0);
}

//細線化(Hit-or-Miss変換 by ChatGPT)
cv::Mat hitOrMiss(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);

    cv::Mat dilated;
    cv::dilate(image, dilated, kernel);

    cv::Mat eroded;
    cv::erode(image, eroded, kernel);

    cv::subtract(image, eroded, result);
    cv::subtract(cv::Scalar(255), dilated, result, cv::Mat(), CV_8U);

    return result;
}

//細線化(メドナム法 by ChatGPT)
cv::Mat medialAxisTransform(const cv::Mat& binaryImage) {
    cv::Mat distanceMap;
    cv::distanceTransform(binaryImage, distanceMap, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    cv::Mat medialAxis = cv::Mat::zeros(binaryImage.size(), CV_8U);
    for (int i = 0; i < binaryImage.rows; i++) {
        for (int j = 0; j < binaryImage.cols; j++) {
            if (binaryImage.at<uchar>(i, j) == 255 && distanceMap.at<float>(i, j) <= 1.0)
                medialAxis.at<uchar>(i, j) = 255;
        }
    }

    return medialAxis;
}

//細線化(Hilditch法 by ChatGPT)
cv::Mat hilditchThinning(const cv::Mat& binaryImage) {
    cv::Mat skeleton = binaryImage.clone();
    cv::Mat prevSkeleton;

    do {
        prevSkeleton = skeleton.clone();

        for (int i = 1; i < skeleton.rows - 1; i++) {
            for (int j = 1; j < skeleton.cols - 1; j++) {
                if (skeleton.at<uchar>(i, j) == 0)
                    continue;

                int p2 = skeleton.at<uchar>(i - 1, j);
                int p3 = skeleton.at<uchar>(i - 1, j + 1);
                int p4 = skeleton.at<uchar>(i, j + 1);
                int p5 = skeleton.at<uchar>(i + 1, j + 1);
                int p6 = skeleton.at<uchar>(i + 1, j);
                int p7 = skeleton.at<uchar>(i + 1, j - 1);
                int p8 = skeleton.at<uchar>(i, j - 1);
                int p9 = skeleton.at<uchar>(i - 1, j - 1);

                int a1 = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) +
                    (p5 == 0 && p6 == 1) + (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int a2 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int a3 = (p2 == 0 || p4 == 0 || (p6 == 0 && p8 == 0));
                int a4 = (p4 == 0 || p6 == 0 || (p8 == 0 && p2 == 0));

                if (a1 == 1 && (a2 >= 2 && a2 <= 6) && a3 == 1 && a4 == 1)
                    skeleton.at<uchar>(i, j) = 0;
            }
        }
    } while (cv::countNonZero(skeleton - prevSkeleton) > 0);

    return skeleton;
}

//細線化(田村の方法 by ChatGPT)
Mat tamuraThinning(const Mat& binaryImage) {
    Mat skeleton = binaryImage.clone();
    Mat prevSkeleton;

    do {
        prevSkeleton = skeleton.clone();

        for (int i = 1; i < skeleton.rows - 1; i++) {
            for (int j = 1; j < skeleton.cols - 1; j++) {
                if (skeleton.at<uchar>(i, j) == 0)
                    continue;

                int p2 = skeleton.at<uchar>(i - 1, j);
                int p3 = skeleton.at<uchar>(i - 1, j + 1);
                int p4 = skeleton.at<uchar>(i, j + 1);
                int p5 = skeleton.at<uchar>(i + 1, j + 1);
                int p6 = skeleton.at<uchar>(i + 1, j);
                int p7 = skeleton.at<uchar>(i + 1, j - 1);
                int p8 = skeleton.at<uchar>(i, j - 1);
                int p9 = skeleton.at<uchar>(i - 1, j - 1);

                int a1 = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) +
                    (p5 == 0 && p6 == 1) + (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int a2 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                if (a1 == 1 && (a2 >= 2 && a2 <= 6)) {
                    if (!(p2 && p4 && p6) && !(p4 && p6 && p8)) {
                        skeleton.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    } while (countNonZero(skeleton - prevSkeleton) > 0);

    return skeleton;
}

//ネガポジ変換
void convertToPositive(cv::Mat& image) {
    // 画像の各ピクセルにアクセスしてネガポジ変換を行う
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
            pixel[0] = 255 - pixel[0];  // Blue
            pixel[1] = 255 - pixel[1];  // Green
            pixel[2] = 255 - pixel[2];  // Red
        }
    }
}


int main()
{
    Mat img_src, img_dst;
    Mat gray_img;
    Mat edge;
    VideoCapture capture(0);//カメラオープン
    if (!capture.isOpened()) {
        cout << "error" << endl;
        return -1;
    }

    //ウインドウ生成
    namedWindow(win_src, WINDOW_AUTOSIZE);
    namedWindow("Harris", WINDOW_AUTOSIZE);
    namedWindow("gaussian_img", WINDOW_AUTOSIZE);
    namedWindow("laplacian_img", WINDOW_AUTOSIZE);

    //namedWindow("thres_binary", WINDOW_AUTOSIZE);
    //namedWindow("img_thinning", WINDOW_AUTOSIZE);

    //ファイル書き込み
    string output_csv_file_path = "Output/result.csv";
    // 書き込むcsvファイルを開く(std::ofstreamのコンストラクタで開く)
    ofstream ofs_csv_file(output_csv_file_path);

    //コーナー検出
    // 参考：http://opencv.jp/opencv2-x-samples/corner_detection/
    //１枚だけ写真を撮る
    capture >> img_src; //カメラ映像の読み込み
    Mat harris_img = img_src.clone();

    cvtColor(img_src, gray_img, COLOR_BGR2GRAY);
    GaussianBlur(gray_img, gray_img, Size(9, 9), 0, 0);
    Mat gaussian_img = gray_img.clone();
    // ラプラシアンフィルタの適用
    cv::Mat laplacian_img;
    cv::Laplacian(gray_img, laplacian_img, CV_16S, 5);
    cv::Mat a = laplacian_img.clone();
    cv::convertScaleAbs(laplacian_img, laplacian_img);
    //normalize(gray_img, gray_img, 0, 255, NORM_MINMAX);

    //細線化(OpenCV拡張モジュール)
    //ximgproc::thinning(gray_img, gray_img);

    //細線化(Zhang-Suen)
    //threshold(gray_img, gray_img, 180, 255, THRESH_BINARY);
    ////// ネガポジ変換を行う
    //convertToPositive(gray_img);
    //Mat img_thres_binary = gray_img.clone();
    //thinning(gray_img, gray_img);

    //細線化(ChatGPT)
    // 2値化処理を行う（任意の閾値を指定）
    //cv::threshold(gray_img, gray_img, 35,255, cv::THRESH_BINARY);
    //thinning2(gray_img);

    //細線化(Hilditchの方法)
    //unsigned char convertedImage;
    //gray_img.convertTo(convertedImage, CV_8UC1);
    //hilditchThinning(gray_img, gray_img, gray_img.cols, gray_img.rows);

    //細線化(Hit-or-Miss変換 by ChatGPT)
    //動かない
    //threshold(gray_img, gray_img, 128, 255, THRESH_BINARY);
    //Mat img_thres_binary = gray_img.clone();
    //Mat kernel = (cv::Mat_<char>(3, 3) << 0, -1, 0, 1, 1, 1, 0, -1, 0); // カーネルパターンを定義
    //gray_img = hitOrMiss(gray_img, kernel);

    //細線化(メドナム法 by ChatGPT)
    // 細線化されるが、オブジェクトの両端を検出してしまう
    //threshold(gray_img, gray_img, 128, 255, cv::THRESH_BINARY);
    //Mat img_thres_binary = gray_img.clone();
    //gray_img = medialAxisTransform(gray_img);

    //細線化(Hilditch法 by ChatGPT)
    //細線化されない
    //threshold(gray_img, gray_img, 180, 255, cv::THRESH_BINARY);
    //Mat img_thres_binary = gray_img.clone();
    //gray_img = hilditchThinning(gray_img);

    //細線化(田村の方法 by ChatGPT)
    //threshold(gray_img, gray_img, 120, 255, cv::THRESH_BINARY);
    //Mat img_thres_binary = gray_img.clone();
    //gray_img = tamuraThinning(gray_img);

    //Mat img_thinning = gray_img.clone();




    vector<Point2f> corners;
    goodFeaturesToTrack(laplacian_img, corners, 80, 0.01, 30, Mat(), 3, true); //コーナーの検出

    // y座標が小さい順にソート
    sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
        });
    // x座標が小さい順にソート
    sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y == b.y) ? (a.x < b.x) : false;
        });
    cout << corners << "\n";

    // 出力画像の作成
    vector<Point2f>::iterator it_corner = corners.begin();
    it_corner = corners.begin();
    for (; it_corner != corners.end(); ++it_corner) {
        circle(harris_img, Point(it_corner->x, it_corner->y), 1, Scalar(0, 255, 0), -1); //関数の説明 http://opencv.jp/opencv-2svn/cpp/drawing_functions.html
        ofs_csv_file << it_corner->x << ", " << it_corner->y << endl;
        circle(harris_img, Point(it_corner->x, it_corner->y), 8, Scalar(0, 255, 0));
    }

    imshow(win_src, img_src); //入力画像を表示
    imshow("Harris", harris_img); //出力画像を表示
    imshow("gaussian_img", gaussian_img); //出力画像を表示
    imshow("laplacian_img", laplacian_img); //出力画像を表示
    imshow("a", a); //出力画像を表示
    //imshow("thres_binary", img_thres_binary);
    //imshow("img_thinning", img_thinning);
    waitKey(0);

    capture.release();
    return 0;
}
