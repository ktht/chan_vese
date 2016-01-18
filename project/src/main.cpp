#include <iostream> // std::cout
#include <cstdlib> // EXIT_SUCCESS
#include <vector> // std::vector<>
#include <algorithm> // std::min()
#include <cmath> // std::pow(), std::sqrt(), std::sin()

#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor(), CV_BGR2RGB cv::threshold(),
                                       // cv::findContours(), cv::drawContours(),
                                       // cv::THRESH_BINARY, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE
#include <opencv2/highgui/highgui.hpp> // cv::imread(), CV_LOAD_IMAGE_COLOR, cv::WINDOW_NORMAL,
                                       // cv::imshow(), cv::waitKey(), cv::namedWindow()

                                       // cv::Mat, cv::Scalar, cv::Vec4i, cv::Point

#include <boost/math/special_functions/sign.hpp> // boost::math::sign()

#if defined(__gnu_linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <boost/math/constants/constants.hpp> // boost::math::constants::pi<>()

#if defined(__gnu_linux__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

typedef unsigned char uchar;
typedef std::vector<std::vector<double>> levelset;

int
levelset_rect(levelset & u,
              int w,
              int h,
              int l)
{
  if(! u.empty()) return 1;
  u.reserve(h);

  std::vector<double> tb_row(w, -1);
  std::vector<double> m_row(w, 1);
  std::fill(m_row.begin(), m_row.begin() + l, -1);
  std::fill(m_row.rbegin(), m_row.rbegin() + l, -1);

  for(int i = 0; i < l; ++i)     u.push_back(tb_row);
  for(int i = l; i < h - l; ++i) u.push_back(m_row);
  for(int i = h - l; i < h; ++i) u.push_back(tb_row);

  return 0;
}

int
levelset_circ(levelset & u,
              int w,
              int h,
              double d)
{
  if(! u.empty()) return 1;
  if(d < 0 || d > 1) return 2;

  const int r = std::min(w, h) * d / 2;
  const int mid_x = w / 2;
  const int mid_y = h / 2;

  u.reserve(h);
  for(int i = 0; i < h; ++i)
  {
    std::vector<double> row(w, -1);
    for(int j = 0; j < w; ++j)
    {
      const double d = std::sqrt(std::pow(mid_x - i, 2) +
                                 std::pow(mid_y - j, 2));
      if(d < r) row[j] = 1;
    }
    u.push_back(row);
  }

  return 0;
}

int
levelset_checkerboard(levelset & u,
                      int w,
                      int h)
{
  if(! u.empty()) return 1;

  u.reserve(h);
  const double pi = boost::math::constants::pi<double>();
  for(int i = 0; i < h; ++i)
  {
    std::vector<double> row;
    row.reserve(w);
    for(int j = 0; j < w; ++j)
      row.push_back(boost::math::sign(std::sin(pi * i / 5) *
                                      std::sin(pi * j / 5)));
    u.push_back(row);
  }

  return 0;
}

cv::Mat
levelset2contour(const levelset & u)
{
  const int h = u.size();
  const int w = u.at(0).size();
  cv::Mat c(h, w, CV_8UC1);

  for(int i = 0; i < h; ++i)
  {
    for(int j = 0; j < w; ++j)
    {
      const double val = u.at(i).at(j);
      c.at<uchar>(i, j) = val <= 0 ? 0 : 255;
    }
  }

  return c;
}

int
draw_contour(cv::Mat & dst,
             const levelset & u)
{
  cv::Mat c = levelset2contour(u);
  cv::Mat th;
  std::vector<std::vector<cv::Point>> cs;
  std::vector<cv::Vec4i> hier;
  cv::threshold(c, th, 100, 255, cv::THRESH_BINARY);
  cv::findContours(th, cs, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

  int idx = 0;
  for(; idx >= 0; idx = hier[idx][0])
  {
    cv::Scalar color(255, 0, 0);
    cv::drawContours(dst, cs, idx, color, 1, 8, hier);
  }
  return 0;
}

int
main()
{
  const cv::Mat _img = cv::imread("data/balls.png", CV_LOAD_IMAGE_COLOR);
  if(! _img.data)
  {
    std::cerr << "No such image!\n";
    return EXIT_FAILURE;
  }
  cv::Mat img;
  cv::cvtColor(_img, img, CV_BGR2RGB);
  const int h = img.rows;
  const int w = img.cols;
  levelset u;
  //levelset_rect(u, w, h, 4);
  //levelset_circ(u, w, h, 0.5);
  levelset_checkerboard(u, w, h);

  cv::Mat nw_img = img.clone();
  draw_contour(nw_img, u);

  cv::namedWindow("Display window", cv::WINDOW_NORMAL);
  cv::imshow("Display window", img);
  cv::waitKey(1000);
  std::cout << "R" << static_cast<int>(img.at<cv::Vec3b>(0, 0)[0]) << " "
               "G" << static_cast<int>(img.at<cv::Vec3b>(0, 0)[1]) << " "
               "B" << static_cast<int>(img.at<cv::Vec3b>(0, 0)[2]) << "\n";

  cv::imshow("Display window", nw_img);
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
