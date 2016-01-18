#include <cmath> // std::abs(), std::min()

#include <opencv2/highgui/highgui.hpp> // cv::imshow()

#include "InteractiveDataRect.hpp"// InteractiveData, cv::Mat, cv::Scalar, CV_64FC1,
                                  // cv::rectangle()

InteractiveDataRect::InteractiveDataRect(cv::Mat * const _img,
                                         const cv::Scalar & _contour_color)
  : InteractiveData(_img, _contour_color)
  , roi(0, 0, 0, 0)
{}

bool
InteractiveDataRect::is_ok() const
{
  return roi.width != 0 && roi.height != 0;
}

cv::Mat
InteractiveDataRect::get_levelset(int h,
                                  int w) const
{
  cv::Mat u = cv::Mat::zeros(h, w, CV_64FC1);
  u(roi) = 1;
  return u;
}

void
InteractiveDataRect::mouse_on(int event,
                              int x,
                              int y)
{
  mouse_on_common(event, x, y);

  if(clicked)
  {
    if(P1.x < 0) P1.x = 0;
    if(P2.x < 0) P2.x = 0;
    if(P1.x > img -> cols) P1.x = img -> cols;
    if(P2.x > img -> cols) P2.x = img -> cols;
    if(P1.y < 0) P1.y = 0;
    if(P2.y < 0) P2.y = 0;
    if(P1.y > img -> rows) P1.y = img -> rows;
    if(P2.y > img -> rows) P2.y = img -> rows;

    roi.x = std::min(P1.x, P2.x);
    roi.y = std::min(P1.y, P2.y);
    roi.width = std::abs(P1.x - P2.x);
    roi.height = std::abs(P1.y - P2.y);
  }

  cv::Mat img_cp = img -> clone();
  cv::rectangle(img_cp, roi, contour_color);
  cv::imshow(WINDOW_TITLE, img_cp);
}
