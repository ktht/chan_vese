#include <opencv2/highgui/highgui.hpp> // cv::imshow()

#include "InteractiveDataCirc.hpp" // InteractiveData, cv::Mat, CV_64FC1, cv::Scalar,
                                   // cv::norm(), cv::circle(), cv::imshow()

InteractiveDataCirc::InteractiveDataCirc(cv::Mat * const _img,
                                         const cv::Scalar & _contour_color)
  : InteractiveData(_img, _contour_color)
  , radius(0)
{}

bool
InteractiveDataCirc::is_ok() const
{
  return radius > 0;
}

cv::Mat
InteractiveDataCirc::get_levelset(int h,
                                  int w) const
{
  cv::Mat u = cv::Mat::zeros(h, w, CV_64FC1);
  cv::circle(u, P1, radius, 1);
  return u;
}

void
InteractiveDataCirc::mouse_on(int event,
                              int x,
                              int y)
{
  mouse_on_common(event, x, y);

  if(clicked) radius = cv::norm(P1 - P2);

  cv::Mat img_cp = img -> clone();
  cv::circle(img_cp, P1, radius, contour_color);
  cv::imshow(WINDOW_TITLE, img_cp);
}
