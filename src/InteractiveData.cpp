#include <opencv2/highgui/highgui.hpp> // cv::Mat, cv::Scalar, CV_EVENT_LBUTTONDOWN
                                       // CV_EVENT_LBUTTONUP, CV_EVENT_MOUSEMOVE

#include "InteractiveData.hpp"

InteractiveData::InteractiveData(cv::Mat * const _img,
                                 const cv::Scalar & _contour_color)
  : img(_img)
  , contour_color(_contour_color)
  , clicked(false)
  , P1(0, 0)
  , P2(0, 0)
{}

void
InteractiveData::mouse_on_common(int event,
                                 int x,
                                 int y)
{
  switch(event)
  {
    case CV_EVENT_LBUTTONDOWN:
      clicked = true;
      P1.x = x;
      P1.y = y;
      P2.x = x;
      P2.y = y;
      break;
    case CV_EVENT_LBUTTONUP:
      P2.x = x;
      P2.y = y;
      clicked = false;
      break;
    case CV_EVENT_MOUSEMOVE:
      if(clicked)
      {
        P2.x = x;
        P2.y = y;
      }
      break;
    default: break;
  }
}
