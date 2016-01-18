#include "FontParameters.hpp"

#include <opencv2/highgui/highgui.hpp> // CV_FONT_HERSHEY_PLAIN, CV_AA

FontParameters::FontParameters()
  : face(CV_FONT_HERSHEY_PLAIN)
  , scale(0.8)
  , thickness(1)
  , type(CV_AA)
  , baseline(0)
{}
FontParameters::FontParameters(int font_face,
                               double font_scale,
                               int font_thickness,
                               int font_linetype,
                               int font_baseline)
  : face(font_face)
  , scale(font_scale)
  , thickness(font_thickness)
  , type(font_linetype)
  , baseline(font_baseline)
{}
