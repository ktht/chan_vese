#if defined(__gnu_linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <boost/filesystem/convenience.hpp> // boost::filesystem::change_extension()

#if defined(__gnu_linux__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "VideoWriterManager.hpp" // VideoWriterManager, cv::VideoWriter, cv::Size, cv::Point
                                  // cv::Rect, CV_FOURCC, cv::Vec4i, cv::putText(),
                                  // cv::drawContours(), cv::findContours(), cv::getTextSize(),
                                  // CV_8UC1, CV_AA, CV_FONT_HERSHEY_PLAIN, cv::THRESH_BINARY,
                                  // CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, ChanVese::TextPosition::,
                                  // ChanVese::Colors::, FontParameters

VideoWriterManager::VideoWriterManager(const std::string & input_filename,
                                       const cv::Mat & _img,
                                       const cv::Scalar & _contour_color,
                                       double fps,
                                       ChanVese::TextPosition _pos,
                                       bool _enable_overlay)
  : img(_img)
  , contour_color(_contour_color)
  , font({CV_FONT_HERSHEY_PLAIN, 0.8, 1, 0, CV_AA})
  , pos(_pos)
  , enable_overlay(_enable_overlay)
{
  const std::string video_filename = boost::filesystem::change_extension(input_filename, "avi").string();
  vw = cv::VideoWriter(video_filename, CV_FOURCC('X','V','I','D'), fps, img.size());
}

void
VideoWriterManager::write_frame(const cv::Mat & u,
                                const std::string & overlay_text)
{
  cv::Mat nw_img = img.clone();
  draw_contour(nw_img, u);
  if(enable_overlay)
  {
    cv::Scalar color;
    cv::Point p;
    overlay_color(overlay_text, color, p);
    cv::putText(nw_img, overlay_text, p, font.face, font.scale, color, font.thickness, font.type);
  }
  vw.write(nw_img);
}


int
VideoWriterManager::draw_contour(cv::Mat & dst,
                                 const cv::Mat & u)
{
  cv::Mat mask(img.size(), CV_8UC1);
  std::vector<std::vector<cv::Point>> cs;
  std::vector<cv::Vec4i> hier;

  cv::Mat u_cp(u.size(), CV_8UC1);
  u.convertTo(u_cp, u_cp.type());
  cv::threshold(u_cp, mask, 0, 1, cv::THRESH_BINARY);
  cv::findContours(mask, cs, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

  int idx = 0;
  for(; idx >= 0; idx = hier[idx][0])
    cv::drawContours(dst, cs, idx, contour_color, 1, 8, hier);

  return 0;
}

int
VideoWriterManager::overlay_color(const std::string txt,
                                  cv::Scalar & color,
                                  cv::Point & p)
{

  const int threshold = 105; // bias towards black font

  const cv::Size txt_sz = cv::getTextSize(txt, font.face, font.scale, font.thickness, &font.baseline);
  const int padding = 5;
  cv::Point q;

  if(pos == ChanVese::TextPosition::TopLeft)
  {
    p = cv::Point(padding, padding + txt_sz.height);
    q = cv::Point(padding, padding);
  }
  else if(pos == ChanVese::TextPosition::TopRight)
  {
    p = cv::Point(img.cols - padding - txt_sz.width, padding + txt_sz.height);
    q = cv::Point(img.cols - padding - txt_sz.width, padding);
  }
  else if(pos == ChanVese::TextPosition::BottomLeft)
  {
    p = cv::Point(padding, img.rows - padding);
    q = cv::Point(padding, img.rows - padding - txt_sz.height);
  }
  else if(pos == ChanVese::TextPosition::BottomRight)
  {
    p = cv::Point(img.cols - padding - txt_sz.width, img.rows - padding);
    q = cv::Point(img.cols - padding - txt_sz.width, img.rows - padding - txt_sz.height);
  }

  cv::Scalar avgs = cv::mean(img(cv::Rect(q, txt_sz)));
  const double intensity_avg = 0.114*avgs[0] + 0.587*avgs[1] + 0.299*avgs[2];
  color = 255 - intensity_avg < threshold ? ChanVese::Colors::black : ChanVese::Colors::white;
  return 0;
}
