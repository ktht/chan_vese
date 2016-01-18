#ifndef VIDEOWRITERMANAGER_HPP
#define VIDEOWRITERMANAGER_HPP

#include <string> // std::string

#include <opencv2/highgui/highgui.hpp> // cv::Mat, cv::Scalar

#include "ChanVeseCommon.hpp" // ChanVese::TextPosition::
#include "FontParameters.hpp" // FontParameters

/** @file */

/**
 * @brief A wrapper for cv::VideoWriter
 *        Holds information about underlying image, optional overlay text position,
 *        contour colors and frame rate
 */
class VideoWriterManager
{
public:
  VideoWriterManager() = default;
  /**
   * @brief VideoWriterManager constructor
   *        Needs minimum information to create a video file
   * @param input_filename  File name of the original image, the file extension of which
   *                        will be renamed to '.avi'
   * @param _img            Underlying image
   * @param _contour_color  Color of the contour
   * @param fps             Frame rate (frames per second)
   * @param _pos            Overlay text position
   * @param _enable_overlay Enables text overlay
   * @sa TextPosition
   */
  VideoWriterManager(const std::string & input_filename,
                     const cv::Mat & _img,
                     const cv::Scalar & _contour_color,
                     double fps,
                     ChanVese::TextPosition _pos,
                     bool _enable_overlay);
  /**
   * @brief Writes the frame with a given zero level set to a video file
   * @param u            Level set
   * @param overlay_text Optional overlay text
   */
  void
  write_frame(const cv::Mat & u,
              const std::string & overlay_text = "");

private:
  cv::VideoWriter vw;
  cv::Mat img;
  cv::Scalar contour_color;
  FontParameters font;
  ChanVese::TextPosition pos;
  bool enable_overlay;

  /**
   * @brief Draws the zero level set on a given image
   * @param dst        The image where the contour is placed.
   * @param u          The level set, the zero level of which is plotted.
   * @param line_color Contour line color
   * @return 0
   * @sa levelset2contour
   */
  int
  draw_contour(cv::Mat & dst,
               const cv::Mat & u);
  /**
   * @brief Finds proper font color for overlay text
   *        The color is determined by the average intensity of the ROI where
   *        the text is placed. The function also finds correct bottom left point
   *        of the text area
   * @param img    3-channel image where the text is placed
   * @param txt    The text itself
   * @param pos    Position in the image (for possibilities: top left corner,
   *               top right, bottom left or bottom right)
   * @param fparam Font parameters that help to determine the dimensions of ROI
   * @param color  Reference to the color variable
   * @param p      Reference to the bottom left point of the text area
   * @return Black color, if the background is white enough; otherwise white color
   * @todo add some check if the text width/height exceeds image dimensions
   * @sa TextPosition, FontParameters
   */
  int
  overlay_color(const std::string txt,
                cv::Scalar & color,
                cv::Point & p);
};


#endif // VIDEOWRITERMANAGER_HPP

