#ifndef CHANVESECOMMON_HPP
#define CHANVESECOMMON_HPP

#include <opencv2/imgproc/imgproc.hpp> // cv::Mat

/** @file */

#define WINDOW_TITLE "Select contour" ///< Title of the window at startup

typedef unsigned char uchar; ///< Short for unsigned char
typedef unsigned long ulong; ///< Short for unsigned long int

namespace ChanVese
{
  /**
   * @brief The Region enum
   * @sa region_variance
   */
  enum Region { Inside, Outside };

  /**
   * @brief Enum for specifying overlay text in the image
   * @sa overlay_color
   */
  enum TextPosition { TopLeft, TopRight, BottomLeft, BottomRight };

  /**
   * @brief The Colors struct
   */
  struct Colors
  {
    static const cv::Scalar white;   ///< White
    static const cv::Scalar black;   ///< Black
    static const cv::Scalar red;     ///< Red
    static const cv::Scalar green;   ///< Green
    static const cv::Scalar blue;    ///< Blue
    static const cv::Scalar magenta; ///< Magenta
    static const cv::Scalar yellow;  ///< Yellow
    static const cv::Scalar cyan;    ///< Cyan
  };

  /**
   * @brief Finite difference kernels
   * @sa curvature
   */
  struct Kernel
  {
    static const cv::Mat fwd_x; ///< Forward difference in the x direction, @f$\Delta_x^+=f_{i+1,j}-f_{i,j}@f$
    static const cv::Mat fwd_y; ///< Forward difference in the y direction, @f$\Delta_y^+=f_{i,j+1}-f_{i,j}@f$
    static const cv::Mat bwd_x; ///< Backward difference in the x direction, @f$\Delta_x^-=f_{i,j}-f_{i-1,j}@f$
    static const cv::Mat bwd_y; ///< Backward difference in the y direction, @f$\Delta_y^-=f_{i,j}-f_{i,j-1}@f$
    static const cv::Mat ctr_x; ///< Central difference in the x direction, @f$\Delta_x^0=\frac{f_{i+1,j}-f_{i-1,j}}{2}@f$
    static const cv::Mat ctr_y; ///< Central difference in the y direction, @f$\Delta_y^0=\frac{f_{i,j+1}-f_{i,j-1}}{2}@f$
  };
}

#endif // CHANVESECOMMON_HPP

