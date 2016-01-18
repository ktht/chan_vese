#ifndef INTERACTIVEDATARECT_HPP
#define INTERACTIVEDATARECT_HPP

#include "InteractiveData.hpp" // InteractiveData, cv::Mat, cv::Scalar, cv::Rect

/** @file */

/**
 * @brief Implements InteractiveData class; the callback function lets the user
 *        draw a rectangular contour.
 */
struct InteractiveDataRect
  : public InteractiveData
{
  /**
   * @brief Simple constructor; initializes rectangular contour
   * @param _img           Image onto which the contour will be drawn
   * @param _contour_color Contour color
   */
  InteractiveDataRect(cv::Mat * const _img,
                      const cv::Scalar & _contour_color);

  bool
  is_ok() const override;

  cv::Mat
  get_levelset(int h,
               int w) const override;

  void
  mouse_on(int event,
           int x,
           int y) override;

private:
  cv::Rect roi; ///< Rectangular contour represented by OpenCV's object
};

#endif // INTERACTIVEDATARECT_HPP

