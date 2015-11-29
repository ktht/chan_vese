#ifndef INTERACTIVEDATACIRC_HPP
#define INTERACTIVEDATACIRC_HPP

#include "InteractiveData.hpp" // InteractiveData, cv::Mat, cv::Scalar

/** @file */

/**
 * @brief Implements InteractiveData class; the callback function lets the user
 *        draw a circular contour.
 */
struct InteractiveDataCirc
  : public InteractiveData
{
  /**
   * @brief Simple constructor; initializes rectangular contour
   * @param _img           Image onto which the contour will be drawn
   * @param _contour_color Contour color
   */
  InteractiveDataCirc(cv::Mat * const _img,
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
  double radius; ///< Radius of the circular contour
};

#endif // INTERACTIVEDATACIRC_HPP

