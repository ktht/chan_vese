#ifndef INTERACTIVEDATA_HPP
#define INTERACTIVEDATA_HPP

#include <opencv2/imgproc/imgproc.hpp> // cv::Mat, cv::Scalar, cv::Point

#include "ChanVeseCommon.hpp" // WINDOW_TITLE

/** @file */

/**
 * @brief Abstract class holding necessary information to let the user specify the
 *        initial contour.
 * @sa InteractiveDataRect, InteractiveDataCirc
 */
struct InteractiveData
{
  /**
   * @brief Simple constructor.
   * @param _img           Image onto which the contour will be drawn
   * @param _contour_color Contour color
   */
  InteractiveData(cv::Mat * const _img,
                  const cv::Scalar & _contour_color);

  virtual ~InteractiveData()
  {}

  /**
   * @brief  Checks whether the contour specified by the user is valid or not
   * @return True, if it's valid, false otherwise
   */
  virtual bool
  is_ok() const = 0;

  /**
   * @brief Returns level set based on the contour specified by the user.
   *        The region enclosed by the contour is filled with ones;
   *        the region outside the contour is filled with zeros.
   * @param h Height of the requested level set
   * @param w Width of the requested level set
   * @return The level set
   */
  virtual cv::Mat
  get_levelset(int h,
               int w) const = 0;

  /**
   * @brief Mouse callback function.
   *        In principle is responsible for displaying the original image and
   *        drawing the contour onto it
   * @param event Event number
   * @param x     x-coordinate of the mouse in the window
   * @param y     y-coordinate of the mouse in the window
   * @sa on_mouse, mouse_on_common
   */
  virtual void
  mouse_on(int event,
           int x,
           int y) = 0;

  /**
   * @brief Common logic in the callback functions implemented by the subclasses:
   *          - if the left button of the mouse is pressed, its position is recorded for both P1 and P2
   *          - if the left button is released, P2 is registered
   *          - in order to show the contour while left button is pressed and moved, we save
   *            the position only into P2
   * @param event Event number
   * @param x     x-coordinate of the mouse in the window
   * @param y     y-coordinate of the mouse in the window
   */
  void
  mouse_on_common(int event,
                  int x,
                  int y);

protected:
  cv::Mat * img = nullptr; ///< Pointer to the original image onto which the contour will be drawn
  cv::Scalar contour_color; ///< Color of the contour that will be drawn on the image

  bool clicked; ///< A boolean that keeps track whether the mouse button is pressed down or not
  cv::Point P1; ///< Coordinate of the mouse while its left button is pressed down
  cv::Point P2; ///< Coordinate of the mouse while its left button is released
};

#endif // INTERACTIVEDATA_HPP
