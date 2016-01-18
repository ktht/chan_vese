#ifndef PARALLELPIXELFUNCTION_HPP
#define PARALLELPIXELFUNCTION_HPP

#include <functional> // std::function<>

#include <opencv2/imgproc/imgproc.hpp> // cv::Mat, cv::Range, cv::ParallelLoopBody

/** @file */

/**
 * @brief Class for calculating function values on the level set in parallel.
 *        Specifically, meant for taking regularized delta function on the level set.
 *
 * Credit to 'maythe4thbewithu' for the idea: http://goo.gl/jPtLI2
 */
class ParallelPixelFunction :
  public cv::ParallelLoopBody
{
public:
  /**
   * @brief Constructor
   * @param _data  Level set
   * @param _w     Width of the level set matrix
   * @param _func  Any function
   */
  ParallelPixelFunction(cv::Mat & _data,
                        int _w,
                        std::function<double(double)> _func);
  /**
   * @brief Needed by cv::parallel_for_
   * @param r Range of all indices (as if the level set is flatten)
   */
  virtual void operator () (const cv::Range & r) const;

private:
  cv::Mat & data;
  const int w;
  const std::function<double(double)> func;
};

#endif // PARALLELPIXELFUNCTION_HPP

