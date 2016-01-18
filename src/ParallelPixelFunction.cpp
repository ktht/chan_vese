
#include "ParallelPixelFunction.hpp" // std::function, cv::Mat, cv::Range, cv::ParallelLoopBody

ParallelPixelFunction::ParallelPixelFunction(cv::Mat & _data,
                                             int _w,
                                             std::function<double(double)> _func)
  : data(_data)
  , w(_w)
  , func(_func)
{}

void
ParallelPixelFunction::operator () (const cv::Range & r) const
{
  for(int i = r.start; i != r.end; ++i)
    data.at<double>(i / w, i % w) = func(data.at<double>(i / w, i % w));
}
