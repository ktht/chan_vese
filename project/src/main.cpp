#include <iostream> // std::cout, std::cerr
#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE
#include <vector> // std::vector<>
#include <algorithm> // std::min()
#include <cmath> // std::pow(), std::sqrt(), std::sin(), std::atan()
#include <exception> // std::exception
#include <string> // std::string
#include <functional> // std::function<>, std::bind(), std::placeholders::_1
#include <limits> // std::numeric_limits<>

#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor(), CV_BGR2RGB cv::threshold(),
                                       // cv::findContours(), cv::drawContours(),
                                       // cv::THRESH_BINARY, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE
#include <opencv2/highgui/highgui.hpp> // cv::imread(), CV_LOAD_IMAGE_COLOR, cv::WINDOW_NORMAL,
                                       // cv::imshow(), cv::waitKey(), cv::namedWindow()

                                       // cv::Mat, cv::Scalar, cv::Vec4i, cv::Point, cv::norm(),
                                       // cv::NORM_L2, CV_64FC1, CV_64FC1

#include <boost/math/special_functions/sign.hpp> // boost::math::sign()

#if defined(__gnu_linux__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <boost/math/constants/constants.hpp> // boost::math::constants::pi<>()
#include <boost/program_options/options_description.hpp> // boost::program_options::options_description,
                                                         // boost::program_options::value<>
#include <boost/program_options/variables_map.hpp> // boost::program_options::variables_map,
                                                   // boost::program_options::store(),
                                                   // boost::program_options::notify()
#include <boost/program_options/parsers.hpp> // boost::program_options::cmd_line::parser
#include <boost/filesystem/operations.hpp> // boost::filesystem::exists()

#if defined(__gnu_linux__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_WIN32)
#include <windows.h> // CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE
#elif defined(__unix__)
#include <sys/ioctl.h> // struct winsize, ioctl(), TIOCGWINSZ
#endif

/**
 * @file
 * @todo Switch from levelset (aka std::vector<std::vector<double>>) to cv::Mat of type CV_64FC1
 */

typedef unsigned char uchar;
typedef unsigned long ulong;
typedef std::vector<std::vector<double>> levelset;

enum Region { Inside, Outside };

/**
 * @brief Returns terminal width.
 *        Should work on all popular platforms.
 * @return Terminal width
 * @note Untested on Windows and MacOS.
 *       Credit to user 'quantum': http://stackoverflow.com/q/6812224/4056193
 */
int
get_terminal_width()
{
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
  return static_cast<int>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
#elif defined(__unix__)
  struct winsize max;
  ioctl(0, TIOCGWINSZ, &max);
  return static_cast<int>(max.ws_col);
#endif
}

/**
 * @brief Regularized (smoothed) Heaviside step function
 * @f[ $H_\epsilon(x)=\frac{1}{2}\Big(1+\frac{2}{\pi}\atan\frac{x}{\epsilon}\Big)$ @f]
 * @param x   Argument of the step function
 * @param eps Smoothing parameter
 * @return Value of the step function at x
 */
constexpr double
regularized_heaviside(double x,
                      double eps = 1)
{
  const double pi = boost::math::constants::pi<double>();
  return (1 + 2 / pi * std::atan(x / eps)) / 2;
}

/**
 * @brief Regularized (smoothed) Dirac delta function
 * @f[ $\delta_\epsilon(x)=\frac{\epsilon}{\pi(\epsilon^2+x^2)}$ @f]
 * @param x   Argument of the delta function
 * @param eps Smoothing parameter
 * @return Value of the delta function at x
 */
constexpr double
regularized_delta(double x,
                  double eps = 1)
{
  const double pi = boost::math::constants::pi<double>();
  return eps / (pi * (std::pow(eps, 2) + std::pow(x, 2)));
}

/**
 * @brief Calculates average regional variance
 * @f[ $c_i = \frac{\int_\Omega I_i(x,y)g(u(x,y))\mathrm{d}x\mathrm{d}y}{
                    \int_\Omega g(u(x,y))\mathrm{d}x\mathrm{d}y}$ @f],
 * where @f[ $u(x,y)$ @f] is the level set function,
 * @f[ $I_i$ @f] is the @f[ $i$ @f]-th channel in the image and
 * @f[ $g$ @f] is either the Heaviside function @f[$H(x)$@f]
 * (for region encolosed by the contour) or @f[$1-H(x)$@f] (for region outside
 * the contour).
 * @param img       Input image (channel), @f[ $I_i(x,y)$ @f]
 * @param u         Level set, @f[ $u(x,y)$ @f]
 * @param h         Height of the image
 * @param w         Width of the image
 * @param region    Region either inside or outside the contour
 * @param heaviside Heaviside function, @f[ $H(x)$ @f]
 *                  One might also try different regularized heaviside functions
 *                  or even a non-smoothed one; that's why we've left it as a parameter
 * @return          Average variance of the given region in the image
 * @sa variance_penalty
 */
double
region_variance(const cv::Mat & img,
                const levelset & u,
                int h,
                int w,
                Region region,
                std::function<double(double)> heaviside)
{
  double nom = 0.0,
         denom = 0.0;
  auto H = (region == Region::Inside)
             ? heaviside
             : [&heaviside](double x) -> double { return 1 - heaviside(x); };
  for(int i = 0; i < h; ++i)
  {
    for(int j = 0; j < w; ++j)
    {
      double h = H(u[i][j]);
      nom += img.at<uchar>(i, j) * h;
      denom += h;
    }
  }
  return nom / denom;
}

/**
 * @brief Creates a level set with rectangular zero level set
 * @param u Empty level set (will be modified)
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @param l Offset in pixels from the underlying image borders
 * @return 0 if success
 *         1 if the given level set is not empty
 */
int
levelset_rect(levelset & u,
              int h,
              int w,
              int l)
{
  if(! u.empty()) return 1;
  u.reserve(h);

  std::vector<double> tb_row(w, -1);
  std::vector<double> m_row(w, 1);
  std::fill(m_row.begin(), m_row.begin() + l, -1);
  std::fill(m_row.rbegin(), m_row.rbegin() + l, -1);

  for(int i = 0; i < l; ++i)     u.push_back(tb_row);
  for(int i = l; i < h - l; ++i) u.push_back(m_row);
  for(int i = h - l; i < h; ++i) u.push_back(tb_row);

  return 0;
}

/**
 * @brief Creates a level set with circular zero level set
 * @param u Empty level set (will be modified)
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @param d Diameter of the circle in relative units
 *          Its value must be within (0, 1); 1 indicates that
 *          the diameter is minimum of the image dimensions
 * @return 0 if success,
 *         1 if the given level set is not empty
 *         2 if the diameter is invalid
 */
int
levelset_circ(levelset & u,
              int h,
              int w,
              double d)
{
  if(! u.empty()) return 1;
  if(d < 0 || d > 1) return 2;

  const int r = std::min(w, h) * d / 2;
  const int mid_x = w / 2;
  const int mid_y = h / 2;

  u.reserve(h);
  for(int i = 0; i < h; ++i)
  {
    std::vector<double> row(w, -1);
    for(int j = 0; j < w; ++j)
    {
      const double d = std::sqrt(std::pow(mid_x - i, 2) +
                                 std::pow(mid_y - j, 2));
      if(d < r) row[j] = 1;
    }
    u.push_back(row);
  }

  return 0;
}

/**
 * @brief Creates a level set with a checkerboard pattern at zero level
 *        The zero level set is found via the formula
 *        @f[ $ \mathrm{sign}\sin\Big(\frac{x}{5}\Big)\sin\Big(\frac{y}{5}\Big) $ @f].
 * @param u Empty level set (will be modified)
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @return 0 if success
 *         1 if the given level set is not empty
 */
int
levelset_checkerboard(levelset & u,
                      int h,
                      int w)
{
  if(! u.empty()) return 1;

  u.reserve(h);
  const double pi = boost::math::constants::pi<double>();
  for(int i = 0; i < h; ++i)
  {
    std::vector<double> row;
    row.reserve(w);
    for(int j = 0; j < w; ++j)
      row.push_back(boost::math::sign(std::sin(pi * i / 5) *
                                      std::sin(pi * j / 5)));
    u.push_back(row);
  }

  return 0;
}

/**
 * @brief Creates a contour from the level set.
 *        In the contour matrix, the negative values are replaced by 0,
 *        whereas the positive values are replaced by 255.
 *        This convention is kept in mind later on.
 * @param u Level set
 * @return Contour
 * @sa draw_contour
 */
cv::Mat
levelset2contour(const levelset & u)
{
  const int h = u.size();
  const int w = u.at(0).size();
  cv::Mat c(h, w, CV_8UC1);

  for(int i = 0; i < h; ++i)
  {
    for(int j = 0; j < w; ++j)
    {
      const double val = u.at(i).at(j);
      c.at<uchar>(i, j) = val <= 0 ? 0 : 255;
    }
  }

  return c;
}

/**
 * @brief Draws the zero level set on a given image
 * @param dst The image where the contour is placed.
 * @param u   The level set, the zero level of which is plotted.
 * @return 0
 * @sa levelset2contour
 */
int
draw_contour(cv::Mat & dst,
             const levelset & u)
{
  cv::Mat th;
  std::vector<std::vector<cv::Point>> cs;
  std::vector<cv::Vec4i> hier;

  cv::Mat c = levelset2contour(u);
  cv::threshold(c, th, 100, 255, cv::THRESH_BINARY);
  cv::findContours(th, cs, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

  int idx = 0;
  for(; idx >= 0; idx = hier[idx][0])
  {
    cv::Scalar color(255, 0, 0); // blue
    cv::drawContours(dst, cs, idx, color, 1, 8, hier);
  }

  return 0;
}

/**
 * @brief Calculates variance penalty matrix,
 * @f[ $\lambda_i\int_\Omega|I_i(x,y)-c_i|^2 g(u(x,y))\mathrm{d}x\mathrm{d}y$ @f],
 * where @f[ $u(x,y)$ @f] is the level set function,
 * @f[ $I_i$ @f] is the @f[ $i$ @f]-th channel in the image and
 * @f[ $g$ @f] is either the Heaviside function @f[$H(x)$@f]
 * (for region encolosed by the contour) or @f[$1-H(x)$@f] (for region outside
 * the contour).
 * @param channel Channel of the input image, @f[ $I_i(x,y)$ @f]
 * @param h       Height of the image
 * @param w       Width of the image
 * @param c       Variance of particular region in the image, @f[ $c_i$ @f]
 * @param lambda  Penalty parameter, @f[ $\lambda_i$ @f]
 * @return Variance penalty matrix
 * @sa region_variance
 */
cv::Mat
variance_penalty(const cv::Mat & channel,
                 int h,
                 int w,
                 double c,
                 double lambda)
{
  cv::Mat channel_term(cv::Mat::zeros(h, w, CV_64FC1));
  channel.convertTo(channel_term, channel_term.type());
  channel_term -= c;
  cv::pow(channel_term, 2, channel_term);
  channel_term *= lambda;
  return channel_term;
}

int
main(int argc,
     char ** argv)
{
  std::string input_filename;
  double mu, nu, eps, tol, dt;
  int max_steps;
  std::vector<double> lambda1, lambda2;
  bool grayscale = false;

///-- Parse command line arguments
///   Negative values in multitoken are not an issue, b/c it doesn't make much sense
///   to use negative values for lambda1 and lambda2
  try
  {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options", get_terminal_width());
    desc.add_options()
      ("help,h",                                                                "this message")
      ("input,i",     po::value<std::string>(&input_filename),                  "input image")
      ("mu",          po::value<double>(&mu) -> default_value(0.5),             "length penalty parameter")
      ("nu",          po::value<double>(&nu) -> default_value(0),               "area penalty parameter")
      ("dt",          po::value<double>(&dt) -> default_value(1),               "timestep")
      ("lambda1",     po::value<std::vector<double>>(&lambda1) -> multitoken(), "penalty of variance inside the contour (default: 1's)")
      ("lambda2",     po::value<std::vector<double>>(&lambda2) -> multitoken(), "penalty of variance outside the contour (default: 1's)")
      ("epsilon,e",   po::value<double>(&eps) -> default_value(1),              "smoothing param in Heaviside/delta")
      ("tolerance,t", po::value<double>(&tol) -> default_value(0.001),          "tolerance in stopping condition")
      ("max-steps,N", po::value<int>(&max_steps) -> default_value(-1),          "maximum nof iterations (default: unlimited)")
      ("grayscale,g", po::bool_switch(&grayscale),                              "read in as grayscale")
    ;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
      std::cout << desc << "\n";
      return EXIT_SUCCESS;
    }
    if(! vm.count("input"))
    {
      std::cerr << "\nError: you have to specify input file name!\n\n";
      return EXIT_FAILURE;
    }
    else if(vm.count("input") && ! boost::filesystem::exists(input_filename))
    {
      std::cerr << "\nError: file \"" << input_filename << "\" does not exists!\n\n";
      return EXIT_FAILURE;
    }
    if(vm.count("dt") && dt < 0)
    {
      std::cerr << "\nCannot have negative timestep: " << dt << ".\n\n";
      return EXIT_FAILURE;
    }
    if(vm.count("lambda1"))
    {
      if(grayscale && lambda1.size() != 1)
      {
        std::cerr << "\nToo many lambda1 values for a grayscale image.\n\n";
        return EXIT_FAILURE;
      }
      else if(!grayscale && lambda1.size() != 3)
      {
        std::cerr << "\nNumber of lambda1 values must be 3 for a colored input image.\n\n";
        return EXIT_FAILURE;
      }
      if(grayscale && lambda1.size() == 1 && lambda1[0] < 0)
      {
        std::cerr << "\nThe value of lambda1 cannot be negative.\n\n";
        return EXIT_FAILURE;
      }
    }
    else if(! vm.count("lambda1"))
    {
      if(grayscale) lambda1 = {1};
      else          lambda1 = {1, 1, 1};
    }
    if(vm.count("lambda2"))
    {
      if(grayscale && lambda2.size() != 1)
      {
        std::cerr << "\nToo many lambda2 values for a grayscale image.\n\n";
        return EXIT_FAILURE;
      }
      else if(!grayscale && lambda2.size() != 3)
      {
        std::cerr << "\nNumber of lambda2 values must be 3 for a colored input image.\n\n";
        return EXIT_FAILURE;
      }
      if(grayscale && lambda2.size() == 1 && lambda2[0] < 0)
      {
        std::cerr << "\nThe value of lambda1 cannot be negative.\n\n";
        return EXIT_FAILURE;
      }
    }
    else if(! vm.count("lambda2"))
    {
      if(grayscale) lambda2 = {1};
      else          lambda2 = {1, 1, 1};
    }
    if(vm.count("eps") && eps < 0)
    {
      std::cerr << "\nCannot have negative smoothing parameter: " << eps << ".\n\n";
      return EXIT_FAILURE;
    }
    if(vm.count("tol") && tol < 0)
    {
      std::cerr << "\nCannot have negative tolerance: " << tol << ".\n\n";
      return EXIT_FAILURE;
    }
    if(vm.count("max-steps") && max_steps < 0)
    {
      std::cerr << "\nNegative maximum nof iterations: " << max_steps << ". "
                << "Switching to INT_MAX instead.\n\n";
    }
  }
  catch(std::exception & e)
  {
    std::cerr << "error: " << e.what() << "n";
    return EXIT_FAILURE;
  }

///-- Read the image (grayscale or BGR? RGB? BGR? help)
  cv::Mat _img;
  if(grayscale) _img = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);
  else          _img = cv::imread(input_filename, CV_LOAD_IMAGE_COLOR);
  if(! _img.data)
  {
    std::cerr << "\nError on opening " << input_filename <<" "
              << "(probably not an image)!\n\n";
    return EXIT_FAILURE;
  }

///-- Second conversion needed since we want to display a colored contour
///   on a grayscale image
  cv::Mat img;
  if(grayscale) cv::cvtColor(_img, img, CV_GRAY2RGB);
  else          img = _img;
  _img.release();

///-- Determine the constants and define functionals
  max_steps = max_steps < 0 ? std::numeric_limits<int>::max() : max_steps;
  const int h = img.rows;
  const int w = img.cols;
  const int nof_channels = grayscale ? 1 : img.channels();
  const auto heaviside = std::bind(regularized_heaviside, std::placeholders::_1, eps);
  const auto delta = std::bind(regularized_delta, std::placeholders::_1, eps);

///-- Construct the level set
  levelset u;
  levelset_checkerboard(u, h, w);

///-- Split the channels
  std::vector<cv::Mat> channels;
  channels.reserve(nof_channels);
  cv::split(img, channels);

///-- Find intensity sum and derive the stopping condition
  cv::Mat intensity_avg = cv::Mat(h, w, CV_64FC1, cv::Scalar::all(0));
  for(int k = 0; k < nof_channels; ++k)
  {
    cv::Mat channel(h, w, intensity_avg.type());
    channels[k].convertTo(channel, channel.type());
    intensity_avg += channel;
  }
  intensity_avg /= nof_channels;
  double stop_cond = tol * cv::norm(intensity_avg, cv::NORM_L2);
  intensity_avg.release();

///-- Timestep loop
  for(int t = 0; t < max_steps; ++t)
  {
///-- Channel loop
    cv::Mat u_diff(cv::Mat::zeros(h, w, CV_64FC1));
    for(int k = 0; k < nof_channels; ++k)
    {
      cv::Mat channel = channels[k];
///-- Find the average regional variances
      const double c1 = region_variance(channel, u, h, w, Region::Inside, heaviside);
      const double c2 = region_variance(channel, u, h, w, Region::Outside, heaviside);

///-- Calculate the contribution of one channel to the level set
      cv::Mat variance_inside = variance_penalty(channel, h, w, c1, lambda1[k]);
      cv::Mat variance_outside = variance_penalty(channel, h, w, c2, lambda2[k]);
      u_diff += -variance_inside + variance_outside;
    }
    u_diff /= nof_channels;
///-- Check if we have achieved the desired precision
  }

///-- Display the zero level set
  cv::Mat nw_img = img.clone();
  draw_contour(nw_img, u);
  cv::namedWindow("Display window", cv::WINDOW_NORMAL);
  cv::imshow("Display window", img);
  cv::waitKey(1000);
  cv::imshow("Display window", nw_img);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
