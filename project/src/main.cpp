#include <iostream> // std::cout, std::cerr
#include <cstdlib> // EXIT_SUCCESS
#include <vector> // std::vector<>
#include <algorithm> // std::min()
#include <cmath> // std::pow(), std::sqrt(), std::sin()
#include <exception> // std::exception
#include <string> // std::string

#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor(), CV_BGR2RGB cv::threshold(),
                                       // cv::findContours(), cv::drawContours(),
                                       // cv::THRESH_BINARY, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE
#include <opencv2/highgui/highgui.hpp> // cv::imread(), CV_LOAD_IMAGE_COLOR, cv::WINDOW_NORMAL,
                                       // cv::imshow(), cv::waitKey(), cv::namedWindow()

                                       // cv::Mat, cv::Scalar, cv::Vec4i, cv::Point

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

typedef unsigned char uchar;
typedef std::vector<std::vector<double>> levelset;

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
              int w,
              int h,
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
              int w,
              int h,
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
                      int w,
                      int h)
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

int
main(int argc,
     char ** argv)
{
  /// @todo add lambda1 and lambda2 as vector arguments
  std::string input_filename;
  double mu, nu;
  try
  {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h",                                                "produce help message")
      ("input,i", po::value<std::string>(&input_filename),      "input image")
      ("mu",      po::value<double>(&mu) -> default_value(0.5), "length penalty parameter")
      ("nu",      po::value<double>(&nu) -> default_value(0),   "area penalty parameter")
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
      std::cerr << "\nError: you have to specify input file name!\n\n"
                << desc << "\n";
      return EXIT_FAILURE;
    }
    if(vm.count("input"))
    {
      if(! boost::filesystem::exists(input_filename))
      {
        std::cerr << "\nError: file \"" << input_filename << "\" does not exists!\n\n"
                  << desc << "\n";
        return EXIT_FAILURE;
      }
    }
  }
  catch(std::exception & e)
  {
    std::cerr << "error: " << e.what() << "n";
    return EXIT_FAILURE;
  }

  const cv::Mat _img = cv::imread(input_filename, CV_LOAD_IMAGE_COLOR);
  if(! _img.data)
  {
    std::cerr << "\nError on opening " << input_filename <<" "
              << "(probably not an image)!\n\n";
    return EXIT_FAILURE;
  }
  cv::Mat img;
  cv::cvtColor(_img, img, CV_BGR2RGB);
  const int h = img.rows;
  const int w = img.cols;
  levelset u;
  //levelset_rect(u, w, h, 4);
  //levelset_circ(u, w, h, 0.5);
  levelset_checkerboard(u, w, h);

  cv::Mat nw_img = img.clone();
  draw_contour(nw_img, u);

  cv::namedWindow("Display window", cv::WINDOW_NORMAL);
  cv::imshow("Display window", img);
  cv::waitKey(1000);
  std::cout << "R" << static_cast<int>(img.at<cv::Vec3b>(0, 0)[0]) << " "
               "G" << static_cast<int>(img.at<cv::Vec3b>(0, 0)[1]) << " "
               "B" << static_cast<int>(img.at<cv::Vec3b>(0, 0)[2]) << "\n";

  cv::imshow("Display window", nw_img);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
