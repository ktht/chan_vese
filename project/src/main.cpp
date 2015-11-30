#include <iostream> // std::cout, std::cerr
#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE
#include <vector> // std::vector<>
#include <algorithm> // std::min(), std::max()
#include <cmath> // std::pow(), std::sqrt(), std::sin(), std::atan()
#include <exception> // std::exception
#include <string> // std::string, std::to_string()
#include <functional> // std::function<>, std::bind(), std::placeholders::_1
#include <limits> // std::numeric_limits<>
#include <map> // std::map<>

#include <boost/math/special_functions/sign.hpp> // boost::math::sign()
#include <boost/algorithm/string/predicate.hpp> // boost::iequals()
#include <boost/algorithm/string/join.hpp> // boost::algorithm::join()

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
#include <boost/filesystem/convenience.hpp> // boost::filesystem::change_extension()

#if defined(__gnu_linux__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_WIN32)
#include <windows.h> // CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE
#include <curses.h>
#elif defined(__unix__)
#include <sys/ioctl.h> // struct winsize, ioctl(), TIOCGWINSZ
#endif

#include "ChanVeseCommon.hpp"        // ChanVese::
#include "InteractiveDataCirc.hpp"   // InteractiveDataCirc
#include "InteractiveDataRect.hpp"   // InteractiveDataRect
#include "VideoWriterManager.hpp"    // VideoWriterManager
#include "ParallelPixelFunction.hpp" // ParallelPixelFunction

                                     // Everything above comes with cv::

/**
 * @file
 * @todo
 *       - add level set reinitialization
 * @mainpage
 * @section intro_sec Introduction
 * This is the implementation of Perona-Malik + Chan-Sandberg-Vese segmentation algorithm in C++.
 * The premise of this code is that CSV segmentation is very sensitive to noise.
 * In order to get rid of the noise, we use Perona-Malik (which is optional) to smooth noisy
 * regions in the image. PM is optimal, because it preserves edges (unlike ordinary Gaussian blur).
 * The contour is calculated
 *
 * The article @cite Getreuer2012 is taken as a starting point in implementing CSV segmentation.
 * However, the text was short of describing vector-valued, i.e. multi-channel (RGB) images.
 * Fortunately, the original paper @cite Chan2000 proved to be useful.
 * PM segmentation is entirely based on the seminal paper @cite Perona1990.
 *
 * The code works for both grayscale and RGB images (any popular format is supported).
 * It mostly relies on OpenCV routines @cite Bradski2000. Some parallelization is done
 * across the channels with OpenMP preprocessors; one method is parallelized with TBB via
 * OpenCV interface.
 */

const cv::Scalar ChanVese::Colors::white   = CV_RGB(255, 255, 255);
const cv::Scalar ChanVese::Colors::black   = CV_RGB(  0,   0,   0);
const cv::Scalar ChanVese::Colors::red     = CV_RGB(255,   0,   0);
const cv::Scalar ChanVese::Colors::green   = CV_RGB(  0, 255,   0);
const cv::Scalar ChanVese::Colors::blue    = CV_RGB(  0,   0, 255);
const cv::Scalar ChanVese::Colors::magenta = CV_RGB(255,   0, 255);
const cv::Scalar ChanVese::Colors::yellow  = CV_RGB(255, 255,   0);
const cv::Scalar ChanVese::Colors::cyan    = CV_RGB(0,   255, 255);

const cv::Mat ChanVese::Kernel::fwd_x = (cv::Mat_<double>(1, 3) << 0,-1,1);
const cv::Mat ChanVese::Kernel::fwd_y = (cv::Mat_<double>(3, 1) << 0,-1,1);
const cv::Mat ChanVese::Kernel::bwd_x = (cv::Mat_<double>(1, 3) << -1,1,0);
const cv::Mat ChanVese::Kernel::bwd_y = (cv::Mat_<double>(3, 1) << -1,1,0);
const cv::Mat ChanVese::Kernel::ctr_x = (cv::Mat_<double>(1, 3) << -0.5,0,0.5);
const cv::Mat ChanVese::Kernel::ctr_y = (cv::Mat_<double>(3, 1) << -0.5,0,0.5);

/**
 * @brief Calculates the terminal/console width.
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
 * @brief Adds suffix to the file name
 * @param path   Path to the file
 * @param suffix Suffix
 * @param delim  String separating the original base name and the suffix
 * @return New file name with the suffix
 */
std::string
add_suffix(const std::string & path,
           const std::string & suffix,
           const std::string & delim = "_")
{
  namespace fs = boost::filesystem;
  const fs::path p(path);
  const fs::path nw_p = p.parent_path() / fs::path(p.stem().string() + delim + suffix + p.extension().string());
  return nw_p.string();
}

/**
 * @brief Displays error message surrounded by newlines and exits.
 * @param msg Message to display.
*/
[[ noreturn ]] void
msg_exit(const std::string & msg)
{
  std::cerr << "\n" << msg << "\n\n";
  std::exit(EXIT_FAILURE);
}

/**
 * @brief Regularized (smoothed) Heaviside step function
 * @f[ H_\epsilon(x)=\frac{1}{2}\Big[1+\frac{2}{\pi}\arctan\Big(\frac{x}{\epsilon}\Big)\Big] @f]
 * where @f$x@f$ is the argument and @f$\epsilon@f$ the smoothing parameter
 * @param x   Argument of the step function, @f$x@f$
 * @param eps Smoothing parameter, @f$\epsilon@f$
 * @return Value of the step function at @f$x@f$
 */
double
regularized_heaviside(double x,
                      double eps = 1)
{
  const double pi = boost::math::constants::pi<double>();
  return (1 + 2 / pi * std::atan(x / eps)) / 2;
}

/**
 * @brief Regularized (smoothed) Dirac delta function
 * @f[ \delta_\epsilon(x)=\frac{\epsilon}{\pi(\epsilon^2+x^2)}\,, @f]
 * where @f$x@f$ is the argument and @f$\epsilon@f$ the smoothing parameter
 * @param x   Argument of the delta function, @f$x@f$
 * @param eps Smoothing parameter, @f$\epsilon@f$
 * @return Value of the delta function at @f$x@f$
 */
constexpr double
regularized_delta(double x,
                  double eps = 1)
{
  const double pi = boost::math::constants::pi<double>();
  return eps / (pi * (std::pow(eps, 2) + std::pow(x, 2)));
}

/**
 * @brief Creates a level set with a checkerboard pattern at zero level
 *        The zero level set is found via the formula
 *        @f[ \mathrm{sign}\Big[\sin\Big(\frac{x}{5}\Big)\sin\Big(\frac{y}{5}\Big)\Big]\,, @f]
 *        where @f$x@f$ and @f$y@f$ are the positions in the image
 * @param w Width of the level set matrix
 * @param h Height of the level set matrix
 * @return The levelset
 */
cv::Mat
levelset_checkerboard(int h,
                      int w)
{
  cv::Mat u(h, w, CV_64FC1);
  const double pi = boost::math::constants::pi<double>();
  double * const u_ptr = reinterpret_cast<double *>(u.data);
  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
      u_ptr[i * w + j] = (boost::math::sign(std::sin(pi * i / 5) *
                                            std::sin(pi * j / 5)));
  return u;
}

/**
 * @brief Calculates average regional variance
 * @f[ c_i = \frac{\int_\Omega I_i(x,y)g(u(x,y))\mathrm{d}x\mathrm{d}y}{
                   \int_\Omega g(u(x,y))\mathrm{d}x\mathrm{d}y}\,, @f]
 * where @f$u(x,y)@f$ is the level set function,
 * @f$I_i@f$ is the @f$i@f$-th channel in the image and
 * @f$g@f$ is either the Heaviside function @f$H(x)@f$
 * (for region encolosed by the contour) or @f$1-H(x)@f$ (for region outside
 * the contour).
 * @param img       Input image (channel), @f$I_i(x,y)@f$
 * @param u         Level set, @f$u(x,y)@f$
 * @param h         Height of the image
 * @param w         Width of the image
 * @param region    Region either inside or outside the contour
 * @param heaviside Heaviside function, @f$H(x)@f$
 *                  One might also try different regularized heaviside functions
 *                  or even a non-smoothed one; that's why we've left it as a parameter
 * @return          Average variance of the given region in the image
 * @sa variance_penalty, Region
 */
double
region_variance(const cv::Mat & img,
                const cv::Mat & u,
                int h,
                int w,
                ChanVese::Region region,
                std::function<double(double)> heaviside)
{
  double nom = 0.0,
         denom = 0.0;
  const auto H = (region == ChanVese::Region::Inside)
                  ? heaviside
                  : [&heaviside](double x) -> double { return 1 - heaviside(x); };

  const double * const u_ptr = reinterpret_cast<double *>(u.data);
  const uchar * const img_ptr = img.data;

  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
    {
      const double h = H(u_ptr[i * w + j]);
      nom += img_ptr[i * w + j] * h;
      denom += h;
    }

  return nom / denom;
}

/**
 * @brief Calculates variance penalty matrix,
 * @f[ \lambda_i\int_\Omega|I_i(x,y)-c_i|^2 g(u(x,y))\,\mathrm{d}x\mathrm{d}y\,, @f]
 * where @f$u(x,y)@f$ is the level set function,
 * @f$I_i@f$ is the @f$i@f$-th channel in the image and
 * @f$g@f$ is either the Heaviside function @f$H(x)@f$
 * (for region encolosed by the contour) or @f$1-H(x)@f$ (for region outside
 * the contour).
 * @param channel Channel of the input image, @f$I_i(x,y)@f$
 * @param h       Height of the image
 * @param w       Width of the image
 * @param c       Variance of particular region in the image, @f$c_i@f$
 * @param lambda  Penalty parameter, @f$\lambda_i@f$
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

/**
 * @brief Calculates the curvature (divergence of normalized gradient)
 *        of the level set:
 *        @f[
 *        \kappa=
 * \Delta_x^-\left(\frac{\Delta_x^+u_{i,j}}
 * {\sqrt{\eta^2+(\Delta_x^+u_{i,j})^2+(\Delta_y^0u_{i,j})^2}}\right)+
 * \Delta_y^-\left(\frac{\Delta_y^+u_{i,j}}
 * {\sqrt{\eta^2+(\Delta_x^0u_{i,j})^2+(\Delta_y^+u_{i,j})^2}}\right)\,,
 *        @f]
 * where
 *   - @f$ \Delta_x^{\pm} @f$ and @f$ \Delta_y^{\pm} @f$ correspond to forward (@f$+@f$)
 *     and backward (@f$-@f$) difference in @f$x@f$ and @f$y@f$ direction, respectively
 *   - @f$\Delta_x^0@f$ and @f$\Delta_y^0@f$ correspond to central differences in
 *     @f$x@f$ and @f$y@f$ direction, respectively
 *   - @f$\eta@f$ is a small parameter to avoid division by zero
 *   - @f$u_{i,j}@f$ is the level set for @f$m\times n@f$ image
 * The curvature is calculated by convoluting forward, backward and central difference
 * kernels with the level set. The method assumes duplicating the pixels near the border:
 * @f[ u_{-1,j}=u_{0,j}\,,\quad u_{m,j}=u_{m-1,j}\,,\quad
 *     u_{i,-1}=u_{i,0}\,,\quad u_{i,n}=u_{n-1,j}\,. @f]
 * This method ensures that the curvature is centered at a given point and only one
 * extra pixel is needed per calculation.
 * @param u       The level set, @f$u_{i,j}@f$
 * @param h       Height of the level set matrix
 * @param w       Width of the level set matrix
 * @return Curvature
 */
cv::Mat
curvature(const cv::Mat & u,
          int h,
          int w)
{
  const double eta = 1E-8;
  const double eta2 = std::pow(eta, 2);
  cv::Mat upx (h, w, CV_64FC1), upy (h, w, CV_64FC1),
          ucx (h, w, CV_64FC1), ucy (h, w, CV_64FC1);
  cv::filter2D(u, upx, CV_64FC1, ChanVese::Kernel::fwd_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(u, upy, CV_64FC1, ChanVese::Kernel::fwd_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(u, ucx, CV_64FC1, ChanVese::Kernel::ctr_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(u, ucy, CV_64FC1, ChanVese::Kernel::ctr_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

  double * const upx_ptr = reinterpret_cast<double *>(upx.data);
  double * const upy_ptr = reinterpret_cast<double *>(upy.data);
  const double * const ucx_ptr = reinterpret_cast<double *>(ucx.data);
  const double * const ucy_ptr = reinterpret_cast<double *>(ucy.data);

#pragma omp parallel for num_threads(NUM_THREADS)
  for(int i = 0; i < h; ++i)
    for(int j = 0; j < w; ++j)
      {
        upx_ptr[i * w + j] = upx_ptr[i * w + j] /
                             std::sqrt(std::pow(upx_ptr[i * w + j], 2) + std::pow(ucx_ptr[i * w + j], 2) + eta2);
        upy_ptr[i * w + j] = upy_ptr[i * w + j] /
                             std::sqrt(std::pow(upy_ptr[i * w + j], 2) + std::pow(ucy_ptr[i * w + j], 2) + eta2);
      }

  cv::filter2D(upx, upx, CV_64FC1, ChanVese::Kernel::bwd_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(upy, upy, CV_64FC1, ChanVese::Kernel::bwd_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  upx += upy;
  return upx;
}

/**
 * @brief Separates the region enclosed by the contour in the image
 * @param img    Original image
 * @param u      Level set (the zero level of which gives us the region)
 * @param h      Height of the image
 * @param w      Width of the image
 * @param invert Invert the selected region
 * @return Image with a white background and the selected object(s) in the foreground
 */
cv::Mat
separate(const cv::Mat & img,
         const cv::Mat & u,
         int h,
         int w,
         bool invert = false)
{
  cv::Mat selection(h, w, CV_8UC3);
  cv::Mat mask(h, w, CV_8U);
  cv::Mat u_cp(h, w, CV_32F); // for some reason cv::threshold() works only with 32-bit floats

  u.convertTo(u_cp, u_cp.type());
  cv::threshold(u_cp, mask, 0, 1, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8U);
  if(invert) mask = 1 - mask;

  selection.setTo(cv::Scalar(255, 255, 255));
  img.copyTo(selection, mask);
  return selection;
}

/**
 * @section perona_malik Perona-Malik segmentation
 *
 * The method implements Perona-Malik segmentation for a multichannel image
 * (the smoothing is applied to each channel separately, hence easily parallelizable).
 * The idea is to evolve the image @f$I=I(t;\,x,\,y)@f$ according to
 *   @f[ I_{t} = \nabla c\cdot\nabla I+c\Delta I\,, @f]
 * where @f$c=c(t;\,x,\,y)@f$ is the diffusion coefficient function.  For the choice of @f$c@f$
 * we picked
 *   @f[ c(||\nabla I||) = \frac{1}{1+\left(\frac{||\nabla I||}{K}\right)^{2}}\,. @f]
 * Note that there are other possibilities for @f$c@f$, e.g. an exponential function
 * dependant on the magnitude of normalized image gradient, @f$||\nabla I||@f$. In general,
 * any function @f$f(x)@f$ for which @f$f(0) = 1@f$ and @f$\lim_{x\to\infty}f(x)=0@f$
 * would do the trick. Parameter @f$K@f$ here regulates the sensitivity of detecting the edges:
 * the larger it is, the more the edges will be smeared.
 *
 * To make any sense of this, consider the case where @f$c@f$ is constant. This way the first
 * term in the RHS of the e.o.m is zero and the equation reduces to that of Gaussian blurring.
 * If the image contains any edges, they will be smoothed out. However, if we use a function
 * that will be zero in the region containing no edges and non-zero if there's an edge,
 * we'd smooth only the regions with no edges.
 *
 * In every time step we first calculate the magnitude of normalized image gradient by
 * convoluting the image with Sobel kernels,
 *   @f[
 *        \mathbf{G}_{x}=\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}*I\,,\quad
 *        \mathbf{G}_{y}=\begin{pmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{pmatrix}*I\,,
 *   @f]
 * and calculating its square modulus with
 * @f$||\nabla I||^2=\mathbf{G}_{x}^{2}+\mathbf{G}_{x}^{2}@f$. This step is actually
 * combined with the calculation of @f$c(||\nabla I||)@f$. At the image border we keep
 * the function constant, though: @f$\left.c\right|_{\partial I}=1@f$.
 *
 * The e.o.m is approximated by (per original paper @cite Perona1990):
 *   @f[
 *       I_{i,j}^{t+1}=I_{i,j}^{t}+L[
 *                     c_{S}\nabla_{S}I+
 *                     c_{E}\nabla_{E}I+
 *                     c_{N}\nabla_{N}I+
 *                     c_{W}\nabla_{W}I
 *                    ]_{i,j}^{t}\,, @f]
 * where
 *   - @f$L\in(0,1/4)@f$ is there for the numerical scheme to be stable (note that in the paper
 *     the authors use @f$\lambda@f$ instead of @f$L@f$); it serves as a time step as well
 *   - @f$\nabla_{k}@f$ with @f$k=\{S,\,E,\,N,\,W\}@f$ denote nearest neighbour differences:
 *       @f{eqnarray*}{ &\nabla_{S}I_{i,j}=I_{i+1,j}-I_{i,j}\,,\\
 *                      &\nabla_{E}I_{i,j}=I_{i,j+1}-I_{i,j}\,,\\
 *                      &\nabla_{N}I_{i,j}=I_{i-1,j}-I_{i,j}\,,\\
 *                      &\nabla_{W}I_{i,j}=I_{i,j-1}-I_{i,j}\,;
 *       @f}
 *   - and @f$c_{k}@f$ denote respective diffusion coefficients:
 *       @f{eqnarray*}{
 * &c_{S_{i,j}}^{t}=c(||(\nabla I)_{i-1/2,j}^{t}||)
 *                  \approx\frac{1}{4}[c(||(\nabla I)_{i+1,j}||)+c(||(\nabla I)_{i,j}||)]^{t}\,,\\
 * &c_{E_{i,j}}^{t}=c(||(\nabla I)_{i,j+1/2}^{t}||)
 *                  \approx\frac{1}{4}[c(||(\nabla I)_{i,j+1}||)+c(||(\nabla I)_{i,j}||)]^{t}\,,\\
 * &c_{N_{i,j}}^{t}=c(||(\nabla I)_{i+1/2,j}^{t}||)
 *                  \approx\frac{1}{4}[c(||(\nabla I)_{i-1,j}||)+c(||(\nabla I)_{i,j}||)]^{t}\,,\\
 * &c_{W_{i,j}}^{t}=c(||(\nabla I)_{i,j-1/2}^{t}||)
 *                  \approx\frac{1}{4}[c(||(\nabla I)_{i,j-1}||)+c(||(\nabla I)_{i,j}||)]^{t}\,.
 *       @f}
 * The iteration process lasts until specified time @f$T@f$.
 *
 * @param channels The original image split into channels.
 * @param h        Height of the image
 * @param w        Width of the image
 * @param K        Edge detection parameter @f$K@f$ (same for all channels).
 * @param L        Laplacian parameter @f$L@f$ (same for all channels).
 * @param T        Time scale @f$T@f$ (same for all channels).
 * @return Smoothed image @f$I(T)@f$.
 */
cv::Mat
perona_malik(const std::vector<cv::Mat> & channels,
             int h,
             int w,
             double K,
             double L,
             double T)
{
  const int nof_channels = channels.size();
  std::vector<cv::Mat> smoothed_channels(nof_channels);

#pragma omp parallel for num_threads(nof_channels)
  for(int k = 0; k < nof_channels; ++k)
  {
    cv::Mat I_prev(h, w, CV_64FC1), // image at previous time step
            I_curr(h, w, CV_64FC1), // image at current time step
            I_res(h, w, CV_8UC1);   // the resulting image
    channels[k].copyTo(I_prev);
    I_prev.convertTo(I_prev, CV_64FC1);

    for (double t = 0; t < T; t += L)
    {
      cv::Mat g(h, w, CV_64FC1),
              dx(h, w, CV_64FC1),
              dy(h, w, CV_64FC1);
      cv::Sobel(I_prev, dx, CV_64FC1, 1, 0, 3);
      cv::Sobel(I_prev, dy, CV_64FC1, 0, 1, 3);
      I_curr = cv::Mat::zeros(h, w, CV_64F);

      const double * const I_prev_ptr = reinterpret_cast<double *>(I_prev.data);
      const double * const dx_ptr = reinterpret_cast<double *>(dx.data);
      const double * const dy_ptr = reinterpret_cast<double *>(dy.data);
      double * const I_curr_ptr = reinterpret_cast<double *>(I_curr.data);
      double * const g_ptr = reinterpret_cast<double *>(g.data);

      for(int i = 0; i < h; ++i)
        for(int j = 0; j < w; ++j)
        {
            const double gx = dx_ptr[i * w + j];
            const double gy = dy_ptr[i * w + j];
            const double d = i == 0 || i == h - 1 || j == 0 || j == w - 1 ?
                             1 :
                             std::pow(1.0 + (std::pow(gx, 2) + std::pow(gy, 2)) / (std::pow(K, 2)), -1);
            g_ptr[i * w + j] = d;
       }

      for(int i = 0; i < h; ++i)
        for(int j = 0; j < w; ++j)
        {
          const int in = i == h - 1 ? i : i + 1;
          const int ip = i == 0     ? i : i - 1;
          const int jn = j == w - 1 ? j : j + 1;
          const int jp = j == 0     ? j : j - 1;

          const double Is = I_prev_ptr[in * w + j ];
          const double Ie = I_prev_ptr[i  * w + jn];
          const double In = I_prev_ptr[ip * w + j ];
          const double Iw = I_prev_ptr[i  * w + jp];
          const double I0 = I_prev_ptr[i  * w + j ];

          const double cs = g_ptr[in * w + j ];
          const double ce = g_ptr[i  * w + jn];
          const double cn = g_ptr[ip * w + j ];
          const double cw = g_ptr[i  * w + jp];
          const double c0 = g_ptr[i  * w + j ];

          I_curr_ptr[i * w + j] = I0 + L * ((cs + c0) * (Is - I0) +
                                            (ce + c0) * (Ie - I0) +
                                            (cn + c0) * (In - I0) +
                                            (cw + c0) * (Iw - I0) ) / 4;
        }

      I_curr.copyTo(I_prev);
      I_prev.convertTo(I_res, CV_8UC1);
    }

    smoothed_channels[k] = I_res;
  }
  cv::Mat smoothed_img;
  cv::merge(smoothed_channels, smoothed_img);

  return smoothed_img;
}

/**
 * @brief Callback function for drawing contour on the image
 *        Calls InteractiveData virtual function mouse_on, which is implemented
 *        in its subclasses InteractiveDataRect (rectangular contour) and
 *        InteractiveDataCirc (circular contour)
 * @param event Event number
 * @param x     x-coordinate of the mouse in the window
 * @param y     y-coordinate of the mouse in the window
 * @param id    Additional data, which will be converted into InteractiveData pointer
 */
void
on_mouse(int event,
         int x,
         int y,
         int,
         void * id)
{
  InteractiveData * ptr = static_cast<InteractiveData *>(id);
  ptr -> mouse_on(event, x, y);
}

int
main(int argc,
     char ** argv)
{
/// @section csv_segmentation Chan-Sandberg-Vese segmentation
///
/// @subsection csv_theory Theory
/// Since the routine contains too many free parameters which makes it unreasonable to place it into a separate
/// function, all the code is kept in main(). Here's a rough explanation of what's Chan-Sandberg-Vese all about,
/// which is based on paper @cite Getreuer2012.
///
/// The Chan-Vese method seeks a contour @f$\mathcal{C}@f$ which minimizes the functional
/// @f[
///    \mathcal{F}[I;\,\mathcal{C},\,c_{1},\,c_{2}]=
///        \mu\mathrm{Length}(\mathcal{C})+
///        \nu\mathrm{Area}(\mathcal{C})+
///        \lambda_{1}\int_{\mathcal{C}}|I-c_{1}|^{2}\,\mathrm{d}x\mathrm{d}y+
///        \lambda_{2}\int_{\Omega\setminus\mathcal{C}}|I-c_{2}|^{2}\,\mathrm{d}x\mathrm{d}y\,,
/// @f]
/// where
///    - the single-channel image @f$I=I(x,\,y)@f$ is defined on the region @f$\Omega=[0,\,a]\times[0,\,b]@f$;
///         - regions in the integral limits, @f$\mathcal{C}@f$ and @f$\Omega\setminus\mathcal{C}@f$,
///           denote the region enclosed by the contour and the region outside the contour, respectively;
///    -  @f$\mu(=0.5)@f$, @f$\nu(=0)@f$, @f$\lambda_{1}(=1)@f$ and @f$\lambda_{2}(=1)@f$ are free parameters,
///       whereby only @f$\nu@f$ can be negative (default values in parentheses);
///    - @f$c_{1}@f$ and @f$c_{2}@f$ are constants that depend on the information of the regions enclosed by and
///      outside of the contour.
///
/// Instead of dealing with @f$\mathcal{C}@f$ explicitly, it's custom to define a level set function @f$u(x,\,y;\,t)@f$
/// so that its zero-level iso-surface (also: zero level set) coincides with the contour:
/// @f$\mathcal{C}=\{\Omega\ni(x,\,y)\,:\,u(x,\,y;\,t)=0\forall t\}@f$. This in turn leads us to a new definition
/// of the functional:
/// @f[
///      \mathcal{F}[I;\,u,\,c_{1},\,c_{2}] =
///             \mu\left(\int_{\Omega}|\nabla H(u)|\,\mathrm{d}x\mathrm{d}y\right)^{p}+
///             \nu\int_{\Omega}H(u)\,\mathrm{d}x\mathrm{d}y+
///             \lambda_{1}\int_{\Omega}|I-c_{1}|^{2}H(u)\,\mathrm{d}x\mathrm{d}y+
///             \lambda_{2}\int_{\Omega}|I-c_{2}|^{2}(1-H(u))\,\mathrm{d}x\mathrm{d}y\,.
/// @f]
/// In our implementation we've picked @f$p=1@f$, so that the first integral reduces to
/// @f[
///      \left.\mu\left(\int_{\Omega}|\nabla H(u)|\,\mathrm{d}x\mathrm{d}y\right)^{p}\right|_{p=1}=
///       \mu\int_{\Omega}\delta(u)|\nabla u|\,\mathrm{d}x\mathrm{d}y\,,
/// @f]
/// where @f$H(x)@f$ and @f$\delta(x)=H'(x)@f$ are Heaviside's step and Dirac's delta functions.
/// In this prescription @f$c_{1}@f$ and @f$c_{2}@f$ are now region averages and take the following form:
/// @f[
///      c_{1}=\frac{\int_{\Omega}IH(u)\mathrm{d}x\mathrm{d}y}{\int_{\Omega}H(u)\,\mathrm{d}x\mathrm{d}y}\,,\quad
///      c_{2}=\frac{\int_{\Omega}I(1-H(u))\mathrm{d}x\mathrm{d}y}{\int_{\Omega}(1 - H(u))\,\mathrm{d}x\mathrm{d}y}\,.
/// @f]
/// For practical reasons the functions are replaced by smooth/regularized versions (see regularized_heaviside() and
/// regularized_delta()):
/// @f[
///      H_{\epsilon}(x)=\frac{1}{2}\left[1+\frac{2}{\pi}\arctan\left(\frac{x}{\epsilon}\right)\right]\,,\quad
///      \delta_{\epsilon}(x)=\frac{\epsilon}{\pi\left(\epsilon^{2}+x^{2}\right)}\,,
/// @f]
/// with @f$\epsilon=1@f$ by default.
/// The interpretation of the functional @f$\mathcal{F}@f$ is the following:
///     - the first term penalizes the length of @f$\mathcal{C}@f$;
///     - the second term penalizes the area enclosed by the curve;
///     - the 3rd and 4th term penalize region averages inside and outside of the contour; in other words
///       it keeps track of the discrepancy between the two regions.
///
/// A stationary solution to @f$\mathcal{F}@f$, or equivalently the equation of motion (e.o.m) for the contour,
/// can be found by solving it with Euler-Lagrange equation, which results in
/// @f[
///    u_{t} = \delta_{\epsilon}(u)\left[\mu\kappa-\nu-\lambda_{1}(I-c_{1})^{2}+\lambda_{2}(I-c_{2})^{2}\right]\,,
/// @f]
/// where @f$\kappa=\nabla\cdot\left(\frac{\nabla u}{|\nabla u|}\right)@f$ is curvature of @f$u@f$.
///
/// If the (still 2D) image has @f$I@f$ has @f$N@f$ channels @f$\{I_{i}(x,\,y)\}_{i=1}^{N}@f$, there should still
/// be a single level set @f$u@f$, which leads us the following functional:
/// @f[
///    \mathcal{F}[I;\,u,\,\mathbf{c_{1}},\,\mathbf{c}_{2}]=
///      \mu\int_{\Omega}|\nabla H(u)|\mathrm{d}x\mathrm{d}y+
///      \nu\int_{\Omega}H(u)\mathrm{d}x\mathrm{d}y+
///      \int_{\Omega}\frac{1}{N}\sum_{i=1}^{N}\lambda_{1}^{(i)}|I_{i}-c_{1}^{(i)}|^{2}H(u)\mathrm{d}x\mathrm{d}y+
///      \int_{\Omega}\frac{1}{N}\sum_{i=1}^{N}\lambda_{2}^{(i)}|I_{i}-c_{2}^{(i)}|^{2}(1-H(u))\mathrm{d}x\mathrm{d}y\,.
/// @f]
/// Variables @f$\{c_{1}^{(i)},\,c_{2}^{(i)}\}_{i=1}^{N}@f$ retain their original meaning,
/// @f[
///     c_{1}^{(i)}=\frac{\int_{\Omega}I_{i}H(u)\mathrm{d}x\mathrm{d}y}{\int_{\Omega}H(u)\mathrm{d}x\mathrm{d}y}\,,\quad
///     c_{2}^{(i)}=\frac{\int_{\Omega}I_{i}(1-H(u))\mathrm{d}x\mathrm{d}y}{\int_{\Omega}(1-H(u))\mathrm{d}x\mathrm{d}y}
///     \quad\forall i=\{1,\,\ldots,\,N\}\,;
/// @f]
/// the constants @f$\{\lambda_{1}^{(i)},\,\lambda_{2}^{(i)}\}_{i=1}^{N}@f$ are defined for each channel separately.
/// This implementation consider only grayscale (@f$N=1@f$) and RGB (@f$N=3@f$) images, for which @f$\lambda_{i}=1@f$
/// by default for any @f$i@f$-th channel.
/// The corresponding e.o.m reads
/// @f[
///      u_{t}=\delta_{\epsilon}(u)\left[\mu\kappa-\nu-
///            \frac{1}{N}\sum_{i=1}^{N}\lambda_{1}^{(i)}\left(I_{i}-c_{1}^{(i)}\right)^{2}+
///            \frac{1}{N}\sum_{i=1}^{N}\lambda_{2}^{(i)}\left(I_{i}-c_{2}^{(i)}\right)^{2}\right]\,.
/// @f]
///
/// @subsection csv_numsch Numerical scheme
///
/// Finite difference expression for the curvature @f$\kappa@f$ is explained in curvature(). The advantage of this scheme
/// is that we only need nearest neighbouring points at current point while keeping the derivative centered at current point,
/// whereas naive implementation would use more distant points. Since we're dealing with a finite domain and therefore
/// boundaries, we don't have to "extend" the region by two pixels each direction. Instead, we just duplicate border pixels.
///
/// Rest of the calculation is actually quite straightforward -- the zero level set is iteratively updated with
/// @f[
///      u_{i,j}^{n+1}=u_{i,j}^{n}+\mathrm{d}t\;\delta_{\epsilon}(u_{i,j}^{n})\left[\kappa_{i,j}^{n}-\nu-
///      \frac{1}{N}\sum_{k=1}^N\lambda_{1}^{(k)}\left(I_{i,j}-c_{1}^{n,(k)}\right)+
///      \frac{1}{N}\sum_{k=1}^N\lambda_{2}^{(k)}\left(I_{i,j}-c_{2}^{n,(k)}\right)\right]\,.
/// @f]
/// The method is inherently implicit and is implemented with ordinary matrix operations. The first term in the brackets
/// has already been discussed; the second term is trivial; the final two terms are explained in region_variance() and
/// variance_penalty().
///
/// There are various ways to initialize the level set, and since we're solving a differential equation, different initial
/// conditions lead to different outcome. The simplest way is to let the user draw either rectangular or circular contour.
/// The level set will be evaluated with @f$+1@f$'s inside the contour and with @f$-1@f$'s outside of it.
/// A more optimal (here the default) contour would be checkerboard
/// @f[
///      u(i,\,j;\,0)=\sin\left(\frac{\pi}{5}i\right)\sin\left(\frac{\pi}{5}j\right)\,,
/// @f]
/// because it converges faster to a solution (see levelset_checkerboard()). The solution is reached when the maximum number
/// of iterations, @f$T_\max@f$, is reached or when @f$||u_{i,j}^{n+1}-u_{i,j}^{n}||_{2}\leqslant\delta ||\bar{I}||_{2}@f$,
/// where the subscript denotes @f$L_{2}@f$-norm, @f$\delta=(10^{-3})@f$ is tolerance parameter and @f$\bar{I}@f$ is
/// the intensity average in the image (averaged across the channels).
///
/// @subsection csv_summary Summary
///
/// The main logic described above starts with a timestep loop (look for the comment below); everything else preciding
/// that is actually sugar coating just to make the program usable for anyone.
///
/// If it isn't clear from above text or the code below, here is the list of variables which the user can pass as an argument
/// (the default values in the parentheses): @f$\mu(=0.5)@f$, @f$\nu(=0)@f$, @f$\mathrm{d}t(=1)@f$,
/// @f$\lambda_{1}^{(i)}(=1)@f$ and @f$\lambda_{1}^{(i)}(=1)@f$ @f$\forall i=1\ldots N@f$, @f$\epsilon(=1)@f$,
/// @f$\delta(=10^{-3})@f$, @f$T_\max@f$(=INT_MAX), @f$N(=1\;\mbox{or}\;3)@f$ (number of channels).
///
/// Other general options include:
///    - object selection (-s) -- the region enclosed by the contour will be cut out and placed onto white canvas and saved;
///    - region inversion (-I) -- sometimes the ROI is inverted; there's an option to circumvent that (goes with -s option);
///    - video output (-V) -- see contour evolution in a video (*.avi, the same filename as the image; see VideoWriterManager);
///    - overlay text (-O) -- puts overlay text (timesteps) on the video (goes with the previous option);
///    - frame rate (-f) -- frame rate of the video;
///    - line color (-l) -- color of the contour line (see Colors);
///    - rectangular (-R) or circular (-C) contour -- lets the user draw it on the image (see InteractiveData and its subclasses);
///    - grayscale image (-g) -- sometimes we just want do perform it on a black-white image, but the original source is RGB.
///
/// For Perona-Malik-specific parameters @f$K@f$, @f$L@f$, @f$T@f$, see perona_malik().
///
/// @sa curvature, region_variance, variance_penalty, levelset_checkerboard, VideoWriterManager, InteractiveData, Colors, perona_malik

  double mu, nu, eps, tol, dt, fps, K, L, T;
  int max_steps;
  std::vector<double> lambda1,
                      lambda2;
  std::string input_filename,
              text_position,
              line_color_str;
  bool grayscale         = false,
       write_video       = false,
       overlay_text      = false,
       object_selection  = false,
       invert            = false,
       segment           = false,
       rectangle_contour = false,
       circle_contour    = false;
  ChanVese::TextPosition pos = ChanVese::TextPosition::TopLeft;
  cv::Scalar contour_color = ChanVese::Colors::blue;

//-- Parse command line arguments
//   Negative values in multitoken are not an issue, b/c it doesn't make much sense
//   to use negative values for lambda1 and lambda2
  try
  {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options", get_terminal_width());
    desc.add_options()
      ("help,h",                                                                               "this message")
      ("input,i",            po::value<std::string>(&input_filename),                          "input image")
      ("mu",                 po::value<double>(&mu) -> default_value(0.5),                     "length penalty parameter (must be positive or zero)")
      ("nu",                 po::value<double>(&nu) -> default_value(0),                       "area penalty parameter")
      ("dt",                 po::value<double>(&dt) -> default_value(1),                       "timestep")
      ("lambda1",            po::value<std::vector<double>>(&lambda1) -> multitoken(),         "penalty of variance inside the contour (default: 1's)")
      ("lambda2",            po::value<std::vector<double>>(&lambda2) -> multitoken(),         "penalty of variance outside the contour (default: 1's)")
      ("epsilon,e",          po::value<double>(&eps) -> default_value(1),                      "smoothing parameter in Heaviside/delta")
      ("tolerance,t",        po::value<double>(&tol) -> default_value(0.001),                  "tolerance in stopping condition")
      ("max-steps,N",        po::value<int>(&max_steps) -> default_value(-1),                  "maximum nof iterations (negative means unlimited)")
      ("fps,f",              po::value<double>(&fps) -> default_value(10),                     "video fps")
      ("overlay-pos,P",      po::value<std::string>(&text_position) -> default_value("TL"),    "overlay tex position; allowed only: TL, BL, TR, BR")
      ("line-color,l",       po::value<std::string>(&line_color_str) -> default_value("blue"), "contour color (allowed only: black, white, R, G, B, Y, M, C")
      ("edge-coef,K",        po::value<double>(&K) -> default_value(10),                       "coefficient for enhancing edge detection in Perona-Malik")
      ("laplacian-coef,L",   po::value<double>(&L) -> default_value(0.25),                     "coefficient in the gradient FD scheme of Perona-Malik (must be [0, 1/4])")
      ("segment-time,T",     po::value<double>(&T) -> default_value(20),                       "number of smoothing steps in Perona-Malik")
      ("segment,S",          po::bool_switch(&segment),                                        "segment the image with Perona-Malik beforehand")
      ("grayscale,g",        po::bool_switch(&grayscale),                                      "read in as grayscale")
      ("video,V",            po::bool_switch(&write_video),                                    "enable video output (changes the extension to '.avi')")
      ("overlay-text,O",     po::bool_switch(&overlay_text),                                   "add overlay text")
      ("invert-selection,I", po::bool_switch(&invert),                                         "invert selected region (see: select)")
      ("select,s",           po::bool_switch(&object_selection),                               "separate the region encolosed by the contour (adds suffix '_selection')")
      ("rectangle,R",        po::bool_switch(&rectangle_contour),                              "select rectangular contour interactively")
      ("circle,C",           po::bool_switch(&circle_contour),                                 "select circular contour interactively")
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
      msg_exit("Error: you have to specify input file name!");
    else if(vm.count("input") && ! boost::filesystem::exists(input_filename))
      msg_exit("Error: file \"" + input_filename + "\" does not exists!");
    if(vm.count("dt") && dt <= 0)
      msg_exit("Cannot have negative or zero timestep: " + std::to_string(dt) + ".");
    if(vm.count("mu") && mu < 0)
      msg_exit("Length penalty parameter cannot be negative: " + std::to_string(mu) + ".");
    if(vm.count("lambda1"))
    {
      if(grayscale && lambda1.size() != 1)
        msg_exit("Too many lambda1 values for a grayscale image.");
      else if(! grayscale && lambda1.size() != 3)
        msg_exit("Number of lambda1 values must be 3 for a colored input image.");
      else if(grayscale && lambda1[0] < 0)
        msg_exit("The value of lambda1 cannot be negative.");
      else if(! grayscale && (lambda1[0] < 0 || lambda1[1] < 0 || lambda1[2] < 0))
        msg_exit("Any value of lambda1 cannot be negative.");
    }
    else if(! vm.count("lambda1"))
    {
      if(grayscale) lambda1 = {1};
      else          lambda1 = {1, 1, 1};
    }
    if(vm.count("lambda2"))
    {
      if(grayscale && lambda2.size() != 1)
        msg_exit("Too many lambda2 values for a grayscale image.");
      else if(! grayscale && lambda2.size() != 3)
        msg_exit("Number of lambda2 values must be 3 for a colored input image.");
      else if(grayscale && lambda2[0] < 0)
        msg_exit("The value of lambda2 cannot be negative.");
      else if(! grayscale && (lambda2[0] < 0 || lambda2[1] < 0 || lambda2[2] < 0))
        msg_exit("Any value of lambda2 cannot be negative.");
    }
    else if(! vm.count("lambda2"))
    {
      if(grayscale) lambda2 = {1};
      else          lambda2 = {1, 1, 1};
    }
    if(vm.count("eps") && eps < 0)
      msg_exit("Cannot have negative smoothing parameter: " + std::to_string(eps) + ".");
    if(vm.count("tol") && tol < 0)
      msg_exit("Cannot have negative tolerance: " + std::to_string(tol) + ".");
    if(vm.count("overlay-pos"))
    {
      if     (boost::iequals(text_position, "TL")) pos = ChanVese::TextPosition::TopLeft;
      else if(boost::iequals(text_position, "BL")) pos = ChanVese::TextPosition::BottomLeft;
      else if(boost::iequals(text_position, "TR")) pos = ChanVese::TextPosition::TopRight;
      else if(boost::iequals(text_position, "BR")) pos = ChanVese::TextPosition::BottomRight;
      else
        msg_exit("Invalid text position requested.\n"\
                 "Correct values are: TL -- top left\n"\
                 "                    BL -- bottom left\n"\
                 "                    TR -- top right\n"\
                 "                    BR -- bottom right"\
                );
    }
    if(vm.count("line-color"))
    {
      if     (boost::iequals(line_color_str, "red"))     contour_color = ChanVese::Colors::red;
      else if(boost::iequals(line_color_str, "green"))   contour_color = ChanVese::Colors::green;
      else if(boost::iequals(line_color_str, "blue"))    contour_color = ChanVese::Colors::blue;
      else if(boost::iequals(line_color_str, "black"))   contour_color = ChanVese::Colors::black;
      else if(boost::iequals(line_color_str, "white"))   contour_color = ChanVese::Colors::white;
      else if(boost::iequals(line_color_str, "magenta")) contour_color = ChanVese::Colors::magenta;
      else if(boost::iequals(line_color_str, "yellow"))  contour_color = ChanVese::Colors::yellow;
      else if(boost::iequals(line_color_str, "cyan"))    contour_color = ChanVese::Colors::cyan;
      else
        msg_exit("Invalid contour color requested.\n"\
                 "Correct values are: red, green, blue, black, white, magenta, yellow, cyan.");
    }
    if(vm.count("laplacian-coef") && (L > 0.25 || L < 0))
      msg_exit("The Laplacian coefficient in Perona-Malik segmentation must be between 0 and 0.25.");
    if(vm.count("segment-time") && (T < L))
      msg_exit("The segmentation duration must exceed the value of Laplacian coefficient, " +
               std::to_string(L) + ".");
    if(rectangle_contour && circle_contour)
      msg_exit("Cannot initialize with both rectangular and circular contour");
  }
  catch(std::exception & e)
  {
    msg_exit("error: " + std::string(e.what()));
  }

//-- Read the image (grayscale or BGR? RGB? BGR? help)
  cv::Mat _img;
  if(grayscale) _img = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);
  else          _img = cv::imread(input_filename, CV_LOAD_IMAGE_COLOR);
  if(! _img.data)
    msg_exit("Error on opening \"" + input_filename + "\" (probably not an image)!");

//-- Second conversion needed since we want to display a colored contour on a grayscale image
  cv::Mat img;
  if(grayscale) cv::cvtColor(_img, img, CV_GRAY2RGB);
  else          img = _img;
  _img.release();

//-- Determine the constants and define functionals
  max_steps = max_steps < 0 ? std::numeric_limits<int>::max() : max_steps;
  const int h = img.rows;
  const int w = img.cols;
  const int nof_channels = grayscale ? 1 : img.channels();
  const auto heaviside = std::bind(regularized_heaviside, std::placeholders::_1, eps);
  const auto delta = std::bind(regularized_delta, std::placeholders::_1, eps);

//-- Construct the level set
  cv::Mat u;
  if(rectangle_contour || circle_contour)
  {
    std::unique_ptr<InteractiveData> id;
    cv::startWindowThread();
    cv::namedWindow(WINDOW_TITLE, cv::WINDOW_NORMAL);

    if     (rectangle_contour)
      id = std::unique_ptr<InteractiveDataRect>(new InteractiveDataRect(&img, contour_color));
    else if(circle_contour)
      id = std::unique_ptr<InteractiveDataCirc>(new InteractiveDataCirc(&img, contour_color));

    if(id) cv::setMouseCallback(WINDOW_TITLE, on_mouse, id.get());
    cv::imshow(WINDOW_TITLE, img);
    cv::waitKey();
    cv::destroyWindow(WINDOW_TITLE);

    if(id)
    {
      if(! id -> is_ok())
        msg_exit("You must specify the contour with non-zero dimensions");
      u = id -> get_levelset(h, w);
    }
  }
  else
    u = levelset_checkerboard(h, w);

//-- Set up the video writer (and save the first frame)
  VideoWriterManager vwm;
  if(write_video)
  {
    vwm = VideoWriterManager(input_filename, img, contour_color, fps, pos, overlay_text);
    vwm.write_frame(u, overlay_text ? "t = 0" : "");
  }

//-- Split the channels
  std::vector<cv::Mat> channels;
  channels.reserve(nof_channels);
  cv::split(img, channels);
  if(grayscale) channels.erase(channels.begin() + 1, channels.end());

//-- Smooth the image with Perona-Malik
  cv::Mat smoothed_img;
  if(segment)
  {
    smoothed_img = perona_malik(channels, h, w, K, L, T);
    channels.clear();
    cv::split(smoothed_img, channels);
    cv::imwrite(add_suffix(input_filename, "pm"), smoothed_img);
  }

//-- Find intensity sum and derive the stopping condition
  cv::Mat intensity_avg = cv::Mat(h, w, CV_64FC1);
#pragma omp parallel for num_threads(nof_channels)
  for(int k = 0; k < nof_channels; ++k)
  {
    cv::Mat channel(h, w, intensity_avg.type());
    channels[k].convertTo(channel, channel.type());
    intensity_avg += channel;
  }
  intensity_avg /= nof_channels;
  double stop_cond = tol * cv::norm(intensity_avg, cv::NORM_L2);
  intensity_avg.release();

//-- Timestep loop
  for(int t = 1; t <= max_steps; ++t)
  {
    cv::Mat u_diff(cv::Mat::zeros(h, w, CV_64FC1));

//-- Channel loop
#pragma omp parallel for num_threads(nof_channels)
    for(int k = 0; k < nof_channels; ++k)
    {
      cv::Mat channel = channels[k];
//-- Find the average regional variances
      const double c1 = region_variance(channel, u, h, w, ChanVese::Region::Inside, heaviside);
      const double c2 = region_variance(channel, u, h, w, ChanVese::Region::Outside, heaviside);

//-- Calculate the contribution of one channel to the level set
      const cv::Mat variance_inside = variance_penalty(channel, h, w, c1, lambda1[k]);
      const cv::Mat variance_outside = variance_penalty(channel, h, w, c2, lambda2[k]);
      u_diff += -variance_inside + variance_outside;
    }
//-- Calculate the curvature (divergence of normalized gradient)
    const cv::Mat kappa = curvature(u, h, w);

//-- Mash the terms together
    u_diff = dt * (mu * kappa - nu + u_diff / nof_channels);

//-- Run delta function on the level set
    cv::Mat u_cp = u.clone();
    cv::parallel_for_(cv::Range(0, h * w), ParallelPixelFunction(u_cp, w, delta));

//-- Shift the level set
    cv::multiply(u_diff, u_cp, u_diff);
    const double u_diff_norm = cv::norm(u_diff, cv::NORM_L2);
    u += u_diff;

//-- Save the frame
    if(write_video) vwm.write_frame(u, overlay_text ? "t = " + std::to_string(t) : "");

//-- Check if we have achieved the desired precision
    if(u_diff_norm <= stop_cond) break;
  }

//-- Select the region enclosed by the contour and save it to the disk
  if(object_selection)
    cv::imwrite(add_suffix(input_filename, "selection"), separate(img, u, h, w, invert));

  return EXIT_SUCCESS;
}
