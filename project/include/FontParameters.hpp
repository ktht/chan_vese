#ifndef FONTPARAMETERS_HPP
#define FONTPARAMETERS_HPP

/** @file FontParameters.hpp */

/**
 * @brief Class for holding basic parameters of a font
 */
class FontParameters
{
public:
  FontParameters();
  /**
   * @brief FontParameters constructor
   * @param font_face      Font (type)face, expecting CV_FONT_HERSHEY_*
   * @param font_scale     Font size (relative; multiplied with base font size)
   * @param font_thickness Font thickness
   * @param font_linetype  Font line type, e.g. 8 (8-connected), 4 (4-connected)
   *                       or CV_AA (antialiased font)
   * @param font_baseline  Bottom padding in y-direction?
   */
  FontParameters(int font_face,
                 double font_scale,
                 int font_thickness,
                 int font_linetype,
                 int font_baseline);

  int face;      ///< Font (type)face
  double scale;  ///< Font size (relative; multiplied with base font size)
  int thickness; ///< Font thickness
  int type;      ///< Font line type
  int baseline;  ///< Bottom padding in y-direction?
};

#endif // FONTPARAMETERS_HPP

