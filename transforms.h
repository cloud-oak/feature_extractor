#include <cmath>
using std::abs;

inline float qhat(float t, float offset)
{
  // Quadratic Hat
  t = t / 30 - offset;
  float y = std::max(0.0f, 1 - std::abs(t / 2));
  return y * y;
}

inline float sinc(float t, float offset)
{
  t = M_PI * (t / 30 - offset);
  if(t == 0)
    return 1;
  else
    return sin(t) / t;
}

inline float dx(float t, float offset)
{
  t = t / 30 - offset;
  if(t < -1 || t > 1)
    return 0;
  else
    return t;
}

inline float esmooth(float t, float offset)
{
  t = t / 30 - offset;
  return exp(-abs(t));
}

// This kernel is exp(-|x|) ⁎ {x: |x| < 1, 0: else}
// equivalently:  esmooth   ⁎ dx
// http://www.wolframalpha.com/input/?i=Convolve%5Bexp%5B-%7Cx%7C%5D,+Piecewise(%7B%7Bx,+%7Cx%7C+%3C+1%7D%7D,+0),+x,+t%5D
inline float ediff(float t, float offset)
{
  t = t / 30 - offset;
  if(t >= 1)
    return 2 * exp(-t - 1);
  else if(t <= -1)
    return -2 * exp(t - 1);
  else
    return 2 * (t + exp(-t - 1) - exp(t - 1));
}
