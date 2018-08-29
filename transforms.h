#include <cmath>

inline float quadratic_hat(float t, float offset)
{
  t = t / 30 + offset;
  float y = std::max(0, 1 - abs(t / 2));
  return y * y;
}

inline float sinc(float t, float offset)
{
  t = M_PI * (t / 30 + offset);
  if(t == 0)
    return 1;
  else
    return sin(t) / t;
}
