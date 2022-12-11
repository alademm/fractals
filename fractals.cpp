// -----------------------------------------------------------------------------
// Copyright (c) 2022 Mohamed Aladem
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// -----------------------------------------------------------------------------

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#endif // USE_TBB

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#ifdef __AVX2__
#include <immintrin.h>
#endif // __AVX2__

namespace
{

  //-----------------------------------------------
  //------------- Coloring Functions --------------
  //-----------------------------------------------
  inline cv::Vec3b HSV2BGR(int H, float S, float V)
  {
    float C = S * V;
    float X = C * (1.0f - std::abs(std::fmod(H / 60.0, 2) - 1.0));
    float m = V - C;
    float r, g, b;
    if (H >= 0 && H < 60)
    {
      r = C, g = X, b = 0;
    }
    else if (H >= 60 && H < 120)
    {
      r = X, g = C, b = 0;
    }
    else if (H >= 120 && H < 180)
    {
      r = 0, g = C, b = X;
    }
    else if (H >= 180 && H < 240)
    {
      r = 0, g = X, b = C;
    }
    else if (H >= 240 && H < 300)
    {
      r = X, g = 0, b = C;
    }
    else
    {
      r = C, g = 0, b = X;
    }
    uint8_t R = (r + m) * 255.0f;
    uint8_t G = (g + m) * 255.0f;
    uint8_t B = (b + m) * 255.0f;
    return cv::Vec3b(B, G, R);
  }

  inline cv::Vec3b ColorBW(int iterations, int maxIterations, double zr, double zi)
  {
    uint8_t c = 255 - (iterations * 255 / maxIterations);
    return cv::Vec3b(c, c, c);
  }

  inline cv::Vec3b ColorHSV(int iterations, int maxIterations, double zr, double zi)
  {
    const int H = (float(iterations) / float(maxIterations)) * 360.0f;
    const float S = 0.8f;
    const float V = (iterations == maxIterations) ? 0.0f : 1.0f;
    return HSV2BGR(H, S, V);
  }

  inline cv::Vec3b ColorSmoothHSV(int iterations, int maxIterations, double zr, double zi)
  {
    // Source: https://stackoverflow.com/questions/369438/smooth-spectrum-for-mandelbrot-set-rendering
    const double log_2 = std::log(2);
    const double m = 2.0 * std::log(zr * zr + zi * zi);
    double n = iterations;
    double nsmooth = n + 1.0 - std::log(m) / log_2;
    nsmooth /= maxIterations;
    nsmooth = 0.95 + 10.0 * nsmooth;
    int hue = 360.0 * (nsmooth - std::floor(nsmooth));
    return HSV2BGR(hue, 0.6f, 1.0f);
  }

  inline cv::Vec3b ColorCalm(int iterations, int maxIterations, double zr, double zi)
  {
    // Source: https://www.reddit.com/r/math/comments/2abwyt/smooth_colour_mandelbrot/
    if (iterations < maxIterations)
    {
      const double log_2 = std::log(2);
      const double m = 2.0 * std::log(zr * zr + zi * zi);
      double v = std::log(iterations + 1.5 - std::log(m / log_2)) / 3.4;
      if (v < 1.0)
      {
        uint8_t r = v * v * v * v * 255.0;
        uint8_t g = std::pow(v, 2.5) * 255.0;
        uint8_t b = v * 255.0;
        return cv::Vec3b(b, g, r);
      }
      else
      {
        v = std::max(0.0, 2.0 - v);
        uint8_t r = v * 255.0;
        uint8_t g = std::pow(v, 1.5) * 255.0;
        uint8_t b = v * v * v * 255.0;
        return cv::Vec3b(b, g, r);
      }
    }
    return cv::Vec3b{};
  }

  //-----------------------------------------------
  //------- Fractal Computation Functions ---------
  //-----------------------------------------------

  void ComputeSerial(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
    const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal)
  {
    const double x_scale = (fractal_br.x - fractal_tl.x) / double(image_size.x);
    const double y_scale = (fractal_br.y - fractal_tl.y) / double(image_size.y);
    for (int pixel_y = 0; pixel_y < image_size.y; ++pixel_y)
    {
      const double fractal_y0 = pixel_y * y_scale + fractal_tl.y;
      for (int pixel_x = 0; pixel_x < image_size.x; ++pixel_x)
      {
        const double fractal_x0 = pixel_x * x_scale + fractal_tl.x;
        double x = 0.0;
        double y = 0.0;
        int N = 0;
        while ((x * x + y * y) < 4.0 && N < max_iterations)
        {
          double x_temp = x * x - y * y + fractal_x0;
          y = 2.0 * x * y + fractal_y0;
          x = x_temp;
          ++N;
        }
        fractal.at<cv::Vec3d>(pixel_y, pixel_x) = cv::Vec3d(N, x, y);
      }
    }
  }

#ifdef USE_TBB
  void ComputeThreaded(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
    const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal)
  {
    const double x_scale = (fractal_br.x - fractal_tl.x) / double(image_size.x);
    const double y_scale = (fractal_br.y - fractal_tl.y) / double(image_size.y);
    tbb::parallel_for(tbb::blocked_range2d<int, int>(0, image_size.y, 0, image_size.x),
      [&](const tbb::blocked_range2d<int, int>& r)
      {
        for (int pixel_y = r.rows().begin(); pixel_y < r.rows().end(); pixel_y++)
        {
          const double fractal_y0 = pixel_y * y_scale + fractal_tl.y;
          for (int pixel_x = r.cols().begin(); pixel_x < r.cols().end(); pixel_x++)
          {
            const double fractal_x0 = pixel_x * x_scale + fractal_tl.x;
            double x = 0.0;
            double y = 0.0;
            int N = 0;
            while ((x * x + y * y) < 4.0 && N < max_iterations)
            {
              double x_temp = x * x - y * y + fractal_x0;
              y = 2.0 * x * y + fractal_y0;
              x = x_temp;
              ++N;
            }

            fractal.at<cv::Vec3d>(pixel_y, pixel_x) = cv::Vec3d(N, x, y);
          }
        }
      });
  }
#endif // USE_TBB

#ifdef __ARM_NEON
  inline int vmovemaskq_u32(uint32x4_t conditions)
  {
    // Source https://rcl-rs-vvg.blogspot.com/2010/08/simd-etudes.html
    const uint32x4_t qMask = {1, 2, 4, 8};
    const uint32x4_t qAnded = vandq_u32(conditions, qMask);
    // these two are no-ops, they only tell compiler to treat Q register as two D regs
    const uint32x2_t dHigh = vget_high_u32(qAnded);
    const uint32x2_t dLow = vget_low_u32(qAnded);
    const uint32x2_t dOred = vorr_u32(dHigh, dLow);
    const uint32x2_t dMask = vpadd_u32(dOred, dOred);
    return vget_lane_u32(dMask, 0);
  }

  void ComputeNEON(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
    const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal)
  {
    const float x_scale = (fractal_br.x - fractal_tl.x) / float(image_size.x);
    const float y_scale = (fractal_br.y - fractal_tl.y) / float(image_size.y);

    const uint32x4_t _one = vmovq_n_u32(1);
    const float32x4_t _two = vmovq_n_f32(2.0f);
    const float32x4_t _four = vmovq_n_f32(4.0f);
    const float32x4_t _x_scale = vld1q_dup_f32(&x_scale);
    float fractal_tl_x = fractal_tl.x;
    const float32x4_t _fractal_tl_x = vld1q_dup_f32(&fractal_tl_x);

    for (int pixel_y = 0; pixel_y < image_size.y; ++pixel_y)
    {
      float32x4_t _pixel_x = {0, 1, 2, 3};
      float fractal_y0 = pixel_y * y_scale + fractal_tl.y;
      float32x4_t _fractal_y0 = vld1q_dup_f32(&fractal_y0);
      for (int pixel_x = 0; pixel_x < image_size.x; pixel_x += 4)
      {
        const float32x4_t _fractal_x0 = vmlaq_f32(_fractal_tl_x, _pixel_x, _x_scale);
        float32x4_t _x = vmovq_n_f32(0.0f);
        float32x4_t _y = vmovq_n_f32(0.0f);
        int32x4_t _N = vmovq_n_s32(0);

        for (int i = 0; i < max_iterations; ++i)
        {
          float32x4_t _xx = vmulq_f32(_x, _x);
          float32x4_t _yy = vmulq_f32(_y, _y);
          float32x4_t _xy = vmulq_f32(_x, _y);
          float32x4_t _z = vaddq_f32(_xx, _yy);
          uint32x4_t _mask = vcltq_f32(_z, _four);
          if (vmovemaskq_u32(_mask) == 0)
          {
            break;
          }
          float32x4_t _x_temp = vsubq_f32(_xx, _yy);
          _x_temp = vaddq_f32(_x_temp, _fractal_x0);
          float32x4_t _y_temp = vmlaq_f32(_fractal_y0, _two, _xy);
          _x = vbslq_f32(_mask, _x_temp, _x);
          _y = vbslq_f32(_mask, _y_temp, _y);
          uint32x4_t _c = vandq_u32(_mask, _one);
          _N = vaddq_s32(vreinterpretq_s32_u32(_c), _N);
        }

        _pixel_x = vaddq_f32(_pixel_x, _four);

        {
          int iters[4];
          vst1q_s32(iters, _N);
          float xs[4];
          float ys[4];
          vst1q_f32(xs, _x);
          vst1q_f32(ys, _y);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x) = cv::Vec3d(iters[0], xs[0], ys[0]);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x + 1) = cv::Vec3d(iters[1], xs[1], ys[1]);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x + 2) = cv::Vec3d(iters[2], xs[2], ys[2]);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x + 3) = cv::Vec3d(iters[3], xs[3], ys[3]);
        }
      }
    }
  }

#ifdef USE_TBB
  void ComputeThreadedNEON(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
    const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal)
  {
    const float x_scale = (fractal_br.x - fractal_tl.x) / float(image_size.x);
    const float y_scale = (fractal_br.y - fractal_tl.y) / float(image_size.y);
    const uint32x4_t _one = vmovq_n_u32(1);
    const float32x4_t _two = vmovq_n_f32(2.0f);
    const float32x4_t _four = vmovq_n_f32(4.0f);
    const float32x4_t _x_scale = vld1q_dup_f32(&x_scale);
    float fractal_tl_x = fractal_tl.x;
    const float32x4_t _fractal_tl_x = vld1q_dup_f32(&fractal_tl_x);
    tbb::parallel_for(tbb::blocked_range<int>(0, image_size.y),
      [&](const tbb::blocked_range<int>& r)
      {
        for (int pixel_y = r.begin(); pixel_y < r.end(); pixel_y++)
        {
          float32x4_t _pixel_x = {0, 1, 2, 3};
          float fractal_y0 = pixel_y * y_scale + fractal_tl.y;
          float32x4_t _fractal_y0 = vld1q_dup_f32(&fractal_y0);
          for (int pixel_x = 0; pixel_x < image_size.x; pixel_x += 4)
          {
            const float32x4_t _fractal_x0 = vmlaq_f32(_fractal_tl_x, _pixel_x, _x_scale);
            float32x4_t _x = vmovq_n_f32(0.0f);
            float32x4_t _y = vmovq_n_f32(0.0f);
            int32x4_t _N = vmovq_n_s32(0);

            for (int i = 0; i < max_iterations; ++i)
            {
              float32x4_t _xx = vmulq_f32(_x, _x);
              float32x4_t _yy = vmulq_f32(_y, _y);
              float32x4_t _xy = vmulq_f32(_x, _y);
              float32x4_t _z = vaddq_f32(_xx, _yy);
              uint32x4_t _mask = vcltq_f32(_z, _four);
              if (vmovemaskq_u32(_mask) == 0)
              {
                break;
              }
              float32x4_t _x_temp = vsubq_f32(_xx, _yy);
              _x_temp = vaddq_f32(_x_temp, _fractal_x0);
              float32x4_t _y_temp = vmlaq_f32(_fractal_y0, _two, _xy);
              _x = vbslq_f32(_mask, _x_temp, _x);
              _y = vbslq_f32(_mask, _y_temp, _y);
              uint32x4_t _c = vandq_u32(_mask, _one);
              _N = vaddq_s32(vreinterpretq_s32_u32(_c), _N);
            }

            _pixel_x = vaddq_f32(_pixel_x, _four);

            {
              int iters[4];
              vst1q_s32(iters, _N);
              float xs[4];
              float ys[4];
              vst1q_f32(xs, _x);
              vst1q_f32(ys, _y);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x) = cv::Vec3d(iters[0], xs[0], ys[0]);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x + 1) = cv::Vec3d(iters[1], xs[1], ys[1]);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x + 2) = cv::Vec3d(iters[2], xs[2], ys[2]);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x + 3) = cv::Vec3d(iters[3], xs[3], ys[3]);
            }
          }
        }
      });
  }
#endif // USE_TBB
#endif // __ARM_NEON

#ifdef __AVX2__
  void ComputeAVX(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
    const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal)
  {
    const double x_scale = (fractal_br.x - fractal_tl.x) / double(image_size.x);
    const double y_scale = (fractal_br.y - fractal_tl.y) / double(image_size.y);

    const __m256i _one = _mm256_set1_epi64x(1);
    const __m256d _two = _mm256_set1_pd(2.0);
    const __m256d _four = _mm256_set1_pd(4.0);
    const __m256d _x_scale = _mm256_set1_pd(x_scale);
    const __m256d _fractal_tl_x = _mm256_set1_pd(fractal_tl.x);

    for (int pixel_y = 0; pixel_y < image_size.y; ++pixel_y)
    {
      __m256d _pixel_x = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);
      double fractal_y0 = pixel_y * y_scale + fractal_tl.y;
      __m256d _fractal_y0 = _mm256_set1_pd(fractal_y0);
      for (int pixel_x = 0; pixel_x < image_size.x; pixel_x += 4)
      {
        const __m256d _fractal_x0 = _mm256_fmadd_pd(_pixel_x, _x_scale, _fractal_tl_x);
        __m256d _x = _mm256_setzero_pd();
        __m256d _y = _mm256_setzero_pd();
        __m256i _N = _mm256_setzero_si256();

        for (int i = 0; i < max_iterations; ++i)
        {
          __m256d _xx = _mm256_mul_pd(_x, _x);
          __m256d _yy = _mm256_mul_pd(_y, _y);
          __m256d _xy = _mm256_mul_pd(_x, _y);
          __m256d _z = _mm256_add_pd(_xx, _yy);
          __m256d _mask = _mm256_cmp_pd(_z, _four, _CMP_LT_OQ);
          if (_mm256_movemask_pd(_mask) == 0)
          {
            break;
          }
          __m256d _x_temp = _mm256_sub_pd(_xx, _yy);
          _x_temp = _mm256_add_pd(_x_temp, _fractal_x0);
          __m256d _y_temp = _mm256_fmadd_pd(_two, _xy, _fractal_y0);
          _x = _mm256_blendv_pd(_x, _x_temp, _mask);
          _y = _mm256_blendv_pd(_y, _y_temp, _mask);
          __m256i _c = _mm256_and_si256(_mm256_castpd_si256(_mask), _one);
          _N = _mm256_add_epi64(_N, _c);
        }

        _pixel_x = _mm256_add_pd(_pixel_x, _four);

        {
          int64_t* iters = (int64_t*)&_N;
          double* xs = (double*)&_x;
          double* ys = (double*)&_y;
          fractal.at<cv::Vec3d>(pixel_y, pixel_x) = cv::Vec3d(iters[3], xs[3], ys[3]);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x + 1) = cv::Vec3d(iters[2], xs[2], ys[2]);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x + 2) = cv::Vec3d(iters[1], xs[1], ys[1]);
          fractal.at<cv::Vec3d>(pixel_y, pixel_x + 3) = cv::Vec3d(iters[0], xs[0], ys[0]);
        }
      }
    }
  }

#ifdef USE_TBB
  void ComputeThreadedAVX(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
    const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal)
  {
    const double x_scale = (fractal_br.x - fractal_tl.x) / double(image_size.x);
    const double y_scale = (fractal_br.y - fractal_tl.y) / double(image_size.y);

    const __m256i _one = _mm256_set1_epi64x(1);
    const __m256d _two = _mm256_set1_pd(2.0);
    const __m256d _four = _mm256_set1_pd(4.0);
    const __m256d _x_scale = _mm256_set1_pd(x_scale);
    const __m256d _fractal_tl_x = _mm256_set1_pd(fractal_tl.x);

    tbb::parallel_for(tbb::blocked_range<int>(0, image_size.y),
      [&](const tbb::blocked_range<int>& r)
      {
        for (int pixel_y = r.begin(); pixel_y < r.end(); pixel_y++)
        {
          __m256d _pixel_x = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);
          double fractal_y0 = pixel_y * y_scale + fractal_tl.y;
          __m256d _fractal_y0 = _mm256_set1_pd(fractal_y0);
          for (int pixel_x = 0; pixel_x < image_size.x; pixel_x += 4)
          {
            const __m256d _fractal_x0 = _mm256_fmadd_pd(_pixel_x, _x_scale, _fractal_tl_x);
            __m256d _x = _mm256_setzero_pd();
            __m256d _y = _mm256_setzero_pd();
            __m256i _N = _mm256_setzero_si256();

            for (int i = 0; i < max_iterations; ++i)
            {
              __m256d _xx = _mm256_mul_pd(_x, _x);
              __m256d _yy = _mm256_mul_pd(_y, _y);
              __m256d _xy = _mm256_mul_pd(_x, _y);
              __m256d _z = _mm256_add_pd(_xx, _yy);
              __m256d _mask = _mm256_cmp_pd(_z, _four, _CMP_LT_OQ);
              if (_mm256_movemask_pd(_mask) == 0)
              {
                break;
              }
              __m256d _x_temp = _mm256_sub_pd(_xx, _yy);
              _x_temp = _mm256_add_pd(_x_temp, _fractal_x0);
              __m256d _y_temp = _mm256_fmadd_pd(_two, _xy, _fractal_y0);
              _x = _mm256_blendv_pd(_x, _x_temp, _mask);
              _y = _mm256_blendv_pd(_y, _y_temp, _mask);
              __m256i _c = _mm256_and_si256(_mm256_castpd_si256(_mask), _one);
              _N = _mm256_add_epi64(_N, _c);
            }

            _pixel_x = _mm256_add_pd(_pixel_x, _four);

            {
              int64_t* iters = (int64_t*)&_N;
              double* xs = (double*)&_x;
              double* ys = (double*)&_y;
              fractal.at<cv::Vec3d>(pixel_y, pixel_x) = cv::Vec3d(iters[3], xs[3], ys[3]);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x + 1) = cv::Vec3d(iters[2], xs[2], ys[2]);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x + 2) = cv::Vec3d(iters[1], xs[1], ys[1]);
              fractal.at<cv::Vec3d>(pixel_y, pixel_x + 3) = cv::Vec3d(iters[0], xs[0], ys[0]);
            }
          }
        }
      });
  }
#endif // USE_TBB
#endif // __AVX2__

}

class FractalsRenderer
{
public:
  FractalsRenderer(int width, int height) : width_(width), height_(height)
  {
    scale_ = {double(width_) / 2.0, double(height_)};
    fractal_ = cv::Mat(height_, width_, CV_64FC3);
    fractalImage_ = cv::Mat(height_, width_, CV_8UC3);
    RegisterColorFunctions();
    RegisterComputeFunctions();
  }

  const cv::Mat& GetFractalImage() const { return fractalImage_; }

  void SetMaxIterations(int maxIterations) { maxIterations_ = maxIterations; }
  int GetMaxIterations() const { return maxIterations_; }

  void SetFractalComputeMethod(int compute_method_id)
  {
    if (compute_method_id > 0 && compute_method_id <= computeFunctions_.size())
    {
      activeCompute_ = compute_method_id;
    }
  }

  void CycleColor()
  {
    activeColor_ = (activeColor_ + 1) % colorFunctions_.size();
  }

  void OnMouseEvent(int event, int x, int y, int flags)
  {
    cv::Point2d mouseLoc{double(x), double(y)};

    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
    {
      startPan_ = mouseLoc;
      isPanning_ = true;
      break;
    }
    case cv::EVENT_LBUTTONUP:
    {
      isPanning_ = false;
      break;
    }
    case cv::EVENT_RBUTTONDOWN:
    {
      startZoom_ = mouseLoc;
      isZooming_ = true;
      break;
    }
    case cv::EVENT_RBUTTONUP:
    {
      isZooming_ = false;
    }
    case cv::EVENT_MOUSEMOVE:
    {
      if (isPanning_)
      {
        auto diff = (mouseLoc - startPan_);
        diff.x /= scale_.x;
        diff.y /= scale_.y;
        offset_ -= diff;
        startPan_ = mouseLoc;
      }
      else if (isZooming_)
      {
        cv::Point2d mouse_1;
        PixelToFractal(mouseLoc, mouse_1);
        double scaleFactor = (mouseLoc.y - startZoom_.y) / height_;
        scaleFactor *= 4.0;
        scale_ *= (1.0 + scaleFactor);
        cv::Point2d mouse_2;
        PixelToFractal(mouseLoc, mouse_2);
        offset_ += (mouse_1 - mouse_2);
        startZoom_ = mouseLoc;
      }
    }
    }
  }

  void Compute()
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Point2i pix_tl = {0, 0};
    cv::Point2i pix_br = {width_, height_};
    cv::Point2d frac_tl;
    cv::Point2d frac_br;
    PixelToFractal(pix_tl, frac_tl);
    PixelToFractal(pix_br, frac_br);
    computeFunctions_[activeCompute_].fn(pix_br, frac_tl, frac_br, maxIterations_, fractal_);
    auto t2 = std::chrono::high_resolution_clock::now();
    frameTime_ = std::chrono::duration<double>(t2 - t1).count() * 1000.0;

    fractalImage_.forEach<cv::Vec3b>([this](cv::Vec3b& pixel, const int* position) {
        const cv::Vec3d p = this->fractal_.at<cv::Vec3d>(position[0], position[1]);
        pixel = this->colorFunctions_[this->activeColor_].fn(p[0], this->maxIterations_, p[1], p[2]);
        });
  }

  void PrintComputeMethods()
  {
    for (int i = 1; i <= computeFunctions_.size(); ++i)
    {
      printf("%d %s\n", i, computeFunctions_[i].name.c_str());
    }
  }

  void PrintStats()
  {
    printf("\rDelta: %-6dms| Iterations: %-6d| Compute: %-20s| Color: %-20s", frameTime_, maxIterations_,
      computeFunctions_[activeCompute_].name.c_str(), colorFunctions_[activeColor_].name.c_str());
  }

private:
  cv::Mat fractal_;
  cv::Mat fractalImage_;
  int width_, height_;
  int maxIterations_ = 200;
  int frameTime_ = 0;
  cv::Point2d startPan_;
  cv::Point2d startZoom_;
  cv::Point2d offset_;
  cv::Point2d scale_;
  bool isPanning_ = false;
  bool isZooming_ = false;

  void PixelToFractal(const cv::Point2i& p, cv::Point2d& f)
  {
    f.x = (double)(p.x) / scale_.x + offset_.x;
    f.y = (double)(p.y) / scale_.y + offset_.y;
  }

  struct ColorCallback
  {
    std::string name;
    using FnDef = cv::Vec3b(*)(int iterations, int maxIterations, double zr, double zi);
    FnDef fn;
  };
  std::vector<ColorCallback> colorFunctions_;
  int activeColor_ = 0;
  void RegisterColorFunctions()
  {
    colorFunctions_.push_back(ColorCallback{"B&W", ColorBW});
    colorFunctions_.push_back(ColorCallback{"HSV", ColorHSV});
    colorFunctions_.push_back(ColorCallback{"SmoothHSV", ColorSmoothHSV});
    colorFunctions_.push_back(ColorCallback{"Calm", ColorCalm});
  }

  struct ComputeCallback
  {
    std::string name;
    using FnDef = void (*)(const cv::Point2i& image_size, const cv::Point2d& fractal_tl,
      const cv::Point2d& fractal_br, int max_iterations, cv::Mat& fractal);
    FnDef fn;
  };
  std::map<int, ComputeCallback> computeFunctions_;
  int activeCompute_ = 1;
  void RegisterComputeFunctions()
  {
    int compute_n = 0;
    ++compute_n;
    computeFunctions_[compute_n] = ComputeCallback{"Serial", ComputeSerial};

#ifdef USE_TBB
    ++compute_n;
    computeFunctions_[compute_n] = ComputeCallback{"MultiThreaded", ComputeThreaded};
#endif // USE_TBB

#ifdef __ARM_NEON
    ++compute_n;
    computeFunctions_[compute_n] = ComputeCallback{"NEON", ComputeNEON};
#ifdef USE_TBB
    ++compute_n;
    computeFunctions_[compute_n] = ComputeCallback{"Threaded-NEON", ComputeThreadedNEON};
#endif // USE_TBB
#endif // __ARM_NEON

#ifdef __AVX2__
    ++compute_n;
    computeFunctions_[compute_n] = ComputeCallback{"AVX", ComputeAVX};
#ifdef USE_TBB
    ++compute_n;
    computeFunctions_[compute_n] = ComputeCallback{"Threaded-AVX", ComputeThreadedAVX};
#endif // USE_TBB
#endif // __AVX2__
  }
};

int main(int argc, char** argv)
{
  int width{};
  int height{};
  if (argc < 3)
  {
    printf("Image size not supplied. You can supply the width and height as the two argemnts to the program. Defaulting to 1280x800.\n");
    width = 1280;
    height = 800;
  }
  else
  {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
  }

  int rem_4 = width % 4;
  if (rem_4 != 0)
  {
    printf("Adjusting width to be divisible by 4.\n");
    width -= rem_4;
  }

  printf("Initializing with image size (%d, %d).\n", width, height);
  FractalsRenderer r(width, height);

  printf("Hold mouse left button to pan and right button to zoom.\n");
  printf("a, z Increase/decrease the number of iterations.\n");
  printf("s Save image.\n");
  printf("c Cycle through coloring methods.\n");
  printf("1, 2, 3,... Change fractal computation method.\n");
  printf("Available computation methods:\n");
  r.PrintComputeMethods();
  printf("\n");

  cv::namedWindow("Fractals", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(
    "Fractals", [](int event, int x, int y, int flags, void* userdata)
    { static_cast<FractalsRenderer*>(userdata)->OnMouseEvent(event, x, y, flags); },
    &r);

  while (true)
  {
    const char key = (char)cv::waitKey(20);
    if (key == 'q' || key == 'Q')
    {
      break;
    }
    else if (key == 'a')
    {
      int N = r.GetMaxIterations() + 50;
      r.SetMaxIterations(N);
    }
    else if (key == 'z')
    {
      int N = std::max(r.GetMaxIterations() - 50, 50);
      r.SetMaxIterations(N);
    }
    else if (key == 's')
    {
      cv::imwrite("./image.png", r.GetFractalImage());
    }
    else if (key == 'c')
    {
      r.CycleColor();
    }
    else
    {
      int num = key - '0';
      r.SetFractalComputeMethod(num);
    }
    r.Compute();
    r.PrintStats();
    cv::imshow("Fractals", r.GetFractalImage());
  }

  return 0;
}
