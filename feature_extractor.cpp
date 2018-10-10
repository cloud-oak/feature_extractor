#include <iostream>
#include <string>
// #include <utility> ?
#include <vector>
#include "cnpy.h"
#include "float_conversion.h"
#include "transforms.h"
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <algorithm>
#include <random>
#include <complex>

using namespace std;
using namespace std::string_literals;

constexpr size_t BLUE   =  0;
constexpr size_t RED    =  1;
constexpr size_t GREEN  =  2;
constexpr size_t RE1    =  3;
constexpr size_t RE2    =  4;
constexpr size_t RE3    =  5;
constexpr size_t NIR    =  6;
constexpr size_t SWIR1  =  7;
constexpr size_t SWIR2  =  8;
constexpr size_t CLOUDS =  9;
constexpr size_t SHAPE  = 10;

constexpr size_t NUM_THREADS = 8;
constexpr size_t BANDS = 1;
constexpr size_t QUANTILES = 1;

constexpr float START_DAY = 31 + 28; // Only use images starting in March

const vector<float> FREQUENCIES = { 0.0, 1.0, 2.0, 3.0, 4.0, 6 };
const int           NUM_FOURIER = FREQUENCIES.size();

typedef array<float, BANDS> Point;
typedef vector<Point> PointCloud;
typedef pair<vector<float>, vector<PointCloud>> timeseries;

time_t basetime;

struct result
{
  string number;
  vector<float> features;
};

array<thread, NUM_THREADS> threads;
array<string, NUM_THREADS> current;
array<vector<result>, NUM_THREADS> results;
mutex queue_lock;
queue<string> todo;

#define DATA(b) half_to_float(raw.data<uint16_t>()[ b * W * H * T + x * H * T + y * T + t ])

float clamp(float x, float min=-2, float max=2)
{
  if (x > max)
  {
    cout << "Clamped " << x << endl;
    return max;
  }
  else if (x < min)
  {
    cout << "Clamped " << x << endl;
    return min;
  }
  else
  {
    return x;
  }
}

timeseries parse(string filename)
{
  vector<float> timestamps;
  vector<PointCloud> all_points;

  cnpy::NpyArray raw = cnpy::npy_load(filename + ".npy"s);
  ifstream indexfile(filename + ".idx"s);
  if(indexfile.is_open())
  {
    string line;
    // Skip two lines
    getline(indexfile, line);
    getline(indexfile, line);
    // Skip the label
    getline(indexfile, line, '~');
    // Get entries
    while(getline(indexfile, line, ','))
    {
      struct tm tm;
      strptime(line.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
      float diff = difftime(mktime(&tm), basetime) / (60 * 60 * 24) - START_DAY;
      timestamps.push_back(diff);
    }
    indexfile.close();
  }

  //  size_t B = raw.shape[0];
  size_t W = raw.shape[1];
  size_t H = raw.shape[2];
  size_t T = raw.shape[3];
  vector<float> used_timestamps;

  assert(timestamps.size() == T);

  for(size_t t = 0; t < T; ++t)
  {
    if(timestamps[t] <= 0)
      continue;
    PointCloud points;
    size_t glitchy_points = 0,
           cloudy_points  = 0,
           total_points   = 0;
    for(size_t x = 0; x < W; ++x)
    {
      for(size_t y = 0; y < H; ++y)
      {
        if(DATA(SHAPE) > 0)
        {
          ++total_points;
          if(DATA(CLOUDS) > 0)
            ++cloudy_points;
          float r    = DATA(RED),
                g    = DATA(GREEN),
                b    = DATA(BLUE),
                nir  = DATA(NIR);
                // swir = DATA(SWIR1),
                // re2  = DATA(RE2);
          if(r == 0 || g == 0 || b == 0)
            ++glitchy_points;
          points.push_back({
              clamp((nir  -   r) / (nir  + r + 1e-8)), // NDVI
          });
        }
      }
    }
    if(cloudy_points  > 0.2 * total_points)
      continue;
    if(glitchy_points > 0.2 * total_points)
      continue;
    all_points.push_back(points);
    used_timestamps.push_back(timestamps[t]);
  }

  return make_pair(used_timestamps, all_points);
}

void work(size_t thread_id)
{
  current[thread_id] = "Initializing";
  bool running = true;
  string number = "";
  while(running)
  {
    {
      lock_guard<mutex> lock(queue_lock);
      if(todo.empty())
      {
        break;
      } else {
        number = todo.front();
        current[thread_id] = number;
        todo.pop();
      }
    }
    timeseries data = parse(
        "/home/konrad/dev/remote_sensing/ibiss_processed/cubes/" + number
    );

//    results[thread_id].push_back({ number, 'F', get_features(data) });
    vector<float> time = data.first;
    vector<PointCloud> point_series = data.second;
    int N = point_series.size();
    int P = point_series[0].size();

    array<vector<float>, BANDS> Q; // Q contains the mean values per band

    for(int p = 0; p < P; p += 5)
    {
      vector<float> features;
      for(size_t band = 0; band < BANDS; ++band)
      {
        vector<complex<float>> coefficients(NUM_FOURIER, 0);

        // Non-uniform discrete Fourier transform
        for(int i = 0; i < N; ++i)
          for(int k = 0; k < NUM_FOURIER; ++k)
          {
            auto cmp = complex<float>(0, -2 * M_PI / 365.0f * time[i] * FREQUENCIES[k]);
            coefficients[k] += point_series[i][p][band] * exp(cmp);
          }
        for(int k = 0; k < NUM_FOURIER; ++k)
        {
          features.push_back(coefficients[k].real() / N);
          features.push_back(coefficients[k].imag() / N);
        }
      }
      results[thread_id].push_back({ number, features });
    }
  }
  current[thread_id] = "Done";
}

int main(int argc, char **argv)
{
  struct tm tm;
  strptime("2017-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S", &tm);
  basetime = mktime(&tm);

  size_t count = 0;
  if(argc == 1)
  {
    string number = "";
    while(getline(cin, number))
    {
      todo.push(number);
      ++count;
    }
  }
  else
  {
    string number = "";
    ifstream input(argv[1]);
    if(!input.is_open())
      return 1;
    while(getline(input, number))
    {
      todo.push(number);
      ++count;
    }
    input.close();
  }

  cout << "Beginning feature extraction" << endl;
  auto features = vector<pair<string, vector<float>>>(count);
  size_t num = features.size();

  for(size_t i = 0; i < 100; ++i)
    cout << '_';
  cout << endl;

  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    threads[i] = thread(work, i);
  }

  size_t progress = 100;
  while(progress > 0)
  {
    if(progress * num > todo.size() * 100)
    {
      cout << '#' << flush;
      --progress;
    }
    else
    {
      this_thread::sleep_for(100ms);
    }
  }

  cout << endl;
  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    cout << "Thread (" << i << ") processed " << results[i].size() << " elements" << endl;
  }

  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    threads[i].join();
  }
  cout << "\nCollected all threads" << endl;

  cout << "\nWriting CSV..." << endl;

  ofstream out("/home/konrad/dev/remote_sensing/ibiss_processed/cfeatures.csv");

  for(size_t i = 0; i < 8*NUM_THREADS; ++i)
    cout << '_';
  cout << endl;

  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    for(auto p : results[i])
    {
      out << p.number;
      for(auto val : p.features)
        out << "," << val;
      out << '\n';
    }
    cout << "########" << flush;
    results[i].clear();
  }
  cout << endl;
  out.close();

  return 0;
}
