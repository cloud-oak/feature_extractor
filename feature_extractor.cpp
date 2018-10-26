#include <iostream>
#include <string>
#include <vector>
#include "cnpy.h"
#include "float_conversion.h"
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;
using namespace std::string_literals;

constexpr size_t BLUE   =  0;
constexpr size_t GREEN  =  1;
constexpr size_t RED    =  2;
constexpr size_t RE1    =  3;
constexpr size_t RE2    =  4;
constexpr size_t RE3    =  5;
constexpr size_t NIR    =  6;
constexpr size_t SWIR1  =  7;
constexpr size_t SWIR2  =  8;
constexpr size_t CLOUDS =  9;
constexpr size_t SHAPE  = 10;

constexpr size_t NUM_THREADS = 8;
constexpr size_t BANDS = 6;

const array<float, 3> QUANTILES {0.2f, 0.5f, 0.8f};
const size_t QS = QUANTILES.size();

typedef array<float, BANDS> Point;
typedef vector<Point> PointCloud;
typedef pair<vector<float>, vector<PointCloud>> timeseries;

constexpr float START_DAY = 31 + 28; // Only use images starting in March
constexpr float MAX_DAY   = 365 - START_DAY - 31; // End with November

time_t basetime;

struct result
{
  string number;
  char type;
  vector<float> features;
};

array<thread, NUM_THREADS> threads;
array<string, NUM_THREADS> current;
array<vector<result>, NUM_THREADS> results;
mutex queue_lock;
queue<string> todo;

#define DATA(b) half_to_float(raw.data<uint16_t>()[ b * W * H * T + x * H * T + y * T + t ])

float clamp(float x, float min=-100, float max=100)
{
  if (x > max)
    return max;
  else if (x < min)
    return min;
  else if (isnormal(x))
    return x;
  else
    return 0;
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
           cloudy_points = 0,
           total_points = 0;
    for(size_t x = 0; x < W; ++x) for(size_t y = 0; y < H; ++y)
    {
      if(DATA(SHAPE) > 0)
      {
        ++total_points;
        if(DATA(CLOUDS) > 0)
          ++cloudy_points;
        float red  = DATA(RED),
              grn  = DATA(GREEN),
              blu  = DATA(BLUE),
              nir  = DATA(NIR),
              swir1= DATA(SWIR1),
              swir2= DATA(SWIR2),
              re1  = DATA(RE1),
              re2  = DATA(RE2),
              re3  = DATA(RE3);
        if(red == 0 && grn == 0 && blu == 0)
          ++glitchy_points;
        float ndvi = (nir - red) / (nir + red); // Normalized Difference Vegetation Index
        float cai = (swir1 - swir2) / (swir1 + swir2); // Cellulose Absorption index
        float n = (2 * (nir * nir - red * red) + 1.5f * nir + 0.5 * red ) / (nir + red + 0.5f);
        float gemi = n * (1.0f - 0.25f * n) - (red - 0.125f) / (1 - red); // Global Environment Monitoring Index
        float gli = (2 * grn - red - blu) / (2 * grn + red + blu); // Green Leaf Index
        float cvi = (nir * red) / (grn * grn); // Chlorophyll vegetation index
        float ccci = ((nir - re1) / (nir + re1)) / ndvi; // Canopy Chlorophyll Content Index
        float dswi = (re3 - grn) / (swir1 + red); // Disease-Water Stress Index 5
        float nd790_670 = (re3 - red) / (re3 + red); // Normalized Difference 790 / 670
        float ndwi = (nir - swir2) / (nir + swir2); // Normalized Difference Water Index
        points.push_back({
            clamp(ndvi),
            // clamp(cai),
            clamp(gemi),
            clamp(gli),
            clamp(cvi),
            // clamp(ccci),
            clamp(dswi),
            // clamp(nd790_670),
            clamp(ndwi),
        });
      }
    }
    if(cloudy_points > 0.2 * total_points)
      continue;
    if(glitchy_points > 0.2 * total_points)
      continue;
    all_points.push_back(points);
    used_timestamps.push_back(timestamps[t] / MAX_DAY);
  }

  return make_pair(used_timestamps, all_points);
}

inline float hat(float t, float center, float width)
{
  return max(1 - abs(t-center) / width, 0.0f) / width;
}

const vector<float> get_features(timeseries data, float subsample=1.0f)
{
  vector<float> time = data.first;
  vector<PointCloud> point_series  = data.second;
  int count = point_series[0].size();

  vector<bool> used(count, true);
  if(subsample < 1.0f)
  {
    count = 0;
    random_device rd;
    mt19937 random(rd());
    uniform_real_distribution<> real(0.0, 1.0);
    for(auto &&b : used)
    {
      b = real(random) < subsample;
      count += b;
    }
  }

  vector<float> features;
  array<vector<float>, BANDS * QS> Q;

  for(PointCloud point_cloud : point_series)
  {
    for(size_t band = 0; band < BANDS; ++band)
    {
      vector<float> points;
      for(size_t i = 0; i < point_cloud.size(); ++i)
      {
        if(used[i] && !isnan(point_cloud[i][band]))
        {
          points.push_back(point_cloud[i][band]);
        }
      }
      if(points.size() > 0)
      {
        sort(points.begin(), points.end());
        for(size_t q = 0; q < QS; ++q)
          Q[band * QS + q].push_back(points[size_t(points.size() * QUANTILES[q])]);
      }
      else
      {
        for(float q : QUANTILES)
          Q[band * QS + q].push_back(0);
      }
    }
  }

  for(auto q : Q)
  {
    constexpr size_t DIM = 8;
    // vector<complex<float>> coefficients(NUM_FOURIER, 0);
    Eigen::Matrix<float, Eigen::Dynamic, DIM> model(q.size(), DIM);
    Eigen::Matrix<float, Eigen::Dynamic, 1> target(q.size(), 1);
    for(size_t k = 0; k < q.size(); ++k)
    {
      target(k, 0) = q[k];
      model(k, 0)  = hat(time[k], 0.0f  , 1.0f  ); // Left Edge
      model(k, 1)  = hat(time[k], 1.0f  , 1.0f  ); // Right Edge
      model(k, 2)  = hat(time[k], 0.5f  , 0.5f  ); // 1/2
      model(k, 3)  = hat(time[k], 0.25f , 0.25f ); // 1/4
      model(k, 4)  = hat(time[k], 0.75f , 0.25f ); // 3/4
      model(k, 5)  = hat(time[k], 0.125f, 0.125f); // 1/8
      model(k, 6)  = hat(time[k], 0.375f, 0.125f); // 3/8
      model(k, 7)  = hat(time[k], 0.625f, 0.125f); // 5/8
      // model(k, 8)  = hat(time[k], 0.875f, 0.125f); // 7/8
    }
    Eigen::Matrix<float, DIM, DIM> H = model.transpose() * model;
    H = H.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::Matrix<float, DIM, 1> weight = H * model.transpose() * target;
    for(size_t k = 0; k < DIM; ++k)
    {
      features.push_back(weight(k, 0));
    }
  }
  return features;
}

inline bool file_exists (const std::string& name)
{
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0); 
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
    
    string filename = "/home/konrad/dev/remote_sensing/ibiss_processed/cubes/" + number;
    if(!file_exists(filename + ".npy"s))
      continue;
    timeseries data = parse(filename);

    results[thread_id].push_back({ number, 'F', get_features(data, 0.5) });
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

  ofstream out("/home/konrad/dev/remote_sensing/data/cfeatures.csv");

  for(size_t i = 0; i < 8*NUM_THREADS; ++i)
    cout << '_';
  cout << endl;

  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    for(auto p : results[i])
    {
      out << p.number << ',' << p.type;
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
