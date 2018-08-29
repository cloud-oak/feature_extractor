#include <iostream>
#include <string>
#include <utility>
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
constexpr size_t BANDS = 4;
constexpr size_t QUANTILES = 5;

typedef array<float, BANDS> Point;
typedef vector<Point> PointCloud;
typedef pair<vector<float>, vector<PointCloud>> timeseries;

time_t basetime;

array<thread, NUM_THREADS> threads;
array<vector<pair<string, vector<float>>>, NUM_THREADS> results;
mutex queue_lock;
queue<string> todo;

#define DATA(b) half_to_float(raw.data<uint16_t>()[ b * W * H * T + x * H * T + y * T + t ])

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
      float diff = difftime(mktime(&tm), basetime) / (60 * 60 * 24);
      timestamps.push_back(diff);
    }
    indexfile.close();
  }

  size_t B = raw.shape[0];
  size_t W = raw.shape[1];
  size_t H = raw.shape[2];
  size_t T = raw.shape[3];

  assert(timestamps.size() == T);

  for(size_t t = 0; t < T; ++t)
  {
    // half_to_float();
    PointCloud points;
    size_t glitchy_points = 0,
           cloudy_points = 0,
           total_points = 0;
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
                nir  = DATA(NIR),
                swir = DATA(SWIR1);
          if(r == 0 && g == 0 && b == 0)
            ++glitchy_points;
          points.push_back({
              (nir  -   r) / (nir +    r + 0.5f) * 1.5f, // SAVI
              (   r -   b) / (  r +    b + 0.5f) * 1.5f, // Redness
              (   g -   b) / (  g +    b + 0.5f) * 1.5f, // Greenness
              (swir - nir) / (nir + swir + 0.5f) * 1.5f  // 
          });
        }
      }
    }
    if(cloudy_points > 0.2 * total_points)
      continue;
    if(glitchy_points > 0.2 * total_points)
      continue;
    all_points.push_back(points);
  }

  return make_pair(timestamps, all_points);
}

vector<float> get_features(timeseries data)
{
  vector<float> time = data.first;
  vector<PointCloud> point_series  = data.second;
  vector<float> features;
  array<vector<float>, QUANTILES * BANDS> Q;

  float max_time = 0;
  for(size_t t = 0; t < time.size(); ++t)
  {
    if(time[t] > max_time)
      max_time = time[t];
  }
  for(size_t t = 0; t < time.size(); ++t)
  {
    time[t] -= max_time;
  }

  for(PointCloud point_cloud : point_series)
  {
    for(size_t band = 0; band < 4; ++band)
    {
      vector<float> points;
      for(Point p : point_cloud)
        points.push_back(p[band]);
      sort(points.begin(), points.end());

      Q[QUANTILES * band + 0].push_back(points[size_t(points.size() * 0.1)]);
      Q[QUANTILES * band + 1].push_back(points[size_t(points.size() * 0.2)]);
      Q[QUANTILES * band + 2].push_back(points[size_t(points.size() * 0.5)]);
      Q[QUANTILES * band + 3].push_back(points[size_t(points.size() * 0.8)]);
      Q[QUANTILES * band + 4].push_back(points[size_t(points.size() * 0.9)]);
    }
  }

  for(auto q : Q)
  {
    float integral = 0;
    array<float, 8> hats;
    array<float, 8> sincs;
    for(size_t i = 0; i < 8; ++i)
    {
      hats[i] = 0;
      sincs[i] = 0;
    }
    float max_time = -1e10;
    float min_time = -1e10;
    float max      = -1e10;
    float min      =  1e10;
    for(size_t t = 0; t < q.size(); ++t)
    {
      float dt = 0;
      if(t > 0)
        dt += (time[t] - time[t-1]) / 2;
      if(t < q.size() - 1)
        dt += (time[t+1] - time[t]) / 2;
      for(size_t month = 0; month < 8; month++)
      {
        sincs[month] += dt * sinc(q[t], month);
        hats[month] += dt * sinc(q[t], month);
      }
      integral += q[t] * dt;

      if(q[t] > max)
      {
        max_time = time[t];
        max = q[t];
      }
      if(q[t] < min)
      {
        min_time = time[t];
        min = q[t];
      }
    }
    features.push_back(integral);
    for(size_t month = 0; month < 8; month++)
      features.push_back(hats[month]);
    for(size_t month = 0; month < 8; month++)
      features.push_back(sincs[month]);
    features.push_back(min_time);
    features.push_back(max_time);
  }

  return features;
}

void work(size_t thread_id)
{
  bool running = true;
  string number = "";
  while(running)
  {
    {
      lock_guard<mutex> lock(queue_lock);
      if(todo.empty())
      {
        running = false;
        break;
      } else {
        number = todo.front();
        todo.pop();
      }
    }
    timeseries data = parse(
        "/home/konrad/dev/remote_sensing/ibiss_processed/cubes/" + number
    );

    results[thread_id].push_back(make_pair(number, get_features(data)));
  }
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

  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    threads[i] = thread(work, i);
  }

  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    threads[i].join();
  }

  ofstream out("/home/konrad/dev/remote_sensing/ibiss_processed/cfeatures.csv");
  for(size_t i = 0; i < NUM_THREADS; ++i)
  {
    for(auto p : results[i])
    {
      out << p.first;
      for(auto val : p.second)
        out << "," << val;
      out << '\n';
    }
    results[i].clear();
  }
  out.close();

  return 0;
}
