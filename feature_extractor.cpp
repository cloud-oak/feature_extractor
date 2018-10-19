#include <iostream>
#include <string>
// #include <utility> ?
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
#include <complex>

using namespace std;
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
constexpr size_t BANDS = 4;
constexpr size_t SAMPLES = 8;

const vector<float> FREQUENCIES = { 0.0, 1.0, 2.0, 3.0, 4.0, 6 };
const int NUM_FOURIER = FREQUENCIES.size();

typedef array<float, BANDS> Point;
typedef vector<Point> PointCloud;
typedef pair<vector<float>, vector<PointCloud>> timeseries;

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

  //  size_t B = raw.shape[0];
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
          float red  = DATA(RED),
                grn  = DATA(GREEN),
                blu  = DATA(BLUE),
                nir  = DATA(NIR),
                swir = DATA(SWIR2);
          if(red == 0 && grn == 0 && blu == 0)
            ++glitchy_points;
          points.push_back({
              (nir - red ) / (nir + red), // NDVI
              2.5f * (nir - red ) / (nir + 6.0f * red - 7.5f * blu + 1.0f), // EVI
              (nir - swir) / (nir + swir), // NDWI
              (grn - nir ) / (grn + nir ), // NDWI2
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

vector<float> get_features(timeseries data, float subsample=1.0f, float quantile=0.5f)
{
  // sort(data.begin(), data.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
  vector<float> time = data.first;
  vector<PointCloud> point_series  = data.second;
  int count = point_series[0].size();
  int N = time.size();

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
  array<vector<float>, BANDS> Q;

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

        Q[band].push_back(points[size_t(points.size() * quantile)]);
      }
      else
      {
        Q[band].push_back(0);
      }
    }
  }

  for(auto q : Q)
  {
    vector<complex<float>> coefficients(NUM_FOURIER, 0);

    // Non-uniform discrete Fourier transform
    for(int i = 0; i < N; ++i)
    {
      for(int k = 0; k < NUM_FOURIER; ++k)
      {
        auto cmp = complex<float>(0, -2 * M_PI / 365.0f * time[i] * FREQUENCIES[k]);
        coefficients[k] += q[i] * exp(cmp);
      }
    }
    for(int k = 0; k < NUM_FOURIER; ++k)
    {
      features.push_back(coefficients[k].real() / N);
      features.push_back(coefficients[k].imag() / N);
    }
  }
  return features;
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

    results[thread_id].push_back({ number, 'F', get_features(data, 0.5) });
    for(float quantile : {0.4f, 0.5f, 0.6f})
      for(float subset : {0.3f, 0.5f, 0.7f})
        results[thread_id].push_back({ number, 'S', get_features(data, subset, quantile) });
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
