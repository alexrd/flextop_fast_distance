#include <torch/torch.h>
#include <vector>
#include "flextop_FD.h"
#include <stdio.h>
#include <string>
#include <chrono>

using namespace Flextop;
using namespace std::chrono;

int randint (int min, int max){
    int n = max - min + 1;
    int remainder = RAND_MAX % n;
    int x;
    while (x >= RAND_MAX - remainder) {
        x = rand();
    }
    return min + x % n;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
	std::cout << "Usage: min_atoms max_atoms iterations <mismatch_cost>" << std::endl;
	return 1;
  }

  // define constants
  const int MIN = stoi(argv[1]);
  const int MAX = stoi(argv[2]);
  const int SIM_COUNT = stoi(argv[3]);

  // set random seed
  srand (time(NULL));

  // set mismatch cost if needed
  double penalty = 1.0;
  if (argc > 4) {
    penalty = stoi(argv[4]);
  }
  // initialize model
  FlexTopFastDistance fftd("../models/ani_model_cpu.pt", penalty);
  
  double hg_total = 0;
  double sk_total = 0;
  double distMat_total = 0;

  double hg_total_no_outliers = 0;
  double sk_total_no_outliers = 0;
  double distMat_total_no_out = 0;

  for (int i = 0; i < SIM_COUNT; i++) {
    // Define the torch::Tensor objects
    int npart1 = randint(MIN, MAX);
    torch::Tensor a1 = torch::rand({npart1, 4},torch::TensorOptions().dtype(torch::kFloat64)); 
    torch::Tensor x1 = torch::rand({npart1, 3},torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64)); 
    
    int npart2 = randint(MIN, MAX);
    torch::Tensor a2 = torch::rand({npart2, 4},torch::TensorOptions().dtype(torch::kFloat64)); 
    torch::Tensor x2 = torch::rand({npart2, 3},torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64)); 

    // Set the target features
    fftd.set_target(x1, a1);

    auto start = high_resolution_clock::now();
    auto distMatrix = fftd.get_dist_matrix(x2,a2);
    auto lap1 = high_resolution_clock::now();
    double distance = fftd.get_target_distance(distMatrix,"hungarian");
    auto lap2 = high_resolution_clock::now();
    double distance2 = fftd.get_target_distance(distMatrix,"sinkhorn-knopp");
    auto stop = high_resolution_clock::now();

    //get times
    double hg_time = duration_cast<microseconds>(lap2 - lap1).count();
    double sk_time = duration_cast<microseconds>(stop - lap2).count();
    double dMat_time = duration_cast<microseconds>(lap1 - start).count();

    hg_total += hg_time;
    sk_total += sk_time;
    distMat_total += dMat_time;

    if (i > 2) {
        hg_total_no_outliers += hg_time;
        sk_total_no_outliers += sk_time;
        distMat_total_no_out += dMat_time;
    }

    //std::cout << hg_time << std::endl;
    //std::cout << sk_time << std::endl;
  }

  // Print the result
  std::cout << "Hungarian average time: " << hg_total/SIM_COUNT << "\u03BCs" << std::endl;
  std::cout << "Sinkhorn-Knopp average time: " << sk_total/SIM_COUNT << "\u03BCs" << std::endl;
  std::cout << "Distance Matrix Calculation Time: " << distMat_total/SIM_COUNT << "\u03BCs" << std::endl;
  std::cout << "Hungarian average time (iteration 3+): " << hg_total_no_outliers/(SIM_COUNT-3) << "\u03BCs" << std::endl;
  std::cout << "Sinkhorn-Knopp average time (iteration 3+): " << sk_total_no_outliers/(SIM_COUNT-3) << "\u03BCs" << std::endl;
  std::cout << "Distance Matrix Calculation Time (it 3+): " << distMat_total_no_out/(SIM_COUNT-3) << "\u03BCs" << std::endl;

  return 0;
}