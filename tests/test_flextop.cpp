#include <torch/torch.h>
#include <vector>
#include "flextop_FD.h"
#include <stdio.h>
#include <string>
#include <chrono>

using namespace Flextop;
using namespace std::chrono;

int main(int argc, char* argv[]) {

  if (argc != 3) {
	std::cout << "Usage:  natoms1 natoms2" << std::endl;
	return 1;
  }
  
  // Define the torch::Tensor objects
  int npart1 = stoi(argv[1]);
  torch::Tensor a1 = torch::rand({npart1, 4},torch::TensorOptions().dtype(torch::kFloat64)); 
  torch::Tensor x1 = torch::rand({npart1, 3},torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64)); 
  
  int npart2 = stoi(argv[2]);
  torch::Tensor a2 = torch::rand({npart2, 4},torch::TensorOptions().dtype(torch::kFloat64)); 
  torch::Tensor x2 = torch::rand({npart2, 3},torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64)); 

  double penalty = 1.0;
  FlexTopFastDistance fftd("../models/ani_model_cpu.pt", penalty);

  // Set the target features
  fftd.set_target(x1, a1); 

  auto start = high_resolution_clock::now();
  double distance = fftd.get_target_distance(x2,a2,"hungarian");
  auto lap = high_resolution_clock::now();
  double distance2 = fftd.get_target_distance(x2,a2,"sinkhorn-knopp");
  auto stop = high_resolution_clock::now();

  //get times
  int h_time = duration_cast<microseconds>(lap - start).count();
  int sk_time = duration_cast<microseconds>(stop - lap).count();

  // Print the result
  std::cout << "Hungarian: " << distance << " Time: " << h_time << "\u03BCs" << std::endl;
  std::cout << "Sinkhorn-Knopp: " << distance2 << " Time: " << sk_time << "\u03BCs" << std::endl;

  return 0;
}