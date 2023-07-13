#include <torch/torch.h>
#include <vector>
#include "flextop_FD.h"
#include <stdio.h>
#include <string>

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
  auto distMatrix = fftd.get_dist_matrix(x2,a2);

  double distance = fftd.get_target_distance(distMatrix,"hungarian");
  double distance2 = fftd.get_target_distance(distMatrix,"sinkhorn-knopp");

  // Print the result
  std::cout << "Hungarian: " << distance << std::endl;
  std::cout << "Sinkhorn-Knopp: " << distance2 << std::endl;

  return 0;
}