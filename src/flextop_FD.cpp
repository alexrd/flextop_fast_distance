#include "flextop_FD.h"
#include "Hungarian.h"
#include "assert.h"
#include <torch/script.h>
#include <ATen/ATen.h>

using namespace Flextop;
using namespace std;

/**
 * @brief
 *
 * @param ptr
 * @param nRows
 * @param nCols
 * @return std::vector<std::vector<double> >
 */
std::vector<std::vector<double> > tensorTo2DVec(double* ptr, int nRows, int nCols) {
	std::vector<std::vector<double> > distMat(nRows, std::vector<double>(nCols));
	for (int i=0; i<nRows; i++) {
		std::vector<double> vec(ptr+nCols*i, ptr+nRows*(i+1));
		distMat[i] = vec;
	}
	return distMat;
}

torch::Tensor getFeatures(torch::Tensor positions, torch::Tensor attr) {
  // get features
  auto charges = attr.index({Slice(), 0});
  vector<torch::jit::IValue> nnInputs = {positions, charges};

  torch::Tensor features = nnModule.forward(nnInputs).toTensor();
  return features;
}


FlexTopFastDistance::~FlexTopFastDistance() {
}


std::vector<int> FlexTopFastDistance::get_target_distance(torch::Tensor positions,
				     torch::Tensor attr) {

  // ensure that positions and attr have the same size (the number of particles)
  at::IntArrayRef pos_sizes = positions.sizes();
  at::IntArrayRef attr_sizes = attr.sizes();

  assert(pos_sizes[0]==attr_sizes[0]);
  int numGhostParticles = pos_sizes[0];

  assert(pos_sizes[1]==3);  // x,y,z
  assert(attr_sizes[1]==4); // charge,epsilon,sigma,lambda

  torch::Tensor features = getFeatures(positions,attr);
  
  // get optimal atom mapping
  torch::Tensor ghFeaturesTensor = torch::cat({features, attr}, 1);
  torch::Tensor distMatTensor = at::norm(ghFeaturesTensor.index({Slice(), None})
					 - targetFeaturesTensor, 2, 2);
  
  std::vector<std::vector<double> > distMatrix = tensorTo2DVec(distMatTensor.data_ptr<double>(),
							       numGhostParticles,
							       static_cast<int>(targetFeaturesTensor.size(0)));

  std::vector<int> assignment = hungAlg.Solve(distMatrix);
  
  // get L2 norm from target
  
  // return
  return assignment;
  
}

void FlexTopFastDistance::set_target(torch::Tensor positions, torch::Tensor attr) {

  // determine target features from positions and attributes
  at::IntArrayRef pos_sizes = positions.sizes();
  at::IntArrayRef attr_sizes = attr.sizes();

  assert(pos_sizes[0]==attr_sizes[0]);
  int numGhostParticles = pos_sizes[0];

  assert(pos_sizes[1]==3);  // x,y,z
  assert(attr_sizes[1]==4); // charge,epsilon,sigma,lambda

  targetFeaturesTensor = getFeatures(positions,attr);
  
  return;
}



