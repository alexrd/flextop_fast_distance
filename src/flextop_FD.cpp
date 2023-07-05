#include "flextop_FD.h"
#include "Hungarian.h"
#include "sinkhorn_knopp.h"
#include "assert.h"
#include <torch/script.h>
#include <ATen/ATen.h>
#include <string>

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


double FlexTopFastDistance::get_target_distance(torch::Tensor positions,
				     torch::Tensor attr, string algorithm) {

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

  // make matrix square by padding with mismatch penalty
  int dim0Size = distMatTensor.sizes()[0];
  int dim1Size = distMatTensor.sizes()[1];
  distMatTensor = torch::constant_pad_nd(distMatTensor, {0, max(dim0Size-dim1Size, 0), 
							0, max(dim1Size-dim0Size, 0)}, 
					Flextop::lambda_mismatch_penalty);

  std::vector<std::vector<double> > distMatrix = tensorTo2DVec(distMatTensor.data_ptr<double>(),
							       distMatTensor.sizes()[0],
							       distMatTensor.sizes()[1]);

  double distance;
  if (algorithm == "hungarian") {
    auto assignment = hungAlg.Solve(distMatrix);
    distance = 0;
    for (int i = 0; i < assignment.size(); i++) {
      distance += distMatrix[i][assignment[i]];
    }
  }
  else if (algorithm == "sinkhorn-knopp") {distance = SKAlg.Solve(distMatTensor);}
  else {throw std::invalid_argument("algorithm must be either \"hungarian\" or \"sinkhorn-knopp\"");}
  
  // get L2 norm from target
  
  // return
  return distance;
  
}

void FlexTopFastDistance::set_target(torch::Tensor positions, torch::Tensor attr) {

  // determine target features from positions and attributes
  at::IntArrayRef pos_sizes = positions.sizes();
  at::IntArrayRef attr_sizes = attr.sizes();

  assert(pos_sizes[0]==attr_sizes[0]);
  int numGhostParticles = pos_sizes[0];

  assert(pos_sizes[1]==3);  // x,y,z
  assert(attr_sizes[1]==4); // charge,epsilon,sigma,lambda

  torch::Tensor ani_features = getFeatures(positions,attr);

  targetFeaturesTensor = torch::cat({ani_features, attr},1);
  
  return;
}



