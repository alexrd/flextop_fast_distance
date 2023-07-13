#ifndef FLEXTOP_FD_H_
#define FLEXTOP_FD_H_

#include "Hungarian.h"
#include "sinkhorn_knopp.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <cstring>

using namespace std;
using namespace torch::indexing;
static const std::vector<string> PARAMETERNAMES={"charge_g", "sigma_g", "epsilon_g", "lambda_g"};

namespace Flextop {

  torch::jit::script::Module nnModule;
  double lambda_mismatch_penalty;
  std::vector<std::vector<double>> target_positions;
  std::vector<std::vector<double>> target_attr;

  class FlexTopFastDistance {
  public:
    FlexTopFastDistance(std::string model_name, double penalty) {
      nnModule = torch::jit::load(model_name);
      nnModule.to(torch::kCPU);
      nnModule.eval();
      lambda_mismatch_penalty = penalty;
    };
    ~FlexTopFastDistance();

    torch::Tensor get_dist_matrix(torch::Tensor positions,
				     torch::Tensor attr);

    double get_target_distance(torch::Tensor distMatrix, string algorithm="hungarian");

    void set_target(torch::Tensor positions,
		    torch::Tensor attr);

  private:
    torch::Tensor targetFeaturesTensor;
    HungarianAlgorithm hungAlg;
    SKAlgorithm SKAlg;

  };

} // namespace Flextop

#endif /*FLEXTOP_FD_H_*/
