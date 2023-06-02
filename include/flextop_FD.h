#ifndef FLEXTOP_FD_H_
#define FLEXTOP_FD_H_

#include "PyTorchKernels.h"
#include "Hungarian.h"
#include <torch/torch.h>
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
      lambda_mismatch_penalty = penalty;
    };
    ~FlexTopFastDistance();

    double get_target_distance(std::vector<std::vector<double>> positions,
			       std::vector<std::vector<double>> attr);

    void set_target(std::vector<std::vector<double>> positions,
		    std::vector<std::vector<double>> attr);

  private:
    torch::Tensor targetFeaturesTensor;
    std::vector<std::vector<double> > targetFeatures;
    HungarianAlgorithm hungAlg;
    std::vector<int> assignment;
    std::vector<int> reverse_assignment;
  };

} // namespace Flextop

#endif /*FLEXTOP_FD_H_*/
