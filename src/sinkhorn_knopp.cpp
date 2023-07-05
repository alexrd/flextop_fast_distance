#include <torch/script.h>
#include <ATen/ATen.h>
#include "sinkhorn_knopp.h"

SKAlgorithm::SKAlgorithm(){}
SKAlgorithm::~SKAlgorithm(){}

double SKAlgorithm::Solve(torch::Tensor DistMatrix, int max_iters, float convergence_eps, float epsilon, int check_frequency)
{
    DistMatrix = DistMatrix.div_(1000.0);
    torch::Tensor Q = torch::exp(-DistMatrix / epsilon);

    torch::Tensor ones_n = torch::ones(Q.size(0));
    torch::Tensor ones_m = torch::ones(Q.size(1));
    bool converged = false;
    int i = 1;

    while (i <= 10 && !converged) {
        // Row normalization
        Q.mul_(torch::reciprocal(torch::sum(Q, 1)).unsqueeze(1));
        
        // Column normalization
        Q.mul_(torch::reciprocal(torch::sum(Q, 0)));

        if (i % check_frequency == 0) {
            double norm_row = torch::max(torch::abs(torch::sum(Q, 1).sub(ones_n))).item<double>();
            double norm_col = torch::max(torch::abs(torch::sum(Q, 0).sub(ones_m))).item<double>();
            converged = (norm_col <= convergence_eps) && (norm_row <= convergence_eps);
        }
        //break;
        i++;
    }
    return Q.mul_(DistMatrix).sum().item<double>()*1000;
}