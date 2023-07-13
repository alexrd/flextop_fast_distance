#ifndef SINKHORN_KNOPP_H
#define SINKHORN_KNOPP_H

#include <iostream>
#include <vector>
#include <torch/script.h>

using namespace std;


class SKAlgorithm
{
public:
	SKAlgorithm();
	~SKAlgorithm();
    double Solve(torch::Tensor DistMatrix, int max_iters=100, float convergence_eps=1e-4, float epsilon=1, int check_frequency=5);
};


#endif
