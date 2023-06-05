#include <torch/torch.h>
#include <vector>
#include "flextop_FD.h" 

using namespace Flextop;

int main() {
    // Define the torch::Tensor objects
    int npart1 = 10;
    torch::Tensor a1 = torch::rand({npart1, 4},torch::TensorOptions().dtype(torch::kFloat64)); 
    torch::Tensor x1 = torch::rand({npart1, 3},torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64)); 
    
    int npart2 = 12;
    torch::Tensor a2 = torch::rand({npart2, 4},torch::TensorOptions().dtype(torch::kFloat64)); 
    torch::Tensor x2 = torch::rand({npart2, 3},torch::TensorOptions().requires_grad(true).dtype(torch::kFloat64)); 

    double penalty = 1.0;
    FlexTopFastDistance fftd("/Users/alexrd/research/flextop_fast_distance/models/ani_model_cpu.pt", penalty);

    // Set the target features
    fftd.set_target(x1, a1); 

    std::vector<int> assignments = fftd.get_target_distance(x2,a2);

    // Print the result
    std::cout << "Assignments: " << assignments << std::endl;

    return 0;
}