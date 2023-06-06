# Flextop Fast Distance

This is a Python/C++ library that calculates the distance between molecules using
the flexible topology framework. 

UNDER CONSTRUCTION

# Install instructions

Clone the repo:

`git clone https://github.com/alexrd/flextop_fast_distance.git`

Create a conda environment and install pytorch (skip this if you have a version of pytorch installed already):
```
conda env create -n flextop python=3.9
conda activate flextop
conda install pytorch -c pytorch
```

Go to the `flextop_fast_distance` folder and make a `build` folder inside it:
```
cd flextop_fast_distance
mkdir build; cd build
```

Make the package with CMake, passing the `torch` cmake path:
```
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
```

Install it:
```
make install
```

You can run this test to see if its working:
```
./TestFlextopFD 10 10
```