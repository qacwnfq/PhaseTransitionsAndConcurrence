[![Build Status](https://travis-ci.org/qacwnfq/phaseTransitionsAndConcurrence.svg?branch=master)](https://travis-ci.org/qacwnfq/phaseTransitionsAndConcurrence)


This code was used to calculate the concurrence in https://arxiv.org/pdf/1612.08265.pdf

Usage example in src/runQuantumAnnealing.cpp.

Get the code with:
```
git clone --recursive https://github.com/qacwnfq/phaseTransitionsAndConcurrence/
```

Compiling QuantumAnnealing.cpp (needs compiler which can compile c++11 standard):
```
cd src/
make
./runQuantumAnnealing.out
```

Run the boost unit tests:

This is optional, but useful for CI. Install the library on UBUNTU with ```sudo apt-get install libboost-test-dev```.
Travis-CI automatically executes the tests and complains if they fail.
```
cd tests/
make
./test_QuantumAnnealing.out
```



The magnetization.cpp script calculates the free energy and the magnetization of the systen in x- and z-direction.
It can be compiled with:
```
g++ -std=c++14 src/magnetization.cpp -O3
```

QuantumAnnealing.py contains additional functions useful for analysis,
like calculating the concurrence in the classical limit.