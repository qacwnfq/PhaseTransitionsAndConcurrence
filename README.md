[![Build Status](https://travis-ci.org/qacwnfq/phaseTransitionsAndConcurrence.svg?branch=master)](https://travis-ci.org/qacwnfq/phaseTransitionsAndConcurrence)


This code was used to calculate the concurrence in https://arxiv.org/pdf/1612.08265.pdf

Get the code with:
```
git clone -recursive https://github.com/qacwnfq/phaseTransitionsAndConcurrence/
```

Install:
```
cd src/
make
./runQuantumAnnealing.out
```

Run the boost unit tests:
This is optional, but useful for CI. Install the library on UBUNTU with `sudo apt-get install libboost-test-dev`.
Travis-CI automatically executes the tests and complains if they fail.
```
cd tests/
make
./test_QuantumAnnealing.out
```