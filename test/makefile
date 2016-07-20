# Author Fredrik Jadebeck
#
# Makefile for the boost unit tests of quantumAnnealing.cpp

test_quantumAnnealing: test_quantumAnnealing.cpp ../code/quantumAnnealing.cpp
	g++ ../code/quantumAnnealing.cpp  test_quantumAnnealing.cpp -I /usr/include/eigen3/ -std=c++11 -lgmpxx -lgmp -lboost_unit_test_framework -I /home/fredrik/repos/phaseTransitionsAndConcurrence/code/ -w -o test_quantumAnnealing.out
