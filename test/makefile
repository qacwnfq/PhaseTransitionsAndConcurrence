# Author Fredrik Jadebeck
#
# Makefile for the boost unit tests of quantumAnnealing.cpp

test_quantumAnnealing: test_quantumAnnealing.cpp ../src/quantumAnnealing.cpp
	g++ ../src/quantumAnnealing.cpp  test_quantumAnnealing.cpp -I ../lib/eigen/ -std=c++11 -lboost_unit_test_framework -I /home/fredrik/repos/phaseTransitionsAndConcurrence/src/ -w -o test_quantumAnnealing.out
