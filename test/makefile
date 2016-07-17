# Author Fredrik Jadebeck
#
# Makefile for the boost unit tests of test_quantumMechanics.cpp
test: test_quantumMechanics test_quantumAnnealing

test_quantumMechanics: ../code/quantumMechanics.cpp test_quantumMechanics.cpp
	g++ ../code/quantumMechanics.cpp test_quantumMechanics.cpp -I /usr/include/eigen3/ -std=c++11 -lboost_unit_test_framework -I /home/fredrik/repos/phaseTransitionsAndConcurrence/code/ -w -o test_quantumMechanics.out

test_quantumAnnealing:  ../code/quantumMechanics.cpp test_quantumAnnealing.cpp ../code/quantumAnnealing.cpp
	g++ ../code/quantumAnnealing.cpp ../code/quantumMechanics.cpp test_quantumAnnealing.cpp -I /usr/include/eigen3/ -std=c++11 -lboost_unit_test_framework -I /home/fredrik/repos/phaseTransitionsAndConcurrence/code/ -w -o test_quantumAnnealing.out
