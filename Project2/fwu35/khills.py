import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
import opt.example.KHillsEvaluationFunction as KHillsEvaluationFunction
from array import array


"""
Commandline parameter(s):
    none
"""

# Random number generator */
random = Random()

# dimension
N = 2

# number of peaks
K = 50

# means of k-peaks
mean = [[50*random.nextDouble() for x in xrange(N)] for x in xrange(K)]

# standard deviations of k-peaks
std = [[20*random.nextDouble() for x in xrange(N)] for x in xrange(K)]

# heights of k-peaks
height = [1000*random.nextDouble() for x in xrange(K)];

# range of bit strings
fill = [100] * N
ranges = array('i', fill)

ef = KHillsEvaluationFunction(mean, std, height)

odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# first find the global optimum by running for a long time
hcp0 = GenericHillClimbingProblem(ef, odd, nf)
rhc0 = RandomizedHillClimbing(hcp0)
i = 0
max = 0
while (i < 1000000):
    rhc0.train()
    i += 1
    max = ef.value(rhc0.getOptimal())
    print "rhc0,", i,",", max
goal = 0.999*max
i
# run RHC
rhc = RandomizedHillClimbing(hcp)
max = 0
i = 0
while (max < goal):
    rhc.train()
    i += 1
    max = ef.value(rhc.getOptimal())
print "rhc,", i,",", max, ',', goal

# run SA
sa = SimulatedAnnealing(100, .95, hcp)
max = 0
i = 0
while (max < goal):
    sa.train()
    i += 1
    max = ef.value(sa.getOptimal())
print "sa,", i,",", max, ',', goal

# run GA
ga = StandardGeneticAlgorithm(200, 150, 25, gap)
max = 0
i = 0
while (max < goal):
    ga.train()
    i += 200
    max = ef.value(ga.getOptimal())
print "ga,", i,",", max, ',', goal

# run MIMIC
mimic = MIMIC(200, 100, pop)
max = 0
i = 0
while (max < goal):
    mimic.train()
    i += 200
    max = ef.value(mimic.getOptimal())
print "mimic,", i, ",", max, ',', goal

