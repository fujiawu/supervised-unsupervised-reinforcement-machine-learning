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
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
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
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array




"""
Commandline parameter(s):
    none
"""

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 50
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)
# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME

# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

timeout = 1E6

# first find the global optimum by running for a long time
hcp0 = GenericHillClimbingProblem(ef, odd, nf)
rhc0 = RandomizedHillClimbing(hcp0)
i = 0
max = 0
while (i < timeout/10):
    rhc0.train()
    i += 1
    max = ef.value(rhc0.getOptimal())
    print "rhc0,", i,",", max
goal = max
pop0 = GenericProbabilisticOptimizationProblem(ef, odd, df)
mimic0 = MIMIC(200, 100, pop)
i = 0
while ( i< timeout/1000):
    mimic0.train()
    i += 1
    max = ef.value(mimic0.getOptimal())
    print "mimic0,", i,",", max
if (max > goal):
    goal = max
gap0 = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
ga0 = StandardGeneticAlgorithm(200, 100, 25, gap0)
i = 0
while ( i< timeout/1000):
    ga0.train()
    i += 1
    max = ef.value(ga0.getOptimal())
    print "ga0,", i,",", max
if (max > goal):
    goal = max

# run RHC
rhc = RandomizedHillClimbing(hcp)
max = 0
i = 0
while (max < goal and i < timeout):
    rhc.train()
    i += 1
    max = ef.value(rhc.getOptimal())
    #print "rhc,", i,",", max, ',', goal
print "rhc,", i,",", max, ',', goal

# run SA
sa = SimulatedAnnealing(1E11, .95, hcp)
max = 0
i = 0
while (max < goal and i < timeout):
    sa.train()
    i += 1
    max = ef.value(sa.getOptimal())
    #print "sa,", i,",", max, ',', goal
print "sa,", i,",", max, ',', goal

# run GA
ga = StandardGeneticAlgorithm(200, 100, 25, gap)
max = 0
i = 0
while (max < goal and i < timeout):
    ga.train()
    i += 200
    max = ef.value(ga.getOptimal())
    #print "ga,", i,",", max, ',', goal
print "ga,", i,",", max, ',', goal

# run MIMIC
mimic = MIMIC(200, 100, pop)
max = 0
i = 0
while (max < goal and i < timeout):
    mimic.train()
    i += 200
    max = ef.value(mimic.getOptimal())
    #print "mimic,", i, ",", max, ',', goal
print "mimic,", i, ",", max, ',', goal


