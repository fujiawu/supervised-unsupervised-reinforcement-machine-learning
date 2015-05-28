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
import opt.OptimizationAlgorithm as OptimizationAlgorithm
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
import opt.example.KColorEvaluationFunction as KColorEvaluationFunction
import opt.example.NeuralNetworkOptimizationProblem as NeuralNetworkOptimizationProblem
import func.nn.backprop.BackPropagationNetworkFactory as BackPropagationNetworkFactory
import func.nn.backprop.BackPropagationNetwork as BackPropagationNetwork
import func.nn.backprop.BatchBackPropagationTrainer as BatchBackPropagationTrainer
import func.nn.backprop.RPROPUpdateRule as RPROPUpdateRule
import shared.SumOfSquaresError as SumOfSquaresError
import shared.DataSet as DataSet
import shared.Instance as Instance
import shared.ConvergenceTrainer as ConvergenceTrainer
from array import array
import random


"""
Commandline parameter(s):
    none
"""

# dictionaries for parsing inputs in car data
dicts = []
dict = {"vhigh":float(1), "high":float(2)/3, "med":float(1)/3, "low": float(0)}
dicts.append(dict)
dicts.append(dict)
dict = {"5more":float(1), "4":float(2)/3, "3":float(1)/3, "2":float(0)}
dicts.append(dict)
dict = {"more":float(1), "4":float(0.5), "2":float(0)}
dicts.append(dict)
dict = {"big":float(1), "med":float(0.5), "small":float(0)}
dicts.append(dict)
dict = {"high":float(1), "med":float(0.5), "low":float(0)}
dicts.append(dict)
dict = {"vgood":[1,0,0,0], "good":[0,1,0,0], "acc":[0,0,1,0], "unacc":[0,0,0,1]}
dicts.append(dict)

# helper function for input parsing
def parse2double(line,dicts):
    line = line.strip().split(',')
    data = []
    for i, item in enumerate(line):
       data.append(dicts[i][item])
    return data

# helper function for multi-class problem
def getMaxIndex(instance):
    max = 0
    index = -1
    for i in range(instance.size()):
       if instance.getContinuous(i) > max:
           max = instance.getContinuous(i)
           index = i
    return index
   
# helper function to calculate error rate
def errorRate(network, instances):
    correct = 0
    wrong  = 0
    for i in instances:
        network.setInputValues(i.getData())
        network.run()
        if getMaxIndex(i.getLabel()) == network.getOutputValues().argMax():
           correct += 1
        else:
           wrong += 1
    return wrong/float(correct+wrong)

# helper function to calculate confusion matrix
def confusionMatrix(network, instances):
    N = instances[0].getLabel().size()
    matrix = [ [0 for x in range(N)] for x in range(N)]
    for i in instances:
        network.setInputValues(i.getData())
        network.run()
        matrix[getMaxIndex(i.getLabel())][network.getOutputValues().argMax()] += 1
    #for i,m in enumerate(matrix):
    #    for j,n in enumerate(m):
    #        matrix[i][j] = float(n)/len(instances)
    return matrix

# read inputs from car data and parse to matrix
# filename = "jython/car.data"
# f = open(filename, 'r')
out = sys.stdin.readlines()
#f.close()
#attributes = [[[] for x in range(2)] for line in out]
instances = [[] for line in out]
for i,line in enumerate(out):
    data = parse2double(line,dicts)
    instances[i] = Instance(data[0:6])
    instances[i].setLabel(Instance(data[6]))

# split the data
split_ratio = 0.3
indices = range(len(instances))
random.shuffle(indices)
separator = int(len(instances) * split_ratio)
test_indices = indices[:separator]
train_indices = indices[separator:]
train = [instances[i] for i in train_indices]
test = [instances[i] for i in test_indices]


# set the training set size
tpercent = 1;
tsize = int(tpercent*len(train))
print "\nTraining Set Size:", tsize
train = train[:tsize]

# set up the structure of neural network
factory = BackPropagationNetworkFactory()
measure = SumOfSquaresError()
set = DataSet(train)
inputLayer = 6
hiddenLayer = 10
outputLayer = 4

# training parameters
it_rhc = 1000
it_sa = it_rhc
it_ga = 100

# learn weigths with back propagation
network_bp = factory.createClassificationNetwork([inputLayer, hiddenLayer, outputLayer])
bp = BatchBackPropagationTrainer(set, network_bp, measure, RPROPUpdateRule())
cvt = ConvergenceTrainer(bp)
cvt.train()
print "\nBP training error:", errorRate(network_bp, train)
print "BP training confusion matrix:", confusionMatrix(network_bp, train)
print "    BP test error:", errorRate(network_bp, test)
print "    BP test confusion matrix:", confusionMatrix(network_bp, test)

# learn weights with randomized hill climbing
network_rhc = factory.createClassificationNetwork([inputLayer, hiddenLayer, outputLayer])
nnop_rhc = NeuralNetworkOptimizationProblem(set, network_rhc, measure)
rhc = RandomizedHillClimbing(nnop_rhc)
fit = FixedIterationTrainer(rhc, it_rhc)
fit.train()
op = rhc.getOptimal();
network_rhc.setWeights(op.getData())
print "\nRHC training error:", errorRate(network_rhc, train)
print "RHC training confusion matrix:", confusionMatrix(network_rhc, train)
print "    RHC test error:", errorRate(network_rhc, test)
print "    RHC test confusion matrix:", confusionMatrix(network_rhc, test)

# learn weights with simulated annealing
network_sa = factory.createClassificationNetwork([inputLayer, hiddenLayer, outputLayer])
nnop_sa = NeuralNetworkOptimizationProblem(set, network_sa, measure)
sa = SimulatedAnnealing(1E11, 0.95, nnop_sa)
fit = FixedIterationTrainer(sa, it_sa)
fit.train()
op = sa.getOptimal();
network_sa.setWeights(op.getData())
print "\nSA training error:", errorRate(network_sa, train)
print "SA training confusion matrix:", confusionMatrix(network_sa, train)
print "    SA test error:", errorRate(network_sa, test)
print "    SA test confusion matrix:", confusionMatrix(network_sa, test)

exit()

# learn weights with generic algorithms
network_ga = factory.createClassificationNetwork([inputLayer, hiddenLayer, outputLayer])
nnop_ga = NeuralNetworkOptimizationProblem(set, network_ga, measure)
ga = StandardGeneticAlgorithm(200, 100, 10, nnop_ga)
fit = FixedIterationTrainer(ga, it_ga)
fit.train()
op = ga.getOptimal();
network_ga.setWeights(op.getData())
print "\nGA training error:", errorRate(network_ga, train)
print "GA training confusion matrix:", confusionMatrix(network_ga, train)
print "    GA test error:", errorRate(network_ga, test)
print "    GA test confusion matrix:", confusionMatrix(network_ga, test)

