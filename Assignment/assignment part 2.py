# %% md
"""

This is CE310-GP-Mini-Project v2 all data.ipynb but as a .py file instead.

# CE310 Spring 2020/21: Genetic Programming and Symbolic Regression
## Assignment: Programming Assignment and mini project Part 2 (of 2) – Mini project

# Rachel Lowe, 1804170

"""

# %% md
"""
# Some things to point out first

The provided code has been rewritten into a reusable object-oriented test harness, allowing tests for different functions to be run for a variety of parameters, testing 10 30-generation iterations for every permutation of:
* Population
    * 2000
    * 500
* Tournament size
    * 2
    * 5
* Crossover rate
    * 0.3
    * 0.7
* Mutation rate
    * 0.3
    * 0.7

The first few cells of this notebook contain the code for this test harness, and the tests using this harness (as well as the written up results) are below them. Most of the information about the harness itself are present as comments within the code for it.

If you just want the summary (in 498 words), just go to the 'conclusions' at the bottom of the report.
"""
# %%

# Import relevant Python modules
import operator
import math
import random
import numpy  as np
from matplotlib import pyplot
from matplotlib import axes

from statistics import median

from statistics import mean

import typing

from typing import List, Tuple, Callable, Dict, Any, Union

from scipy import stats

from inspect import getsource

# Import DEAP modules
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import faulthandler

faulthandler.enable()

# %%

UseSqError = True  # use Least Squares approach

memorySaving = True  # set this to true if you want to yeet the tests/psets from the global scope after they're done.


# this is mainly intended for memory saving purposes.


# %%

# the protectedDiv function from earlier. returns 1 if trying to divide by 0
def protectedDiv(left, right):
    return left / right if right else 1


# the rand101 lambda from the given pset except it's not a lambda
def rand101() -> int:
    return random.randint(-1, 1)


# a protectedSqrt function. returns 1 if trying to find a square root of a negative number
def protectedSqrt(x: float) -> float:
    if x < 0:
        return 1
    else:
        return math.sqrt(x)


# %%

# this is here mostly because of an ePhEmErAlS nEeD dIfFeReNt NaMeS eVeN iN dIfFeReNt PsEtS
# but then it still didn't work properly but im keeping this here just in case
uniqueNameSuffixCounter = 0


def appendUniqueNameSuffix(appendToThis: str) -> str:
    global uniqueNameSuffixCounter
    theSuffix = str(uniqueNameSuffixCounter)
    theName = appendToThis + "_" + theSuffix
    uniqueNameSuffixCounter += 1
    return theName


# this is the same as the already given primitive set.
# it's here as a fallback just in case something goes wrong and the user doesn't give a primitive set.
# it takes 1 argument and it renames that argument x
if "pset" not in globals():
    pset = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN"), 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    # numerical constants +-1
    pset.addTerminal(1)
    pset.addTerminal(-1)
    # also allowing the GP to use randomness
    pset.addEphemeralConstant(appendUniqueNameSuffix("rand101"), rand101)
    pset.renameArguments(ARG0='x')

# and here's the other stats stuff that was included in the given sample code
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("mdn", np.median)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


# %%

# this class basically holds the info about the populations from a given configuration of parameters.

class PopInfo:
    """
     this class basically holds the info about the populations from a given
     configuration of parameters.
    """

    def __init__(self, label: str, popSize: int, tourn: int, pXO: float, pMut: float, maximization: bool = False):
        """
        constructs this object
        :label: string for labelling the graphs later on
        :popSize: size of the population for this config
        :tourn: tournament size for config
        :pXO: crossover chance for config
        :pMut: mutation rate for config
        :maximization: whether or not this is a maximization problem. true if it is.
        """
        self.pop: int = popSize
        self.tourn: int = tourn
        self.p_xo: float = pXO
        self.p_m: float = pMut
        self.logs: List[tools.Logbook] = []  # logs for each iteration of the config
        self.hofIndividuals = []  # the hall-of-fame individiuals for each iteration of the config
        self.fitnesses: List[float] = []  # fitnesses of aforementioned individuals
        self.fittestCursor: int = 0  # cursor pointing to index of fittest
        self.iterationCount: int = 0  # how many iterations have been done
        self.label = label  # the string with the config info
        self.fittestFitness: float = 0  # how fit the fittest fitness is
        self.maximization = maximization  # whether this is trying to maximize or minimize that fitness
        self.evaluations = 0  # how many evaluations have been done
        self.compiledFunctions = []  # a list of all the compiled fittests
        self.fittestCompiled = lambda x: x
        # this function basically adds the logbook and hall of fame individual from an iteration to the results

    def addTestResults(self, log: tools.Logbook, hallOfFame: tools.HallOfFame, compiledHOF) -> None:
        """
        adds the results from an iteration to the info recorded here
        :param log: the logbook from the iteration
        :param hallOfFame: the HallOfFame from the iteration
        :param compiledHOF: the compiled version of the individual in that HallOfFame
        :return: nothing
        """
        newIndividual = hallOfFame[0]
        newFitness = newIndividual.fitness.values[0]
        # also updates what the fittest result so far is if necessary
        if self.iterationCount > 0:
            if self.maximization:
                if self.fittestFitness < newFitness:  # if it's a maximization problem, we get the smallest fitness
                    self.fittestCursor = self.iterationCount
                    self.fittestFitness = newFitness
                    self.fittestCompiled = compiledHOF
            elif self.fittestFitness > newFitness:  # if it's a minimization problem, we get the largest fitness
                self.fittestCursor = self.iterationCount
                self.fittestFitness = newFitness
                self.fittestCompiled = compiledHOF
        else:
            self.fittestFitness = newFitness
            self.fittestCompiled = compiledHOF
            self.fittestCursor = 0
        self.iterationCount += 1
        self.logs.append(log)
        self.hofIndividuals.append(newIndividual)
        self.fitnesses.append(newFitness)
        self.compiledFunctions.append(compiledHOF)
        self.evaluations += sum(
            log.select("nevals"))  # adds the count of evaluations to the current count of evaluations

    def getLabel(self) -> str:
        """
        :return: returns the label indicating what config this is
        """
        return self.label

    def getFittestFitness(self) -> float:
        """
        :return: returns the fittest fitness value
        """
        return self.fittestFitness

    def getMeanFitness(self) -> float:
        """
        :return: returns the mean of all the fitness values
        """
        return mean(self.fitnesses)

    def getMedianFitness(self) -> float:
        """
        :return: returns the median of all the fitness values
        """
        return median(self.fitnesses)

    def getAverageFitness(self) -> float:
        """
        :return: returns the mean of (the best fitness, mean fitness, median fitness)
        """
        return mean([self.fittestFitness, self.getMeanFitness(), self.getMedianFitness()])

    def getFittest(self):
        """
        :return: returns the fittest individual
        """
        return self.hofIndividuals[self.fittestCursor]

    def fittestResults(self):
        """
        :return: tuple with the logs and the hallOfFame individual for fittest iteration
        """
        return self.logs[self.fittestCursor], self.hofIndividuals[self.fittestCursor]

    def getAllResults(self):
        """
        :return: list of tuples (hall of fame individuals, logs) for each iteration
        """
        results = []
        for i in range(0, self.iterationCount):
            results.append((self.hofIndividuals[i], self.logs[i]))
        return results

    def getNormalizedFitnessVsSizeResults(self):
        """
        :return: a list of tuples with normalized fitnesses and sizes
        """
        res = []
        maxFit = max(max(l.chapters['fitness'].select("mdn")) for l in self.logs)
        maxSize = max(max(l.chapters['size'].select("mdn")) for l in self.logs)

        for i in range(0, self.iterationCount):
            l = self.logs[i]
            fits = l.chapters['fitness'].select("mdn")
            sizes = l.chapters['size'].select("mdn")
            normFits = [(f / maxFit) for f in fits]
            normSizes = [(s / maxSize) for s in sizes]
            res.append((normFits, normSizes))
        return res

    def getAvgFitnessAndSize(self):
        """
        :return: a tuple with the aggregate average fitness and average size for all iterations.
        """
        avgFits = 0
        avgSizes = 0
        for i in range(0, self.iterationCount):
            l = self.logs[i]
            avgFits += l.chapters['fitness'].select('avg')[-1]
            avgSizes += l.chapters['size'].select('avg')[-1]

        return avgFits / self.iterationCount, avgSizes / self.iterationCount


# this is used instead of a lambda method when we need to sort popInfo objects by their fittest fitness.
def sortByFitness(popInfo: PopInfo) -> float:
    return popInfo.fittestFitness


# %%

# given a method taking a float in and returning a float,
# a boundary of points, an offset for that boundary, and a number of points,
# this method basically plots the points for that method.

def generateTestPointsAndTarget(measurement: Callable[[float], float],
                                limit: float = math.pi, offset: float = 0, count: int = 65):
    """

    :param measurement: method taking a float in and returning a float to be plotted
    :param limit: upper/lower bound for the points (defaults to pi)
    :param offset: offset for the boundary (defaults to 0)
    :param count: number of points to plot (defaults to 65)
    :return: tuple with an array of x points in the given range, and another array for the y values of the x points put through the given function
    """
    lowerBound = offset - limit
    upperBound = offset + limit
    testPoints = np.linspace(lowerBound, upperBound, count).tolist()
    target = np.empty(len(testPoints))
    for i in range(len(testPoints)): target[i] = measurement(testPoints[i])
    return testPoints, target


# %%

# This class basically tests a single configuration of parameters for a run.

class ConfigTester:
    """
    A class that basically handles running the GP stuff with a particular configuration
    of parameters.
    """

    def __init__(self, genCount: int, popSize: int, tourn: int, pXO: float, pMut: float,
                 measure: Callable[[float], float], testPoints, target, iterations: int,
                 pset: gp.PrimitiveSet, maximization: bool = False):
        """
        This constructs this object
        :param genCount: total number of generations to use
        :param popSize: size of population
        :param tourn: tournament size
        :param pXO: crossover rate
        :param pMut: mutation rate
        :param measure: method that is being tested
        :param testPoints: points on the x axis that will be put through the measure function
        :param target: points on the y axis that are the results of putting testPoints through the measure
        :param iterations: how many iterations to use
        :param pset: the primitive set to use
        :param maximization: whether this is a maximization of problem
        """
        self.no_generations = genCount
        self.no_population = popSize
        self.no_tournaments = tourn
        self.p_xo = pXO
        self.p_m = pMut
        self.measurement = measure

        self.test_points = testPoints
        self.target = target
        self.toolbox = base.Toolbox()

        self.pset = pset
        self.maximization = maximization

        self.iterations = iterations
        self.theLabel = "pop: " + str(self.no_population) + \
                        ", tourn: " + str(self.no_tournaments) + \
                        ", xo: " + str(self.p_xo) + \
                        ", mut: " + str(self.p_m)

        self.popInfo = PopInfo(self.theLabel, self.no_population, self.no_tournaments, self.p_xo, self.p_m,
                               self.maximization)

    # basically the function that calculates the fitness function for a given individual
    def evalSymbReg(self, individual):

        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)

        if UseSqError:
            # squared error
            error = (abs(func(x) - self.measurement(x)) ** 2 for x in self.test_points)
        else:
            # Absolute distance between target curve and solution
            error = (abs(func(x) - self.measurement(x)) for x in self.test_points)

        return math.fsum(error) / len(self.test_points),

    # runs the genetic programming stuff
    def runTheGP(self) -> None:
        """
        sets up and runs the GP, logging stuff in the popInfo object of this object
        :return: nothing
        """
        # sets up the GP stuff
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.evalSymbReg)
        self.toolbox.register("select", tools.selTournament, tournsize=self.no_tournaments)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=64))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=64))

        random.seed()

        # does the GP stuff for as many iterations as it needs to do, logging the results in the popInfo object.
        for i in range(0, self.iterations):
            print("Starting iteration", str(i))
            pop = self.toolbox.population(n=self.no_population)
            hof = tools.HallOfFame(1)

            # verbose is false, because, if it was set to true, we'd be pretty much out of memory
            pop, log = algorithms.eaSimple(pop, self.toolbox, self.p_xo, self.p_m, self.no_generations,
                                           stats=mstats, halloffame=hof, verbose=False)
            print("Iteration", str(i), "done,", str(sum(log.select("nevals"))), "evaluations.")
            print("\t", hof[0])
            print("\t", hof[0].fitness.values[0])
            self.popInfo.addTestResults(log, hof, self.toolbox.compile(expr=hof[0]))


# %%


"""
These are all the hyper-parameters we will be testing
"""
generations: int = 30  # we're using 30 generations

all_populations: List[int] = [500, 2000]
all_tournaments: List[int] = [2, 5]
all_xo: List[float] = [0.7, 0.3]
all_m: List[float] = [0.3, 0.7]

"""
And these are some dictionaries to hold info about the fitnesses
for every single run with each particular value for these hyper-parameters
"""

allPopFitnesses: Dict[int, List[float]] = {}
# allPopSizes: Dict[int, List[float]] = {}
for p in all_populations:
    allPopFitnesses[p] = []
#    allPopSizes[p] = []
allTournFitnesses: Dict[int, List[float]] = {}
# allTournSizes: Dict[int, List[float]] = {}
for t in all_tournaments:
    allTournFitnesses[t] = []
#    allTournSizes[t] = []
allXoFitnesses: Dict[float, List[float]] = {}
# allXoSizes: Dict[float, List[float]] = {}
for x in all_xo:
    allXoFitnesses[x] = []
#    allXoSizes[x] = []
allMutFitnesses: Dict[float, List[float]] = {}
# allMutSizes: Dict[float, List[float]] = {}
for m in all_m:
    allMutFitnesses[m] = []
#    allMutSizes[m] = []

# basically the headers for each of the tests in the dictionary.
# these are held in the same order that the tests will be put in the
# dictionaries, as they will follow the same overall iterations basically
popHeaders: List[str] = []
tournHeaders: List[str] = []
xoHeaders: List[str] = []
mutHeaders: List[str] = []

# putting in population headers in the right order
for t in all_tournaments:
    tString = ("tourn " + str(t)).ljust(8)
    for x in all_xo:
        xString = ("xo: " + str(x)).ljust(8)
        for m in all_m:
            mString = ("mut: " + str(m)).ljust(8)
            popHeaders.append(tString + " " + xString + " " + mString)

# putting in tournament size headers in the right order
for p in all_populations:
    pString = ("pop " + str(p)).ljust(8)
    for x in all_xo:
        xString = ("xo: " + str(x)).ljust(8)
        for m in all_m:
            mString = ("mut: " + str(m)).ljust(8)
            tournHeaders.append(pString + " " + xString + " " + mString)

# putting in crossover rate headers in the right order
for p in all_populations:
    pString = ("pop " + str(p)).ljust(8)
    for t in all_tournaments:
        tString = ("tourn " + str(t)).ljust(8)
        for m in all_m:
            mString = ("mut: " + str(m)).ljust(8)
            xoHeaders.append(pString + " " + tString + " " + mString)

# putting in mutation rate headers in the right order
for p in all_populations:
    pString = ("pop " + str(p)).ljust(8)
    for t in all_tournaments:
        tString = ("tourn " + str(t)).ljust(8)
        for x in all_xo:
            xString = ("xo: " + str(x)).ljust(8)
            mutHeaders.append(pString + " " + tString + " " + xString)

# set this to true if any tests are going to be performed which are stupid and will pollute the results for all params etc
stupidityAlert: bool = False


class ParameterTester:
    """
    basically runs a given genetic programming problem for various configurations of parameters,
    and also produces graphs and such for it
    """

    def __init__(self, measurementToUse: Callable[[float], float], primSet: gp.PrimitiveSet = pset,
                 maximization: bool = False, iterations: int = 10, xLimit: float = math.pi,
                 xOffset: float = 0, xCount: int = 65):
        """
        This constructs this object
        :param measurementToUse: f(x)->y method being symbolic regression'd
        :param primSet: primitive set to use
        :param maximization: whether this is a maximization of problem or not
        :param iterations: how many iterations to use for each config
        :param xLimit: x bounds for generateTestPointsAndTarget
        :param xOffset: offset for generateTestPointsAndTarget
        :param xCount: count for generateTestPointsAndTarget
        """

        self.measure = measurementToUse

        self.maximization = maximization

        self.stringMeasure = getsource(measurementToUse)

        print("Testing: ")
        print(self.stringMeasure)

        self.iterations = iterations
        (self.test_points, self.target) = generateTestPointsAndTarget(self.measure, xLimit, xOffset, xCount)

        self.targetFig, ax0 = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 9))
        ax0.plot(self.test_points, self.target, linestyle="solid", alpha=0.5)
        ax0.scatter(self.test_points, self.target, label=self.stringMeasure)
        ax0.set_xlabel('Test points')
        ax0.set_ylabel('Measurements')
        ax0.set_title('Data set')
        ax0.grid(b=True, which='major', color='#666666', linestyle='-', axis="y")
        ax0.grid(b=True, which='minor', color='#999999', linestyle='-', axis="y", alpha=0.2)
        ax0.legend()
        self.targetFig.show()

        self.fitSizeFig, self.fitSizePlots = pyplot.subplots(nrows=4, ncols=4, figsize=(48, 27),
                                                             sharex="all", sharey="all")
        self.fitSizeFig.suptitle("Error of result vs result size")

        self.resTargetFig, self.resTargetPlots = pyplot.subplots(nrows=4, ncols=4, figsize=(48, 27),
                                                                 sharex="all", sharey="all")
        self.resTargetFig.suptitle("Result(s) compared to target")

        self.allResults: List[PopInfo] = []

        self.pset = primSet

        self.popFitnesses: Dict[int, List[float]] = {}
        # self.popSizes: Dict[int, List[float]] = {}
        for p in all_populations:
            self.popFitnesses[p] = []
            # self.popSizes[p] = []
        self.tournFitnesses: Dict[int, List[float]] = {}
        # self.tournSizes: Dict[int, List[float]] = {}
        for t in all_tournaments:
            self.tournFitnesses[t] = []
            # self.tournSizes[t] = []
        self.xoFitnesses: Dict[float, List[float]] = {}
        # self.xoSizes: Dict[float, List[float]] = {}
        for x in all_xo:
            self.xoFitnesses[x] = []
            # self.xoSizes[x] = []
        self.mutFitnesses: Dict[float, List[float]] = {}
        # self.mutSizes: Dict[float, List[float]] = {}
        for m in all_m:
            self.mutFitnesses[m] = []
            # self.mutSizes[m] = []

    # runs the GP for every single configuration of parameters
    def runTheTests(self) -> None:
        """
        runs the GP for every single configuration of parameters
        :return: nothing.
        """

        global allPopFitnesses
        global allTournFitnesses
        global allMutFitnesses
        global allXoFitnesses

        xCount = 0
        yCount = 0
        for pop in all_populations:
            for tour in all_tournaments:
                for xo in all_xo:
                    for mut in all_m:

                        print("x" + str(xCount) + ", y" + str(yCount))

                        test = ConfigTester(generations, pop, tour, xo, mut,
                                            self.measure, self.test_points, self.target,
                                            self.iterations, self.pset, self.maximization)
                        print(test.theLabel)
                        test.runTheGP()
                        print("done")
                        self.allResults.append(test.popInfo)

                        self.__plotResults(test.popInfo,
                                           self.fitSizePlots[xCount][yCount],
                                           self.resTargetPlots[xCount][yCount])

                        best = test.popInfo.fittestFitness

                        self.popFitnesses[pop].append(best)
                        self.tournFitnesses[tour].append(best)
                        self.xoFitnesses[xo].append(best)
                        self.mutFitnesses[mut].append(best)

                        # self.popSizes[pop].append(sizes)
                        # self.tournSizes[tour].append(sizes)
                        # self.xoSizes[xo].append(sizes)
                        # self.mutSizes[mut].append(sizes)

                        if not stupidityAlert:
                            allPopFitnesses[pop].append(best)
                            allTournFitnesses[tour].append(best)
                            allXoFitnesses[xo].append(best)
                            allMutFitnesses[mut].append(best)

                            # allPopSizes[pop].append(sizes)
                            # allTournSizes[tour].append(sizes)
                            # allXoSizes[xo].append(sizes)
                            # allMutSizes[mut].append(sizes)

                        yCount += 1
                yCount = 0
                xCount += 1

    # shows the results of the runs after they're done running
    def showResults(self) -> None:
        """
        shows the results of the runs after they're done running
        :return: nothing.
        """
        self.fitSizeFig.show()

        self.resTargetFig.show()

        bestOverallPop = sorted(self.allResults, key=lambda info: info.fittestFitness, reverse=self.maximization)
        bestMeanPop = sorted(self.allResults, key=lambda info: info.getMeanFitness(), reverse=self.maximization)
        bestMedianPop = sorted(self.allResults, key=lambda info: info.getMedianFitness(), reverse=self.maximization)

        bestAvgPop = sorted(self.allResults, key=lambda info: info.getAverageFitness(), reverse=self.maximization)

        evaluationPop = sorted(self.allResults, key=lambda info: info.evaluations)

        theBest = bestOverallPop[0]
        meanBest = bestMeanPop[0]
        medianBest = bestMedianPop[0]

        avgBest = bestAvgPop[0]

        evalBest = evaluationPop[0]

        print("")
        print("")
        print("The best overall result, with an error of", str(theBest.fittestFitness), "was:")
        print(theBest.getFittest())
        print(theBest.label)

        print("\n--")
        print("full overall rankings:")
        for p in bestOverallPop:
            print("\t", p.label)
            print("\t\t", p.fittestFitness)
            print("\t\t", p.getFittest())
            print("---")

        print("")
        print("")
        print("The mean best configuration with a mean fitness of", str(meanBest.getMeanFitness()), "was:")
        print(meanBest.label)

        print("\n--")
        print("full mean rankings:")
        for p in bestMeanPop:
            print("\t", p.label)
            print("\t\t", p.getMeanFitness())
            print("---")

        print("")
        print("")
        print("The median best configuration with a median fitness of", str(medianBest.getMedianFitness()), "was:")
        print(medianBest.label)
        print("\n--")
        print("full median rankings:")
        for p in bestMedianPop:
            print("\t", p.label)
            print("\t\t", p.getMedianFitness())
            print("---")

        print("")
        print("")
        print("And, from the mean of the mean, median, and best fitnesses, the best result is",
              str(avgBest.getAverageFitness()))
        print(avgBest.label)
        print("\n--")
        print("full avg rankings:")
        for p in bestAvgPop:
            print("\t", p.label)
            print("\t\t", p.getAverageFitness())
            print("---")

        print("")
        print("")
        print("And the one with the fewest evaluations, with", str(evalBest.evaluations), "was")
        print(evalBest.label)

        print("\n--")
        print("full evaluation count rankings:")
        for p in evaluationPop:
            print("\t", p.label)
            print("\t\t", p.evaluations)
            print("---")

        print("----")

    def statPrinter(self) -> None:
        """
        prints the pairs of fitness stats for each configuration for each parameter
        idk how to best put it into words but like if we're testing pop 500 vs pop2000,
        it'll print the pop500 and pop2000 fitnesses for a given tourn T, crossover XO, and mut M
        so you can like compare the 500 vs 2000 stuff a bit easier

        but like for every pair of stats

        :return: nothing
        """

        print("Stats for this run")

        print("")

        print("Best fitnesses with each parameter config")
        print("")
        for h in range(0, 8):
            print(popHeaders[h] + ":")
            print("\t 500 : " + str(self.popFitnesses[500][h]))
            # print("\t     s: " + str(self.popSizes[500][h]))
            print("\t2000 : " + str(self.popFitnesses[2000][h]))
        # print("\t     s: " + str(self.popSizes[2000][h]))
        print("")

        print("tournament fitnesses with each parameter config")
        print("")
        for h in range(0, 8):
            print(tournHeaders[h] + ":")
            print("\t 2 : " + str(self.tournFitnesses[2][h]))
            # print("\t   s: " + str(self.tournSizes[2][h]))
            print("\t 5 : " + str(self.tournFitnesses[5][h]))
            # print("\t   s: " + str(self.tournSizes[5][h]))
        print("")

        print("crossover fitnesses with each parameter config")
        print("")
        for h in range(0, 8):
            print(xoHeaders[h] + ":")
            print("\t 0.7 : " + str(self.xoFitnesses[0.7][h]))
            # print("\t     s: " + str(self.xoSizes[0.7][h]))
            print("\t 0.3 : " + str(self.xoFitnesses[0.3][h]))
            # print("\t     s: " + str(self.xoSizes[0.3][h]))
        print("")

        print("mutation fitnesses with each parameter config")
        print("")
        for h in range(0, 8):
            print(mutHeaders[h] + ":")
            print("\t 0.3 f: " + str(self.mutFitnesses[0.3][h]))
            # print("\t     : " + str(self.mutSizes[0.3][h]))
            print("\t 0.7 f: " + str(self.mutFitnesses[0.7][h]))
            # print("\t     : " + str(self.mutSizes[0.7][h]))
        print("")

    def __ttester(self, values: Dict[Union[float, int], List[float]], valTitle: str, maxi: bool = False) -> None:
        """
        performs the T-test on the given dictionary of lists of floats
        :param values: dictionary with lists of values
        :param valTitle: name of the value we're testing
        :return: nothing
        """
        keyList: List[Union[float, int]] = [*values.keys()]

        p1: Union[float, int] = keyList[0]
        p2: Union[float, int] = keyList[1]

        tStat, pVal = stats.ttest_rel(values[p1], values[p2])

        # t = difference
        # p = significance

        best: Union[float, int] = p1

        # if t < 0:
        #     p1 < p2
        # if t > 0:
        #     p1 > p2

        if maxi:
            # if this is a maximization problem, p2 is best if t is negative (p2 bigger).
            if tStat < 0:
                best = p2
        elif tStat > 0:
            # if this is a minimization problem, p2 is best if t is positive (p1 bigger)
            best = p2

        print("The best " + valTitle + " is " + str(best))
        print("\tDifference: " + str(tStat))
        print("\tUncertainty: " + str(pVal))

        if pVal > 0.05:  # less than 95% certainty
            print("This is not a significant difference. (<95%)")
        elif pVal >= 0.01:  # 99-95% certainty
            print("This is a significant difference. (95%<=sig<=99%)")
        else:  # 99% or better certainty
            print("This is a very significant difference. >99%")

    def statTester(self) -> None:
        """
        performs the T-test for all paired parameter config results,
        both for this particular test of them all, and for all that
        have ever happened so far.
        :return: nothing
        """

        print("T-test for all the paired parameter config results.")

        print("")
        print("Tests for this measure")
        print("")

        self.__ttester(self.popFitnesses, "population size", self.maximization)

        # print("")
        # self.__ttester(self.popSizes,"population size sizes", False)
        print("")
        self.__ttester(self.tournFitnesses, "tournament size", self.maximization)

        # print("")
        # self.__ttester(self.tournSizes,"tournament size sizes", False)

        print("")
        self.__ttester(self.xoFitnesses, "crossover rate", self.maximization)
        # print("")
        # self.__ttester(self.xoSizes, "crossover rate sizes", False)

        print("")
        self.__ttester(self.mutFitnesses, "mutation rate", self.maximization)
        # print("")
        # self.__ttester(self.mutSizes, "mutation rate sizes", False)

        print("")
        print("Stats for all runs so far")
        print("")

        print("")

        self.__ttester(allPopFitnesses, "population size", self.maximization)

        # print("")
        # self.__ttester(allPopSizes,"population size sizes", False)
        print("")
        self.__ttester(allTournFitnesses, "tournament size", self.maximization)

        # print("")
        # self.__ttester(allTournSizes,"tournament size sizes", False)

        print("")
        self.__ttester(allXoFitnesses, "crossover rate", self.maximization)
        # print("")
        # self.__ttester(allXoSizes, "crossover rate sizes", False)

        print("")
        self.__ttester(allMutFitnesses, "mutation rate", self.maximization)
        # print("")
        # self.__ttester(allMutSizes, "mutation rate sizes", False)

    # private method to plot the results of a run after every run
    def __plotResults(self, test: PopInfo, fitSizeAx: axes.Axes,
                      bestResultTargetAx: axes.Axes) -> None:
        """
        private method to plot the results of a run after every run
        :param test: popInfo object for this run
        :param fitSizeAx: fitness vs size axes to use
        :param bestResultTargetAx: axes to plot the result vs target stuff on
        :return: nothing
        """
        testLabel = test.getLabel()
        fitSizeAx.set_title(testLabel)

        fitSizeXValues = np.arange(0, generations + 1)

        fitSizeAx.set_ybound(0, 1)
        fitSizeAx.grid(b=True, which='major', color='#666666', linestyle='-', axis="both")
        fitSizeAx.grid(b=True, which='minor', color='#999999', linestyle='-', axis="both", alpha=0.4)

        fitSizeAx.set_ylabel("Normalized error/size")
        fitSizeAx.set_xlabel("Generations")

        fitnessesAndSizes = test.getNormalizedFitnessVsSizeResults()

        for i in range(0, len(fitnessesAndSizes)):
            stringI = str(i)
            lin: List[pyplot.Line2D] = fitSizeAx.plot(fitSizeXValues, fitnessesAndSizes[i][0],
                                                      label="Error (" + stringI + ")")
            colour = lin[0].get_color()  # both lines same colour.
            fitSizeAx.plot(fitSizeXValues, fitnessesAndSizes[i][1], label="Size (" + stringI + ")",
                           dashes=[6, 2], color=colour)

        fitSizeAx.legend()

        bestResultTargetAx.set_title(testLabel)
        bestResultTargetAx.grid(b=True, which='major', color='#666666', linestyle='-', axis="both")
        bestResultTargetAx.grid(b=True, which='minor', color='#999999', linestyle='-', axis="both", alpha=0.4)

        bestResultTargetAx.set_xbound(lower=min(self.test_points), upper=max(self.test_points))
        bestResultTargetAx.set_ylabel("f(x)")
        bestResultTargetAx.set_xlabel("x values")

        bestResultTargetAx.plot(self.test_points, self.target, color="#000000", label="Target")
        pointCount = len(self.test_points)

        print(test.getFittest())
        print(test.getFittestFitness())

        for f in test.compiledFunctions:
            y = np.empty(pointCount)
            for i in range(pointCount): y[i] = f(self.test_points[i])
            bestResultTargetAx.plot(self.test_points, y, linestyle="dashed", alpha=0.5, linewidth=0.9)

        y = np.empty(pointCount)
        for i in range(pointCount): y[i] = test.fittestCompiled(self.test_points[i])
        bestResultTargetAx.plot(self.test_points, y, label=testLabel, color="#bd1e24", linewidth=1.1)

        bestResultTargetAx.legend()

        print("plotted")


# %% md
"""
# Test 1: the example function

This is the function that was provided in the premade jupyter notebook
"""


# %%

# we're testing the example function because why not.

def p0(x: float) -> float:
    return x ** 2 + x - 7 + 3 * math.sin(1 * math.pi * x)


pset0 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_0"), 1)
pset0.addPrimitive(operator.add, 2)
pset0.addPrimitive(operator.sub, 2)
pset0.addPrimitive(operator.mul, 2)
pset0.addPrimitive(protectedDiv, 2)
pset0.addPrimitive(operator.neg, 1)
pset0.addPrimitive(math.cos, 1)
pset0.addPrimitive(math.sin, 1)
# numerical constants +-1
pset0.addTerminal(1)
pset0.addTerminal(-1)
pset0.addTerminal(math.pi)
pset0.addTerminal(-math.pi)
# also allowing the GP to use randomness
pset0.addEphemeralConstant(appendUniqueNameSuffix("rand101_0"), rand101)
pset0.renameArguments(ARG0='x')

theZerothTest: ParameterTester = ParameterTester(p0, primSet=pset0)

theZerothTest.runTheTests()

theZerothTest.showResults()

# %%

theZerothTest.statPrinter()

# %%

theZerothTest.statTester()

# %% md
"""
So far, we can see that a tournament size of 5 appears to be significantly better for the GA than a tournament size of 2, with an uncertainty factor so small it has to go into scientific notation.

There doesn't appear to be a very significant difference for the other hyper-parameters. There's a ~91% significant difference for crossover rate (0.7 better), and a ~89% significant difference for the mutation rate (0.3 better), with only a ~65% significant difference for population size (2000 better).

This is reflected in the graphs as well.

The graphs for a tournament size of 5 appeared to follow the target curve much more smoothly, whilst the tournament size of 2 appeared to lead to a very loose approximation of the curve, not following the bumps and such of the target. Looking at the normalized error rate and size results, the graphs using a tournament size of 2 got stuck at a normalized error of roughly 0.4, whilst the tournament size of 5 was able to get below that threshold.

The normalized error rates for a tournament of 2 might have been stuck at 0.4 because they could have started off lower than with a tournament of 5, meaning that it would be harder for it to decrease to less than 40% of the original accuracy. However, seeing as tournament size has no bearing on the initial population, this theory is unlikely. Additionally, as tournament sizes of 2 appear to produce a general approximation instead of a curve that follows the bumps of the original (like with a tournament of 5), this 0.4 floor could be because the smooth curve was a local minimum for the error rate, and the tournament of 2 prevented the GP from deviating from this local minimum.
"""
# %%

# this is here to save memory by yeeting the last test from the globals
if memorySaving:
    del theZerothTest

    del pset0

# %% md
"""
# Test 2: Function p1(x) from the assignment brief
"""


# %%

# this is the given equation p1(x) from the assignment brief.

def p1(x: float) -> float:
    return (x ** 6) - 2 * (x ** 4) - 13 * (x ** 2)


# it's been given a  somewhat basic primitive set, without any math constants, because those aren't needed.

pset1 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_1"), 1)
pset1.addPrimitive(operator.add, 2)
pset1.addPrimitive(operator.sub, 2)
pset1.addPrimitive(operator.mul, 2)
pset1.addPrimitive(protectedDiv, 2)
pset1.addPrimitive(operator.neg, 1)
# numerical constants +-1
pset1.addTerminal(1)
pset1.addTerminal(-1)
# also allowing the GP to use randomness
# self.pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset1.addEphemeralConstant(appendUniqueNameSuffix("rand101_1"), rand101)
# self.pset.addEphemeralConstant(ephemeral = rand101)
pset1.renameArguments(ARG0='x')

theFirstTest: ParameterTester = ParameterTester(p1, primSet=pset1, iterations=10)

theFirstTest.runTheTests()

theFirstTest.showResults()

# %%

theFirstTest.statPrinter()

# %%

theFirstTest.statTester()

# %% md
"""
Once again, a tournament size of 5 outperformed a tournament size of 2. But, interestingly, a population size of 2000 significantly outperformed a tournament size of 500.

The significance of the tournament size difference (\~98%) was less significant than that of the population size difference (\~99.3%) for this run, and this is reflected in the significances of the differences for all runs. Population size has an overall significance of ~98.5%, whilst tournament size only has an overall significance of ~97.2%

The tournament size difference is still very visible in the normalized error rate/size graphs, with the vast majority of cases with a tournament size of 2 having an error rate that stops decreasing earlier on, whilst a tournament size of 5 appears to allow the error rate to continue to decrease further and for longer.

However, the population difference is most visible in the graphs showing the target curves vs best curves. The population of 500 leads to curves that prematurely curve up at the start/end of the 'bowl' in the graph, whilst a population of 2000 most closely follows these curves.

The best crossover rate appears to be 0.7, albeit with a significance of only ~50%, and the differences between the mutation rate successes only has a significance of ~5%; in other words, there is a ~95% chance that mutation rate doesn't have a significant impact on the performance of the GP, and a ~50% chance that crossover rate also doesn't have an impact.
"""
# %%

# this is here to save memory by yeeting the last test from the globals
if memorySaving:
    del theFirstTest

    del pset1

# %% md
"""
# Test 3: Equation p2(x)
"""


# %%

# equation p2(x) from the assignment brief

def p2(x: float) -> float:
    return math.sin((math.pi / 4) + 3 * x)


pset2 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_2"), 1)
pset2.addPrimitive(operator.add, 2)
pset2.addPrimitive(operator.sub, 2)
pset2.addPrimitive(operator.mul, 2)
pset2.addPrimitive(protectedDiv, 2)
pset2.addPrimitive(operator.neg, 1)
# we're giving the primitive set access to the sine and cosine functions
pset2.addPrimitive(math.cos, 1)
pset2.addPrimitive(math.sin, 1)
# numerical constants +-1
pset2.addTerminal(1)
pset2.addTerminal(-1)
# and we're also letting it use pi
pset2.addTerminal(math.pi)
pset2.addTerminal(-math.pi)
# also allowing the GP to use randomness
# self.pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset2.addEphemeralConstant(appendUniqueNameSuffix("rand101_2"), rand101)
# self.pset.addEphemeralConstant(ephemeral = rand101)
pset2.renameArguments(ARG0='x')

theSecondTest: ParameterTester = ParameterTester(p2, primSet=pset2)

theSecondTest.runTheTests()

theSecondTest.showResults()

# %%

theSecondTest.statPrinter()

# %%

theSecondTest.statTester()

# %% md
"""
Like the first test, the only parameter with a significant impact on the GA's performance was the tournament size, with a tournament size of 5 having a 99% significant positive impact on performance.

None of the others had a significant impact.

Once again, this is reflected in the graphs; the tournament of 2 ended up hitting a normalized error floor at roughly 0.3, with a generally notable difference between the red (best) line and the black (target) line.

However, the running totals tell a slightly different story.

Population size (of 2000) still has the most significant positive improvement (of ~98%), followed by tournament size (5 best, ~97% significance), with crossover rate and mutation rate still being very insignificant.
"""
# %%

# this is here to save memory by yeeting the last test from the globals
if memorySaving:
    del theSecondTest

    del pset2


# %% md

# Test 4: function p3(x)

# %%


# function p3(x) from the assignment brief
def p3(x: float) -> float:
    return p3n(-1.7, 0.5, x) + p3n(1.3, 0.8, x)


# and this is the 'where M(thing, other thing) = ' from p3(x)
def p3n(a: float, b: float, x: float) -> float:
    return (1 / (b * math.sqrt(2 * math.pi))) * math.e ** (-0.5 * ((x - a) / b) ** 2)


pset3 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_3"), 1)
pset3.addPrimitive(operator.add, 2)
pset3.addPrimitive(operator.sub, 2)
pset3.addPrimitive(operator.mul, 2)
pset3.addPrimitive(protectedDiv, 2)
pset3.addPrimitive(operator.neg, 1)
pset3.addPrimitive(protectedSqrt, 1)  # it has access to the protected square root function as well now
pset3.addTerminal(math.pi)  # as well as pi
pset3.addTerminal(-math.pi)
pset3.addTerminal(math.e)  # and the big famous constant e (2.718281828something)
pset3.addTerminal(-math.e)
# numerical constants +-1
pset3.addTerminal(1)
pset3.addTerminal(-1)
# also allowing the GP to use randomness
pset3.addEphemeralConstant(appendUniqueNameSuffix("rand101_3"), rand101)
pset3.renameArguments(ARG0='x')

theThirdTest: ParameterTester = ParameterTester(p3, primSet=pset3)

theThirdTest.runTheTests()

theThirdTest.showResults()

# %%

theThirdTest.statPrinter()

# %%

theThirdTest.statTester()

# %% md
"""
Interestingly, unlike all the other tests, this test indicated that crossover rate does have a significant impact on the GA's performance, showing a crossover rate of 0.3 being better than a crossover rate of 0.7, with significance of just below 99%.

It still shows the same strong significance between population size/tournament size and GP performance, both in this run (significances of >99%), and for all the tests performed so far.

However, the overall stats still show a complete lack of any significance for the mutation rates and crossover rates, even if the crossover rates are significant for this one test.


The normalized error vs size graphs for this test are still rather interesting. Unlike all of the previous runs, there doesn't appear to be a particular 'floor' which some configurations get stuck in; instead, they all appear to reach roughly the same 'floor' as each other.

When looking at the result vs target graphs, they also tell an interesting story. Instead of any of the lines following the curve closely, they are all very clearly approximations. generally attempting to follow the first curve and having the second curve as a straight line. In fact, only the best result, from the configuration `pop: 2000, tourn: 5, xo: 0.3, mut: 0.3`, has a 'double-hump' shape, and none of the other results from that configuration came close. However, this does imply that this result was only achieved through luck for that particular iteration, and not due to any inherent benefits from this particular configuration of hyper-parameters.
"""
# %%

# this is here to save memory by yeeting the last test from the globals
if memorySaving:
    del theThirdTest

    del pset3

# %% md
"""
# Test 5: p(x) = cos(x)

This next test is to see whether or not the GP can reverse-engineer the cosine function
"""


# %%


# attempting to reverse-engineer the cosine function.
def p4(x: float) -> float:
    return math.cos(x)


pset4 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_4"), 1)
pset4.addPrimitive(operator.add, 2)
pset4.addPrimitive(operator.sub, 2)
pset4.addPrimitive(operator.mul, 2)
pset4.addPrimitive(protectedDiv, 2)
pset4.addPrimitive(operator.neg, 1)

# numerical constants +-1
pset4.addTerminal(1)
pset4.addTerminal(-1)
# also allowing the GP to use randomness
pset4.addEphemeralConstant(appendUniqueNameSuffix("rand101_4"), rand101)
pset4.renameArguments(ARG0='x')

# theFourthTest
theFourthTest: ParameterTester = ParameterTester(p4, primSet=pset4)

theFourthTest.runTheTests()

theFourthTest.showResults()

# %%

theFourthTest.statPrinter()

# %%

theFourthTest.statTester()

# %% md
"""
Just like the first and third tests, the only significant improvement in GA performance was with a tournament size of 5, with the improvement it gave having a significance of ~98%, whilst all the difference for all the other parameters was completely insignificant.

Once again, the overall stats indicate that a tournament size of 5 is significantly better than a tournament size of 2, and that a population size of 2000 is even more significantly better than a tournament size of 500, whilst the other hyper-parameters were still ultimately insignificant.

The graphs present a variety of shapes, some of which are more accurate than others. Some particularly interesting shapes noted in there are the lines from the not-best iterations which deviate significantly above/below the target, before suddenly spiking up/down to meet the line at x=0; made even more interesting by the best line from the configuration with `population 500 tournament size 5 crosssover 0.3 mutation rate 0.3`, which follows the target line somewhat closely before suddenly spiking above and away from it around x=0. However, that was the configuration that took the fewest total evaluations to run, so it could be seen as a case of 'you get what you pay for' computationally.
"""
# %%

if memorySaving:
    del theFourthTest

    del pset4

# %% md
"""
# Test 6: Ripple Benchmark

"""


# %%

# basically the ripple benchmark from https://deap.readthedocs.io/en/master/api/benchmarks.html#module-deap.benchmarks.gp
def p5(x: float) -> float:
    return ((x - 3) * (x - 3)) + (2 * math.sin((x - 4) * (x - 4)))


pset5 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_5"), 1)
pset5.addPrimitive(operator.add, 2)
pset5.addPrimitive(operator.sub, 2)
pset5.addPrimitive(operator.mul, 2)
pset5.addPrimitive(operator.neg, 1)
pset5.addPrimitive(math.sin, 1)
# numerical constants +-1
pset5.addTerminal(1)
pset5.addTerminal(-1)
# also allowing the GP to use randomness
# self.pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset5.addEphemeralConstant(appendUniqueNameSuffix("rand101_5"), rand101)
# self.pset.addEphemeralConstant(ephemeral = rand101)
pset5.renameArguments(ARG0='x')

theFifthTest: ParameterTester = ParameterTester(p5, primSet=pset5)

# theFifthTest

theFifthTest.runTheTests()

theFifthTest.showResults()

# %%

theFifthTest.statPrinter()

# %%

theFifthTest.statTester()

# %% md
"""
Just like tests 1, 3, and 5, tournament size was the only hyper-parameter with any significant impact on the results for this test.

Unfortunately, the vast majority of hyper-parameter configurations still lead to the fittest result being a mere approximation of the target points, making a smooth curve from the starting point to the end, instead of following the 'ripples' in-between. Despite this, some of the configurations did still appear to make a token effort at following these ripples; with all of these partially-rippled results coming from configurations with a tournament size of 5.

One thing of note was the configuration `pop: 2000, tourn: 2, xo: 0.7, mut: 0.3`: All 10 iterations for it produced different equations, yet 8 of those iterations produced a best result with an error of 2.070870318353 to 13 significant figures. This could have been because of a low mutation rate causing convergence, however, this doesn't appear to have happened anywhere else in the tests. Additionally, as this had a population of 2000, you would expect to see a wider range of results in the initial population, leading to much less chance of convergence happening. Therefore, it is unclear how this did happen, but it somehow did.
"""
# %%

if memorySaving:
    del theFifthTest

    del pset5

# %% md
"""
# Test 7: p(x) = x

This final test is to see whether or not the GP can reverse-engineer the simple function of p(x) = x, even when it does have a primitive set containing the tools to produce a lot of other expressions.

This isn't intended as a serious test, more as a one-off test to see if the GA can handle stupid problems being thrown at it. I don't expect any sort of important results to come from the statistical tests, therefore, the fitnesses for these tests won't be added to the ongoing totals for fitnesses. However, if this test does result in a noticable difference for the parameter configurations for this run, I shall re-run it, adding the outcomes to the total.
"""


# %%


# yep we're giving the GP a stupid problem to screw around with it.
def p6(x: float) -> float:
    return x


pset6 = gp.PrimitiveSet(appendUniqueNameSuffix("MAIN_6"), 1)
pset6.addPrimitive(operator.add, 2)
pset6.addPrimitive(operator.sub, 2)
pset6.addPrimitive(operator.mul, 2)
pset6.addPrimitive(protectedDiv, 2)
pset6.addPrimitive(operator.neg, 1)
pset6.addPrimitive(math.sin, 1)
pset6.addPrimitive(math.cos, 1)
# numerical constants +-1
pset6.addTerminal(1)
pset6.addTerminal(-1)
pset6.addTerminal(math.pi)
pset6.addTerminal(math.e)
# also allowing the GP to use randomness
# self.pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset6.addEphemeralConstant(appendUniqueNameSuffix("rand101_6"), rand101)
# self.pset.addEphemeralConstant(ephemeral = rand101)
pset6.renameArguments(ARG0='x')

# yep, this is a stupid test. so we're going to avoid polluting the overall results with this act of stupidity.
stupidityAlert = True

theSixthTest: ParameterTester = ParameterTester(p6, primSet=pset6)

theSixthTest.runTheTests()

theSixthTest.showResults()

# %%

theSixthTest.statPrinter()

# %%

theSixthTest.statTester()

# %% md
"""
To nobody's surprise, every single iteration of this produced a version of `f(x) = x`, with an error rate of 0.

This also meant that the statistical tests automatically run for this iteration failed, because there was no difference to find and evaluate the significance of. Therefore, I shall not be re-running this test with `stupidityAlert` disabled, as it would not provide any useful information for the overall statistics for all tests.

No iterations for any configurations were able to return `f(x) = x` directly, instead, all of them had to put `x` through at least one operator. I was unable to find anything in the DEAP documentation to explicitly confirm or deny this, however, there may be a requirement within DEAP for all tree-based individuals to hold at least one non-terminal operator within them when the primitive set contains terminals and non-terminals, leading to these somewhat convoluted methods of returning `x`.
"""
# %%

if memorySaving:
    del theSixthTest

    del pset6

# %% md
"""
# Conclusions (The full report within 500 words)

From the stats, it appears that population size and tournament size have very significant impacts on GA performance, whilst crossover rates and mutation rates have a very insignificant impact.

The best population size is 2000, not 500, with a population of 2000 reducing error rates by 2.4293308997848957 compared to a population size of 500, with a significance of 98.0997022723%. This may be because the larger population allows a larger search space to be explored (due to more individuals existing), thereby reducing the likelihood of the GP getting caught in a local maximum/minimum. 

The best tournament size is 5, not 2, with a tournament size of 5 reducing error rates by 2.2092418704649415 compared to a tournament size of 2, with a significance of 96.7932444531%. This is probably due to the larger tournament size making it more likely that a good result from a list will be picked, as fewer weaker individuals are able to get picked.

The insignificance for crossover/mutation may be due to the inherent randomness of crossover and mutation. In short, just because a particular reproduction strategy chance is used, that doesn't necessarily mean that the reproduction strategy will be used. Additionally, the usage of the reproduction strategy doesn't have any bearing on success, as they both have some element of randomness. Crossover does not guarantee that suitable parents and suitable sub-trees from those parents will be used, and mutation does not guarantee that an appropriate new sub-tree will be generated at a suitable position in the muted individual. Therefore, because of this lack of inherent advantange for these reproduction strategy usage chances, the differences in the fitnesses from configurations with differences in these hyper-parameters is likely to be insignificant.

Therefore, the best configuration to use is:

* Population of 2000
    * **Statistically significant**
        * uncertainty of 0.019002977276693627
* Tournament size of 5
    * **Statistically significant**
        * uncertainty of 0.03206755546894427
* Crossover rate of ~~0.7~~
    * **Statistically insignificant**
        * uncertainty of 0.4744135041601194
* Mutation rate of ~~0.7~~
    * **Statistically insignificant**
        * uncertainty of 0.9468722838494585



Unfortunately, I will admit that there are still some problems with this conclusion.

For example, this conclusion was found out from running the *Dependent t-test for paired samples* on only the fitness of the __best__ result from the 10 iterations for each configuration, meaning that, if one 'best' result was anomalously good (such as the best overall result from test 4), that would skew the results considerably.

I should have used the fitnesses from the results of all 10 iterations for each config for each function instead, so, if any such anomalies did happen, their could have be balanced out by the non-anomalous majority of results.

Additionally, I did not consider the size of the individuals produced by the GP as well, meaning that the optimal configuration identified may produce bloated, inefficient, results. I also didn't properly look at the total number of evaluations used to produce each of the individuals, therefore, I might have selected an incredibly inefficient configuration as the best one.

However, the stats do indicate that this configuration should be, at very least, 'good enough'.
"""