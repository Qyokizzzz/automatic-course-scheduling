__author__ = 'Ldaze'
# To be improved: Should support a grade at the same time.
# This project will be optimized in the future.

import random
import numpy
import copy


def lesson_initialization(lessonQuantity):
    """Input lessons and distribute a number for every lesson."""
    lessons = []
    lessonNum = []
    for i in range(lessonQuantity):
        tmp = input('Please input lesson:')
        lessons.append(tmp)
        while 1:
            tmp = random.randrange(lessonQuantity)
            if tmp not in lessonNum:
                lessonNum.append(tmp)
                break
            else:
                continue
    catalog = dict(zip(lessonNum,lessons))
    return catalog


def lesson_initialization2(lessonQuantity, *args):
    """Input lessons and distribute a number for every lesson."""
    lessons = list(args)[0]
    lessonNum = [ i for i in range(0,lessonQuantity)]
    catalog = dict(zip(lessonNum,lessons))
    return catalog


def translate(chromosome, catalog):
    """Visualizing gene coding."""
    timeTable = [[] for i in range(0,len(chromosome))]
    tmp = []
    for i,v in enumerate(chromosome):
        for j in v:
            tmp.append(catalog.get(j))
        timeTable[i] = copy.deepcopy(tmp)
        tmp.clear()
    return timeTable


def Chromosome(workdays, times, catalog):
    """
    Initialize a chromosome.
    :param workdays: Teaching days required in a week.
    :param times: Teaching arrangement in a day.
    """
    num = len(catalog)
    chromosome = [0 for i in range(workdays)]
    gene = [0 for i in range(times)]
    for i in range(workdays):
        for j in range(times):
            gene[j] = random.randrange(num)
        chromosome[i] = copy.deepcopy(gene)
    return chromosome


def species_origin(commNum, *args):
    '''
    Initialize a original species.
    :param commNum: Define the number of species.
    '''
    species = [0 for i in range(commNum)]
    for i in range(commNum):
        species[i] = Chromosome(*args)
    species = numpy.array(species)
    return species


def fitness(chromosome, lessonQuantity):
    """Evaluation adaptability."""

    score = 0
    for i,v in enumerate(chromosome):
        for j in range(0,lessonQuantity):
            if numpy.sum(v == j) <= 1:
                score += 80
            elif numpy.sum(v == j)>1 and numpy.sum(v == j)<=3:
                score -= 60
            elif numpy.sum(v == j)>2 and numpy.sum(v == j)<=4:
                score -= 80
            elif numpy.sum(v == j)>3 and numpy.sum(v == j)<=5:
                score -= 100
            else:
                score -= 800
        if i < len(chromosome) - 1:
            for k in v:
                if k not in chromosome[i+1]:
                    score += 4
                else:
                    score -= 4
    return score


def unifie(scores, species):
    """Delete every negative score."""
    for i,v in enumerate(scores):
        if v < 0:
            del scores[i]
            species = numpy.delete(species, i, axis=0)
    return species


def sum(scores):
    """Sum of the scores."""
    total = 0
    for i in scores:
        total += i
    return total


def cumsum(pro):
    """Divide intervals for roulette.
       example：[0.1,0.2,0.15,0.3,0.25]
                [0.1,0.3,0.45,0.75,1]
                [0,0.1,0.3,0.45,0.75,1]
    """
    intervals = []
    for i, v in enumerate(pro):
        if i == 0:
            intervals.append(v)
        else:
            intervals.append(v + intervals[i-1])
    intervals.insert(0,0)
    return intervals


def select(scores, species):
    """Select individuals with high adaptability. """
    total = sum(scores)
    probabilities = []
    for i in range(len(scores)):
        probabilities.append(scores[i] / total)
    intervals = cumsum(probabilities)
    new_species = []
    con = 0
    while con < len(species):
        rand = random.random()
        for i, v in enumerate(intervals):
            if rand < v and rand >= intervals[i - 1]:
                if len(new_species) == 0:
                    new_species = numpy.array([species[i - 1]])
                else:
                    new_species = numpy.append(new_species,[species[i - 1]],axis=0)
        con += 1
    return new_species


def arb_hybridization(species):
    """Born new individual."""
    tmp1 = random.choice(species)
    tmp2 = random.choice(species)
    rand = random.randrange(0, len(species[0]))
    child1 = numpy.vstack((tmp1[0:rand], tmp2[rand:len(species[0])]))
    # child2 = numpy.vstack((tmp1[rand:len(species[0])], tmp2[0:rand]))
    # new_species = numpy.vstack(([child1],[child2]))
    # return new_species
    return child1


def delete(species, index):
    """Delete the chromosome in species."""
    if index == 0:
        return numpy.array(species[1:len(species)])
    else:
        tmp1 = numpy.array(species[0:index])
        tmp2 = numpy.array(species[index + 1:len(species)])
    return numpy.append(tmp1,tmp2,axis=0)


def select_bestIndex(scores):
    """Select index of the best individual in the species."""
    max_score = max(scores)
    best_Index = scores.index(max_score)
    return best_Index


def select_best_species(species, scores, n):
    """Select excellent individuals."""
    tmp_scores = scores.copy()
    tmp_species = numpy.array(species)
    index = select_bestIndex(tmp_scores)
    exce_species = numpy.array([tmp_species[index]])
    tmp_species = delete(tmp_species, index)
    del tmp_scores[index]
    for i in range(0,n-1):
        index = select_bestIndex(tmp_scores)
        tmp = numpy.array([tmp_species[index]])
        exce_species = numpy.append(exce_species,tmp,axis = 0)
        tmp_species = delete(tmp_species, index)
        del tmp_scores[index]
    return exce_species


def crossover(species, scores, n):
    """
    Cross combination.
    :param n: Retain several excellent chromosomes.
    """
    num = len(species)
    new_species = numpy.array(select_best_species(species,scores,n))# Retain several excellent chromosomes
    # new_species = numpy.array(select_best(species,scores))
    for i in range(0,num-n):
        new_species = numpy.append(new_species,[arb_hybridization(species)],axis=0)
    return new_species


def vary(species, lessonQuantity, iter = 40, p = 0.1,):
    """
    Produce variation.
    :param p: Probability threshold of variation.
    """
    c = 0
    while c<iter:
        rand = random.random()
        if rand <= p:
            rand1 = random.randrange(0,len(species))
            rand2 = random.randrange(0,len(species[0]))
            rand3 = random.randrange(0, len(species[0][0]))
            species[rand1][rand2][rand3] = random.randrange(0,lessonQuantity)
        c += 1


def iteration(lessonQuantity, species, threshold, n=1):
    """
    Genetic iteration.
    :param lessonQuantity: The quantity of lessons.
    :param threshold: Fitness threshold.
    :param n: Retain several excellent chromosomes.
    :return: The best species.
    """
    score = []
    scores = []
    while 1:
        for i in species:
            scores.append(fitness(i,lessonQuantity))
        species = unifie(scores, species)
        species = select(scores, species)
        species = crossover(species,scores,n)
        vary(species, lessonQuantity)
        for i in species:
            score.append(fitness(i,lessonQuantity))
        # print(score)
        for i, v in enumerate(score):
            if v >= threshold:
                return species[i]
        scores.clear()
        score.clear()


def run(lessonQuantity, workdays, times, commNum, threshold=2000, n=10):
    """
    Run the program.
    :param lessonQuantity: The quantity of lessons.
    :param workdays: Teaching days required in a week.
    :param times: Teaching arrangement in a day.
    :param commNum: Initial number of original species.
    :param threshold: Fitness threshold.
    :param n: Retain several excellent chromosomes.
    """
    lessons = lesson_initialization(lessonQuantity)
    species = species_origin(commNum, workdays, times, lessons)
    best_chromosome = iteration(lessonQuantity, species, threshold, n)
    time_table = translate(best_chromosome,lessons)
    print(lessons)
    print(best_chromosome)
    for i in time_table:
        print(i)


if __name__ == '__main__':
    try:
        run(6, 5, 6, 50, 2200)  # 2220
    except (IndexError, ValueError):
        print('The species have already been died out.')
    # dic = ['数学','物理','英语','化学','语文','生物']
    # info = lesson_initialization2(6, dic)
    # lessons = lesson_initialization(6)
    # print(lessons)
