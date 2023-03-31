#    Generates data for the VAE
#    Uses DEAP https://deap.readthedocs.io/en/master/index.html

import random
import argparse
import importlib
import pickle
import time, os
import numpy as np

from deap import base
from deap import creator
from deap import tools

# global
DATA_DIRECTORY = 'data/' # directory storing the output data

#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='enter the name of the file containing all the configuration settings, e.g., enter config to import the file config.py')
parser.add_argument('-s', '--seed', help='enter -1 for a random seed, or enter the seed number. if the argument is not provided, it defaults to -1.')
parser.add_argument('-d', '--debug', action='store_true', help='if argument is used, generate debug info.')
parser.add_argument('-t', '--time', action='store_true', help='if argument is used, calculate run time.')

args = parser.parse_args()

#----------
# Import file containing all the configuration settings
#----------
if args.config:
    file_name = args.config
    config = importlib.__import__(file_name) # import file
else:
    exit("Error. Please specify config filename in the command line. Use --help for more information.")

#----------
# Set seed
#----------
if not args.seed or int(args.seed) < 0: # if args.seed is not provided or is negative
    seed = int(time.time()) # use current time as random seed
else:
    seed = int(args.seed)
print('Seed', seed)
random.seed(seed)

#----------
# Set flags
#----------
DEBUG = args.debug
CALCULATE_RUNTIME = args.time

#----------
# Start set up DEAP
#----------
# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# START_TIMES = []
# # multiplier = DATAGEN_MAX_RANGE
# for i in range(DATAGEN_MAX_RANGE):
#     multiplier = round(2 * (6.9141414141414 - 0.103729603729714 * i))
#     START_TIMES += [i] * multiplier
#     # START_TIMES += [i] * multiplier
#     # # multiplier -= 1
#     # if i < 2:
#     #     START_TIMES += [i] * 100000
#     # elif i < 10:
#     #     START_TIMES += [i] * 10
#     # # elif i < 20:
#     # #     START_TIMES += [i] * 10
#     # else:
#     #     START_TIMES += [i] * 1
#     # START_TIMES += [i] * 1
#     # if i < round(0.2 * NUM_TIME_SLOTS):
#     #     START_TIMES += [i] * multiplier * multiplier
#     #     multiplier -= 1
#     # else:
#     #     START_TIMES += [i] * 1

# print(START_TIMES)
# print(len(START_TIMES))
# exit()


def random_start_time():
    # random_start_time = random.choice(START_TIMES)
    # print(random_start_time)
    # start_time = abs(round(random.gauss(mu=0, sigma=40)))
    # start_time = max()
    # return abs(round(random.gauss(mu=-40, sigma=20)))
    random_number = abs(round(random.gauss(mu=0, sigma=20)))
    if random_number > config.DATAGEN_MAX_RANGE:
        return config.DATAGEN_MAX_RANGE
    elif random_number < config.DATAGEN_MIN_RANGE:
        return config.DATAGEN_MIN_RANGE
    else:
        return random_number
    # return 0


# def random_duration():
    # half_time = round(NUM_TIME_SLOTS/2)

    # return 6
    # return abs(round(random.gauss(mu=half_time, sigma=40)))

# Attribute generator 
#                      define 'attr_int' to be an attribute ('gene')
#                      which corresponds to int sampled uniformly
#                      from the specified range
toolbox.register("attr_int", random.randint, config.DATAGEN_MIN_RANGE, config.DATAGEN_MAX_RANGE)
toolbox.register("attr_skewed_int", random_start_time)
# toolbox.register("attr_int", random_duration)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of integers
# toolbox.register("individual", tools.initRepeat, creator.Individual, 
#     toolbox.attr_int, NUM_VARIABLES)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_skewed_int, toolbox.attr_int), n=config.TOTAL_ROBOTS)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  

#----------
# Operator registration
#----------
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

# register the crossover operator
toolbox.register("mate", tools.cxUniform, indpb=0.05)
toolbox.decorate("mate", checkBounds(config.DATAGEN_MIN_RANGE, config.DATAGEN_MAX_RANGE))

# register a mutation operator
from itertools import repeat

def mutGaussianInt(individual, mu, sigma, indpb):
    """Updated to take individuals consisting of integers"""
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    if not isinstance(mu, base.Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, base.Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            individual[i] += round(random.gauss(m, s))

    return individual,

toolbox.register("mutate", mutGaussianInt, mu=0, sigma=3, indpb=0.2)
toolbox.decorate("mutate", checkBounds(config.DATAGEN_MIN_RANGE, config.DATAGEN_MAX_RANGE))

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------
# register the goal / fitness function
#----------
# # unnormalise each variable in the individual into their original range
# def unnormalise_to_range(individual):
#     result = []
#     for i in range(NUM_VARIABLES):
#         xi_std = (individual[i] - DATAGEN_MIN_RANGE)/(DATAGEN_MAX_RANGE - DATAGEN_MIN_RANGE)
#         # xi_scaled = xi_std * (eq.VARIABLE_RANGE[i]['max'] - eq.VARIABLE_RANGE[i]['min']) + eq.VARIABLE_RANGE[i]['min']
#         xi_scaled = xi_std * (eq.X_MAX_RANGE - eq.X_MIN_RANGE) + eq.X_MIN_RANGE
#         result.append(xi_scaled)
#     return result


def convert_individual_to_time_slots(individual):
    time_slots = []
    total_schedule = [0] * config.NUM_TIME_SLOTS
    for i in range(config.TOTAL_ROBOTS):
        start_time = individual[(i*2)]
        additional_slots = individual[(i*2)+1]
        total_slots = config.MIN_SLOTS + additional_slots
        time_slot = [0] * config.NUM_TIME_SLOTS

        j = start_time
        end_time = start_time + total_slots
        while (j < end_time and j <config.NUM_TIME_SLOTS):
            time_slot[j] = 1
            total_schedule[j] += 1
            j += 1
        # print('robot ', i, ': ', time_slot)
        # print(''.join(map(str, time_slot)))
        # print('start_time', start_time, 'additional_slots', additional_slots, 'total_slots', total_slots)
        time_slots.append(time_slot)

    return time_slots, total_schedule


def evaluate_constraint(individual):
    time_slots, total_schedule = convert_individual_to_time_slots(individual)

    constraint_fitness = 0.0
    for i in range(config.NUM_TIME_SLOTS):
        if total_schedule[i] > config.MAX_RUNNING_ROBOTS:
            constraint_fitness += 1.0

    return constraint_fitness, total_schedule


def evaluate_fitness(individual):
    constraint_fitness, total_schedule = evaluate_constraint(individual)

    maximise_running_slots_of_robots = 100.0/sum(total_schedule)

    fitness = constraint_fitness + maximise_running_slots_of_robots

    return fitness,


toolbox.register("evaluate", evaluate_fitness)

#----------
# End set up DEAP
#----------

def data_generator_ga():

    min_list = []
    max_list = []
    avg_list = []
    std_list = []

    # create an initial population of individuals
    pop = toolbox.population(n=config.DATAGEN_POP)
    
    # for item in pop:
    #     time_slots, total_schedule = convert_individual_to_time_slots(item)
    #     # for slot in time_slots:
    #     #     # print(time_slots)
    #     #     print(''.join(map(str, slot)))
    #     print(','.join(map(str, total_schedule)))
    # exit()

    if DEBUG:
        print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    if DEBUG:
        print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < config.DATAGEN_GEN:
        # A new generation
        g = g + 1

        if DEBUG:
            if (g % 100) == 0:
                print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)
        min_list.append(min(fits))
        max_list.append(max(fits))
        avg_list.append(mean)
        std_list.append(std)

        best_ind_for_gen = tools.selBest(pop, 1)[0]
        # print(best_ind_for_gen)

        # time_slots, total_schedule = convert_individual_to_time_slots(best_ind_for_gen)
        # # for slot in time_slots:
        # #     # print(time_slots)
        # #     print(''.join(map(str, slot)))
        # print(','.join(map(str, total_schedule)))

        fitness_in_gen = best_ind_for_gen.fitness.values[0]
        # print(best_ind_for_gen.fitness)
        # print('best fitness in gen', fitness_in_gen)

        # if fitness_in_gen == 0.0:
        if (evaluate_constraint(best_ind_for_gen)[0] == 0.0): # if constraint is met
            break
    
    if DEBUG:
        print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]

    if DEBUG:
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


    # unnormalised_item = unnormalise_to_range(best_ind)
    # print(unnormalised_item)
    # print("Converted to actual values %s", unnormalised_item)

    # plt.plot(min_list, label='min')
    # plt.plot(max_list, label='max')
    # plt.plot(avg_list, label='avg')
    # plt.plot(std_list, label='std')
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.legend()
    # plt.savefig('output_ga/' + str(seed) + '_ga.png', dpi=72, bbox_inches='tight', pad_inches=0)

    # only return an individual if it meets the constraints

    if (evaluate_constraint(best_ind)[0] == 0.0): # if constraint is met
        return best_ind
    else:
        return None


def run_correction(individual):
    # remove over time such that start_time + additional_slots is always <= total_slots
    corrected_individual = []

    for i in range(config.TOTAL_ROBOTS):
        start_time = individual[(i*2)]
        additional_slots = individual[(i*2)+1]
        if (start_time + additional_slots) > config.DATAGEN_MAX_RANGE:
            additional_slots = config.DATAGEN_MAX_RANGE - start_time
        corrected_individual += [start_time, additional_slots]

    return corrected_individual


def generate_data():
    print("Generating data")

    total = 0
    data = []
    while total < config.NUM_DATA_POINTS:
        valid_data = data_generator_ga()
        if valid_data:
            # valid_data = [59, 6, 60, 6, 61, 6, 0, 70, 7, 70, 1, 70, 60, 6, 61, 6, 0, 70, 7, 70] # test correction
            # print(valid_data)
            corrected_valid_data = run_correction(valid_data)
            # print(corrected_valid_data)
            # time_slots, total_schedule = convert_individual_to_time_slots(corrected_valid_data)
            # for slot in time_slots:
            #     # print(time_slots)
            #     print(''.join(map(str, slot)))
            # print(','.join(map(str, total_schedule)))
            data.append(list(corrected_valid_data)) # VAE takes in a 2D array
            total = total + 1
            if total % 1000 == 0:
                print('Data points generated: %d out of %d' % (total, config.NUM_DATA_POINTS))

    current_file = DATA_DIRECTORY + 't' + str(config.TOTAL_ROBOTS) + '_r' + str(config.MAX_RUNNING_ROBOTS) + '_d' + str(config.NUM_DATA_POINTS) + '.pkl'
    pickle.dump([seed, data], open(current_file, 'wb'))
    print("Seed and data saved to", current_file)


def main():
    # if the directory to save data does not exist, create it
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    #----------
    # Run optimiser
    #----------
    if CALCULATE_RUNTIME:
        start = time.time()

    generate_data()
    
    if CALCULATE_RUNTIME:
        end = time.time()
        total_time = end - start
        if total_time < 60.0:
            unit = "seconds"
        elif total_time < 3600.0:
            total_time = total_time/60.0
            unit = "minutes"
        else:
            total_time = total_time/3600.0
            unit = "hours"
        print("Run time %.2lf " % total_time + unit)
    

if __name__ == "__main__":
    main()