#    Comparison GA
#    Uses DEAP https://deap.readthedocs.io/en/master/index.html

import importlib
import argparse
import random
import pickle
import numpy as np
import time, os

from deap import base
from deap import creator
from deap import tools

from matplotlib import pyplot as plt

# import scheduler
from scheduler import scheduler

# directories
RESULTS_DIRECTORY = 'results/'
# IMAGE_DIRECTORY = 'image/'

# image settings
# IMAGE_TYPE = 'pdf'
# TITLE_FONT_SIZE = 18
# LEGEND_FONT_SIZE = 14
# AXIS_FONT_SIZE = 14
#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='enter the name of the file containing all the configuration settings, e.g., enter config to import the file config.py')
parser.add_argument('-r', '--runs', help='enter the number of runs. if the argument is not provided, it defaults to 1 run')
parser.add_argument('-s', '--seed', help='enter -1 for a random seed, or enter the seed number. if the argument is not provided, it defaults to -1.')
parser.add_argument('-d', '--debug', action='store_true', help='if argument is used, generate debug info.')
# parser.add_argument('-i', '--image', action='store_true', help='if argument is used, a GA image is generated for each run.')
parser.add_argument('-t', '--time', action='store_true', help='if argument is used, calculate run time.')

args = parser.parse_args()

REQUEST_LIST = []

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
# Set number of runs
#----------
if args.runs:
    NUM_RUNS = int(args.runs)
else:
    NUM_RUNS = 1

DEBUG = args.debug
# GENERATE_IMAGE = args.image
CALCULATE_RUNTIME = args.time

#----------
# Start set up DEAP
#----------
# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

GA_MIN_RANGE = -1.0
GA_MAX_RANGE = 1.0

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

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


# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to real numbers sampled uniformly
#                      from the specified range
# toolbox.register("attr_float", random.uniform, GA_MIN_RANGE, GA_MAX_RANGE)
toolbox.register("attr_int", random.randint, config.DATAGEN_MIN_RANGE, config.DATAGEN_MAX_RANGE)
toolbox.register("attr_skewed_int", random_start_time)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of floats
# toolbox.register("individual", tools.initRepeat, creator.Individual, 
    # toolbox.attr_float, GA_NUM_INDIVIDUALS)
toolbox.register("individual", tools.initCycle, creator.Individual,
             (toolbox.attr_skewed_int, toolbox.attr_int), n=config.TOTAL_ROBOTS)

# define the population to be a list of individuals
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
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
# toolbox.decorate("mate", checkBounds(GA_MIN_RANGE, GA_MAX_RANGE))
toolbox.decorate("mate", checkBounds(config.DATAGEN_MIN_RANGE, config.DATAGEN_MAX_RANGE))

# register a mutation operator
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
# toolbox.decorate("mutate", checkBounds(GA_MIN_RANGE, GA_MAX_RANGE))


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
# Here we use our own custom developed fitness function

# unnormalise each variable in the individual into their original range
# def unnormalise_to_range(individual):
#     result = []
#     for i in range(NUM_VARIABLES):
#         xi_std = (individual[i] - GA_MIN_RANGE)/(GA_MAX_RANGE - GA_MIN_RANGE)
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
        while (j < end_time and j < config.NUM_TIME_SLOTS):
            time_slot[j] = 1
            total_schedule[j] += 1
            j += 1
        # print('robot ', i, ': ', time_slot)
        # print(''.join(map(str, time_slot)))
        # print('start_time', start_time, 'additional_slots', additional_slots, 'total_slots', total_slots)
        time_slots.append(time_slot)

    return time_slots, total_schedule


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


# def evaluate_constraint(individual):
#     time_slots, total_schedule = convert_individual_to_time_slots(individual)

#     constraint_fitness = 0.0
#     for i in range(config.NUM_TIME_SLOTS):
#         if total_schedule[i] > config.MAX_RUNNING_ROBOTS:
#             constraint_fitness += 1.0

#     return constraint_fitness, total_schedule

def evaluate_constraint(total_schedule):
    constraint_fitness = 0.0
    for i in range(config.NUM_TIME_SLOTS):
        if total_schedule[i] > config.MAX_RUNNING_ROBOTS:
            constraint_fitness += 1.0

    return constraint_fitness



# def evaluate_fitness(individual):
#     constraint_fitness, total_schedule = evaluate_constraint(individual)

#     use_as_many_robots_as_possible = 100.0/sum(total_schedule)

#     fitness = constraint_fitness + use_as_many_robots_as_possible

#     return fitness,


def get_actual_timeslots(time_slots, schedule):
    actual_time_slots = []
    actual_schedule = [0] * config.NUM_TIME_SLOTS
    for i in range(config.TOTAL_ROBOTS):
        start_time = time_slots[i].index(1) # the index where time_slot first becomes 1
        total_slots = np.ceil((schedule['robots'][i]['original_duration'] - schedule['robots'][i]['remaining_duration'])/config.SLOT_IN_MINUTES)
        time_slot = [0] * config.NUM_TIME_SLOTS

        j = start_time
        end_time = start_time + total_slots
        while (j < end_time and j < config.NUM_TIME_SLOTS):
            time_slot[j] = 1
            actual_schedule[j] += 1
            j += 1

        actual_time_slots.append(time_slot)

    return actual_time_slots, actual_schedule


def calculate_objective_and_constraints(individual, verbose=False):
    requests = []
    robots = []

    # make a list of requests following REQUEST_LIST to work on
    for i, duration in enumerate(REQUEST_LIST):
        requests.append({'id': i, 'duration': duration, 'scheduled': False, 'robot_id': None})

    corrected_individual = run_correction(individual)
    time_slots, total_schedule = convert_individual_to_time_slots(corrected_individual)

    for robot_id, slot in enumerate(time_slots):
        robots.append({'id': robot_id, 'original_duration': sum(slot)*config.SLOT_IN_MINUTES, 'remaining_duration': sum(slot)*config.SLOT_IN_MINUTES})

    schedule = scheduler(requests, robots, verbose)

    actual_time_slots, actual_schedule = get_actual_timeslots(time_slots, schedule)

    if DEBUG:
        for time_slot in time_slots:
            print(''.join(map(str, time_slot)))
        print("========")
        print(schedule['robots'])
        print("========")
        for time_slot in actual_time_slots:
            print(''.join(map(str, time_slot)))
        print("========")

    constraint = evaluate_constraint(actual_schedule)

    result = {'obj': schedule['total_requests_met'], 'constraint': evaluate_constraint(total_schedule), 'scheduled_constraint': evaluate_constraint(actual_schedule)}

    if verbose:
        result['requests'] = schedule['requests']
        result['robots'] = schedule['robots']
        result['time_slots'] = time_slots
        result['total_schedule'] = total_schedule
        result['corrected_individual'] = corrected_individual
        # result['actual_time_slots'] = corrected_individual
        # result['actual_schedule'] = corrected_individual
        result['actual_time_slots'] = actual_time_slots
        result['actual_schedule'] = actual_schedule

    return result


def eval_winner_for_func(duel0, duel1):
    if (config.MIN_OR_MAX_FLAG == 0): # minimise
        if (duel0.calculated_values['obj'] < duel1.calculated_values['obj']):
            return True
        else:
            return False
    else: # maximise
        if (duel0.calculated_values['obj'] > duel1.calculated_values['obj']):
            return True
        else:
            return False


def play(duel):
    if duel[0].calculated_values['obj'] == duel[1].calculated_values['obj']: 
        duel[0].gathered_score += 1
        duel[1].gathered_score += 1
    elif eval_winner_for_func(duel[0], duel[1]):
        duel[0].gathered_score += 1
    else:
        duel[1].gathered_score += 1

    duel0_constraint_performance = duel[0].calculated_values['constraint']
    duel1_constraint_performance = duel[1].calculated_values['constraint']

    if (duel0_constraint_performance != 0.0) and (duel1_constraint_performance != 0.0): # both wrong
        if duel0_constraint_performance < duel1_constraint_performance:
            duel[0].gathered_score += 1
        else:
            duel[1].gathered_score += 1
    else:
        if (duel0_constraint_performance == 0.0):
            duel[0].gathered_score += 1

        if (duel1_constraint_performance == 0.0):
            duel[1].gathered_score += 1

    duel[0].num_matches += 1
    duel[1].num_matches += 1


def eval_objective_function(item):
    if (item.num_matches == 0):
        return 0
    else:
        return item.gathered_score/item.num_matches

#----------
# End set up DEAP
#----------
def pure_ga(run):
    # if GENERATE_IMAGE:
    #     min_list = []
    #     max_list = []
    #     avg_list = []
    #     std_list = []
    #     avg_obj_list = []
    #     avg_dist_from_constraint = []

    # create an initial population of 200 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=config.GA_POP)

    if DEBUG:
        print("Start of evolution")

    for ind in pop:
        # calculate objective to produce a list
        ind.calculated_values = calculate_objective_and_constraints(ind)
        ind.gathered_score = 0
        ind.num_matches = 0

    for ind in pop:
        participants = random.sample(pop, 5)
        for t in range(10): # 10 tournaments
            duel = random.sample(participants, 2)
            play(duel)

    for ind in pop:
        ind.fitness.values = (eval_objective_function(ind),)

    if DEBUG:
        print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < config.GA_GEN:
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

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values


        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            del mutant.fitness.values # make all invalid

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # calculate objective value and constraints values for each individual
        for ind in offspring:
            ind.calculated_values = calculate_objective_and_constraints(ind)
            ind.gathered_score = 0
            ind.num_matches = 0

        for x in offspring:
            participants = random.sample(offspring, 5)

            for t in range(10): # 10 tournaments
                duel = random.sample(participants, 2)
                play(duel)


        for ind in offspring:
            ind.fitness.values = (eval_objective_function(ind),)

        # print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        if DEBUG:
            best_ind_for_gen = tools.selBest(pop, 1)[0]
            result = calculate_objective_and_constraints(best_ind_for_gen)
            print(result['obj'], ",", result['constraint'])

        # if GENERATE_IMAGE:
        #     distance_from_answer = []
        #     distance_from_constraint = []

        #     for ind in pop:
        #         unnormalised_item = unnormalise_to_range(ind)
        #         performance = eq.func(unnormalised_item, config.NUM_VARIABLES)
        #         distance_from_answer.append(np.abs(ANSWER - performance))
        #         for constraint in eq.CONSTRAINTS:
        #             distance_from_constraint.append(constraint(unnormalised_item, config.NUM_VARIABLES)) # add all the distances from all constraints
            
        #     length = len(pop)
        #     mean = sum(fits) / length
        #     sum2 = sum(x*x for x in fits)
        #     std = abs(sum2 / length - mean**2)**0.5
            
        #     # print("  Min %s" % min(fits))
        #     # print("  Max %s" % max(fits))
        #     # print("  Avg %s" % mean)
        #     # print("  Std %s" % std)
        #     # print(a)
        #     min_list.append(min(fits))
        #     max_list.append(max(fits))
        #     avg_list.append(mean)
        #     avg_obj_list.append(sum(distance_from_answer) / length)
        #     avg_dist_from_constraint.append(sum(distance_from_constraint) / length)
        #     std_list.append(std)
        
    if DEBUG:
        print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    result = calculate_objective_and_constraints(best_ind, verbose=True)

    if DEBUG:
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    run_result = {
    'best_individual': result['corrected_individual'],
    'time_slots': result['time_slots'],
    'total_schedule': result['total_schedule'],
    'objective': result['obj'],
    'constraint': result['constraint'],
    'robots': result['robots'],
    'requests': result['requests'],
    'actual_time_slots': result['actual_time_slots'],
    'actual_schedule': result['actual_schedule'],
    'scheduled_constraint': result['scheduled_constraint'],
    }


    # if GENERATE_IMAGE:
    #     axes = plt.gca()
    #     # axes.set_ylim([-1000000,0])
    #     # plt.plot(min_list, label='min')
    #     # plt.plot(max_list, label='max')
    #     # plt.plot(avg_list, label='avg')
    #     plt.plot(avg_obj_list, linewidth=3, label='GA', color="#4c72b0")
    #     plt.title('Average distance from objective', fontsize=TITLE_FONT_SIZE)
    #     # plt.plot(std_list, label='std')
    #     plt.xlabel("Generation", fontsize=AXIS_FONT_SIZE)
    #     plt.ylabel("Distance", fontsize=AXIS_FONT_SIZE)
    #     plt.legend(fontsize=LEGEND_FONT_SIZE)
    #     plt.tick_params(labelsize=AXIS_FONT_SIZE)
    #     if not os.path.exists(IMAGE_DIRECTORY):
    #         os.makedirs(IMAGE_DIRECTORY)
    #     plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + str(seed) + '_run' + str(run) + '_ga_obj.' + IMAGE_TYPE, bbox_inches='tight', pad_inches=0)

    #     plt.clf()
    #     plt.plot(avg_dist_from_constraint, linewidth=3, label='GA', color="#4c72b0")
    #     plt.title('Average distance from constraint', fontsize=TITLE_FONT_SIZE)
    #     # plt.plot(std_list, label='std')
    #     plt.xlabel("Generation", fontsize=AXIS_FONT_SIZE)
    #     plt.ylabel("Distance", fontsize=AXIS_FONT_SIZE)
    #     plt.legend(fontsize=LEGEND_FONT_SIZE)
    #     plt.tick_params(labelsize=AXIS_FONT_SIZE)
    #     if not os.path.exists(IMAGE_DIRECTORY):
    #         os.makedirs(IMAGE_DIRECTORY)
    #     plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + str(seed) + '_run' + str(run) + '_ga_constraint.' + IMAGE_TYPE, bbox_inches='tight', pad_inches=0)

    return run_result


def main():
    #----------
    # Run optimiser
    #----------
    global REQUEST_LIST

    run_results = []
    requests = []

    if CALCULATE_RUNTIME:
        start = time.time()
    
    for run in range(NUM_RUNS):
        REQUEST_LIST = []
        for request in range(config.TOTAL_REQUESTS):
            REQUEST_LIST.append(random.randint(config.MIN_REQUEST_DURATION, config.MAX_REQUEST_DURATION))

        run_result = pure_ga(run)
        run_results.append(run_result)
    
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


    #----------
    # Save results
    #----------
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)
    results_file = RESULTS_DIRECTORY + 't' + str(config.TOTAL_ROBOTS) + '_r' + str(config.MAX_RUNNING_ROBOTS) + '_ga' + '.pkl'    
    pickle.dump([seed, run_results], open(results_file, 'wb'))
    print("Seed and results saved to", results_file)


    #----------
    # Print stats
    #----------
    # run_results_objective_error = []
    # run_results_constraint_error = []
    # for run in range(NUM_RUNS):
    #     run_results_objective_error.append(run_results[run]['distance_from_optimal'])
    #     run_results_constraint_error.append(run_results[run]['distance_from_constraints'])
    # print("mean objective error", np.mean(np.array(run_results_objective_error), axis=0))
    # print("std objective error", np.std(np.array(run_results_objective_error), axis=0))
    # print("mean constraints error", np.mean(np.array(run_results_constraint_error), axis=0))
    # print("std constraints error", np.std(np.array(run_results_constraint_error), axis=0))




if __name__ == "__main__":
    main()