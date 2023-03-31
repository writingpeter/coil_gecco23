import numpy as np
import argparse
import importlib
import os, time
import pickle
import matplotlib.pyplot as plt

RESULTS_DIRECTORY = 'results/'
# IMAGE_DIRECTORY = 'image/'

image_format = 'pdf' # png

# if the directory to save image does not exist, create it
# if not os.path.exists(IMAGE_DIRECTORY):
#     os.makedirs(IMAGE_DIRECTORY)

#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='enter the name of the file containing all the configuration settings, e.g., enter config to import the file config.py')

args = parser.parse_args()

#----------
# Import file containing all the configuration settings
#----------
if args.config:
    file_name = args.config
    config = importlib.__import__(file_name) # import file
else:
    exit("Error. Please specify config filename in the command line. Use --help for more information.")


# MIN_VARIABLES = eq.MIN_VARIABLES

NUM_RUNS = 100
# variable_list = []


# for i in range(MIN_VARIABLES, MAX_VARIABLES+1):
#     variable_list.append(str(i))


def get_results(experiment):
    mean_objective_error = []
    mean_constraint_error = []
    stderr_objective_error = []
    stderr_constraint_error = []

    if experiment == 'ga':
        data_file = RESULTS_DIRECTORY + 't' + str(config.TOTAL_ROBOTS) + '_r' + str(config.MAX_RUNNING_ROBOTS) + '_ga' + '.pkl' 
    else:
        data_file = RESULTS_DIRECTORY + 't' + str(config.TOTAL_ROBOTS) + '_r' + str(config.MAX_RUNNING_ROBOTS) + '_d' + str(config.NUM_DATA_POINTS_USED) + '_l' + str(config.NUM_LATENT) + '_coil' + '.pkl'    
    optimise_seed, run_results = pickle.load(open(data_file, 'rb'))

    run_results_objective = []
    run_results_constraint = []
    run_results_alt_constraint = []
    # if experiment == 'coil':
    #     print(experiment)
    #     print('objective,constraint')
    for run in range(NUM_RUNS):
        # if run == 91 and experiment == 'ga':
        # if run == 1 and experiment == 'coil':
        #     # print(str(run_results[run]['objective']) + ',' + str(run_results[run]['constraint']))

        #     # time_slots = run_results[run]['time_slots']
        #     # for slot in time_slots:
        #     #     print(','.join(map(str, slot)))

        #     print(','.join(map(str, run_results[run]['total_schedule'])))
            
            # print(str(run_results[run]['robots']))
            # print(str(run_results[run]['requests']))
        # print('variables', run_results[run]['variables'])
        # print(run_results[run]['distance_from_optimal'])
        run_results_objective.append(run_results[run]['objective'])
        run_results_constraint.append(run_results[run]['constraint'])
        run_results_alt_constraint.append(run_results[run]['scheduled_constraint'])
        # if experiment == 'coil':
        print(str(run_results[run]['objective']) + ',' + str(run_results[run]['scheduled_constraint']))

    return run_results_objective, run_results_constraint, run_results_alt_constraint


# def draw_image(dataset1, dataset1_stderr, dataset1_name, dataset2, dataset2_stderr, dataset2_name, xlabel, ylabel, title, imagename):
#     error_kw = dict(lw=0.8, capsize=3, capthick=0.8)
#     x = np.arange(len(variable_list))  # the label locations
#     width = 0.4  # the width of the bars
#     fig, ax = plt.subplots()
#     plt.tick_params(labelsize=14)
#     rects1 = ax.bar(x - width/2, dataset1, width, yerr=dataset1_stderr, error_kw=error_kw, label=dataset1_name, color="#4c72b0")
#     rects2 = ax.bar(x + width/2, dataset2, width, yerr=dataset2_stderr, error_kw=error_kw, label=dataset2_name, color="#dd8452")

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_title(title, fontsize=18)
#     ax.set_xlabel(xlabel, fontsize=14)
#     ax.set_ylabel(ylabel, fontsize=14)
#     ax.set_xticks(list(range(len(variable_list))))
#     ax.set_xticklabels(variable_list)
#     # ax.set_xticks(np.arange(min(x),max(x),1))
#     ax.legend(fontsize=14)

#     # ax.bar_label(rects1, padding=3)
#     # ax.bar_label(rects2, padding=3)

#     fig.tight_layout()

#     # plt.show()

#     plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + imagename, dpi=72, bbox_inches='tight', pad_inches=0)

vaega_run_results_objective, vaega_run_results_constraint, vaega_run_results_alt_constraint = get_results('coil')

print('coil')
print('----')

mean = np.mean(np.array(vaega_run_results_objective), axis=0)
std = np.std(np.array(vaega_run_results_objective), axis=0)

print('avg objective', mean)
print('std objective', std)
print('min objective', min(vaega_run_results_objective))
print('max objective', max(vaega_run_results_objective))

mean = np.mean(np.array(vaega_run_results_constraint), axis=0)
std = np.std(np.array(vaega_run_results_constraint), axis=0)

print('avg constraint', mean)
print('std constraint', std)
print('min constraint', min(vaega_run_results_constraint))
print('max constraint', max(vaega_run_results_constraint))

mean = np.mean(np.array(vaega_run_results_alt_constraint), axis=0)
std = np.std(np.array(vaega_run_results_alt_constraint), axis=0)

print('avg scheduled constraint', mean)
print('std scheduled constraint', std)
print('min scheduled constraint', min(vaega_run_results_alt_constraint))
print('max scheduled constraint', max(vaega_run_results_alt_constraint))


ga_run_results_objective, ga_run_results_constraint, ga_run_results_alt_constraint = get_results('ga')

print('ga')
print('----')

mean = np.mean(np.array(ga_run_results_objective), axis=0)
std = np.std(np.array(ga_run_results_objective), axis=0)

print('avg objective', mean)
print('std objective', std)
print('min objective', min(ga_run_results_objective))
print('max objective', max(ga_run_results_objective))

mean = np.mean(np.array(ga_run_results_constraint), axis=0)
std = np.std(np.array(ga_run_results_constraint), axis=0)

print('avg constraint', mean)
print('std constraint', std)
print('min constraint', min(ga_run_results_constraint))
print('max constraint', max(ga_run_results_constraint))

mean = np.mean(np.array(ga_run_results_alt_constraint), axis=0)
std = np.std(np.array(ga_run_results_alt_constraint), axis=0)

print('avg scheduled constraint', mean)
print('std scheduled constraint', std)
print('min scheduled constraint', min(ga_run_results_alt_constraint))
print('max scheduled constraint', max(ga_run_results_alt_constraint))



# coil = vaega_mean_objective_error
# coil_stderr = vaega_stderr_objective_error
# coil_name = 'SOLVE'
# ga = ga_mean_objective_error
# ga_stderr = ga_stderr_objective_error
# ga_name = 'GA'
# xlabel = 'Number of variables'
# ylabel = 'Average percentage error'
# title = equation_name.upper() + ': Average percentage objective error'
# imagename = 'objective_error.' + image_format
# draw_image(ga, ga_stderr, ga_name, coil, coil_stderr, coil_name, xlabel, ylabel, title, imagename)

# coil = vaega_mean_constraint_error
# coil_stderr = vaega_stderr_constraint_error
# coil_name = 'SOLVE'
# ga = ga_mean_constraint_error
# ga_stderr = ga_stderr_constraint_error
# ga_name = 'GA'
# xlabel = 'Number of variables'
# ylabel = 'Average criterion error'
# title = equation_name.upper() + ': Average criterion error (per variable)'
# imagename = 'constraint_error.' + image_format
# draw_image(ga, ga_stderr, ga_name, coil, coil_stderr, coil_name, xlabel, ylabel, title, imagename)

