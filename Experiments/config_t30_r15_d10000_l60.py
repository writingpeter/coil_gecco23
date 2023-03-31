# generate data: specify number of data points generated
NUM_DATA_POINTS = 10000

# generate data: specify robot numbers, duration (also used in optimise and comparison GA)
TOTAL_ROBOTS = 30
MAX_RUNNING_ROBOTS = 15 # MAX_RUNNING_ROBOTS must be <= TOTAL_ROBOTS
SLOT_IN_MINUTES = 10 # each slot is 10 mintes long
NUM_SLOTS_PER_HOUR = int(60/SLOT_IN_MINUTES) # 6 slots per hour
TOTAL_HOURS = 12
NUM_TIME_SLOTS = TOTAL_HOURS * NUM_SLOTS_PER_HOUR
MIN_SLOTS = 6 # minimum number of slots (each robot has to operate a minimum of 1 hour)

# generate data: specify num gen and num pop for the data generator GA
DATAGEN_GEN = 200 #500
DATAGEN_POP = 200

# generate data: specify min and max range for data
DATAGEN_MIN_RANGE = 0
DATAGEN_MAX_RANGE = NUM_TIME_SLOTS - MIN_SLOTS

# total number of variables
NUM_VARIABLES = 2 * TOTAL_ROBOTS

# learn representation: specify the number of data points used for learning
NUM_DATA_POINTS_USED = NUM_DATA_POINTS # NUM_DATA_POINTS_USED must be <= NUM_DATA_POINTS

# learn representation: specify the number of latent variables and epochs for the vae
NUM_LATENT = NUM_VARIABLES
NUM_EPOCHS = 200

# optimise: specify num gen and num pop for the optimiser GA
VAEGA_GEN = 50
VAEGA_POP = 20

# optimse: the range for the GA to generate random numbers for the latent variable
VAEGA_MIN_RANGE = -2.0
VAEGA_MAX_RANGE = 2.0

# optimse: specify requests (also used in comparison GA)
TOTAL_REQUESTS = 120
MIN_REQUEST_DURATION = 60 # in minutes
MAX_REQUEST_DURATION = 180 # in minutes

# optimse: specify whether to minimise or maximise function, 0 for min 1 for max (also used in comparison GA)
MIN_OR_MAX_FLAG = 1

# comparison GA: specify num gen and num pop for the GA
# GA_NUM_INDIVIDUALS = NUM_VARIABLES # the number of individuals for the GA is the number of variables
GA_GEN = 50
GA_POP = 20

