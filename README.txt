README
------
Code for the paper entitled "Using a Variational Autoencoder to Learn Valid Search Spaces of Safely Monitored Autonomous Robots for Last-Mile Delivery"

Please cite:
Peter J. Bentley, Soo Ling Lim, Paolo Arcaini and Fuyuki Ishikawa (2023). Using a Variational Autoencoder to Learn Valid Search Spaces of Safely Monitored Autonomous Robots for Last-Mile Delivery. In Genetic and Evolutionary Computation Conference (GECCO’23). ACM, Lisbon, Portugal.


Top-level directory
.
├── Experiments                 # Experiments
└── README.txt            	# README file

Required packages:
deap==1.3.1
pytorch==1.9.0 
numpy==1.18.5
matplotlib==3.4.3


----
Directory: Experiments
----
.
├── ...
├── Experiments            		# COIL code, standard GA code and experiment settings
│   ├── config_t30_r10_d10000_l60.py	# Settings for E1.1: COIL vs GA (also acts as a baseline for all other experiments)
│   ├── config_t30_r15_d10000_l60.py	# Settings for E1.2: Number of simultaneously running robots RT
│   ├── config_t30_r20_d10000_l60.py	# Settings for E1.2: Number of simultaneously running robots RT
│   ├── config_t20_r10_d10000.py	# Settings for E1.3: Total number of robots Rb
│   ├── config_t25_r10_d10000.py	# Settings for E1.3: Total number of robots Rb
│   ├── max_req_durations             	# COIL code, standard GA code and experiment settings for E1.4: Changing request durations dr
│   ├── config_t30_r10_d2500_l60.py	# Settings for E2.1: Reducing dataset size DS
│   ├── config_t30_r10_d5000_l60.py	# Settings for E2.1: Reducing dataset size DS
│   ├── config_t30_r10_d7500_l60.py	# Settings for E2.1: Reducing dataset size DS
│   ├── config_t30_r10_d10000_l10.py	# Settings for E2.2: Modifying number of latent variables maxlv
│   ├── config_t30_r10_d10000_l20.py	# Settings for E2.2: Modifying number of latent variables maxlv
│   ├── config_t30_r10_d10000_l30.py	# Settings for E2.2: Modifying number of latent variables maxlv
│   ├── config_t30_r10_d10000_l40.py	# Settings for E2.2: Modifying number of latent variables maxlv
│   ├── config_t30_r10_d10000_l50.py	# Settings for E2.2: Modifying number of latent variables maxlv
│   ├── generate_data.py		# COIL Step 1: generate data
│   ├── learn_representation.py 	# COIL Step 2: learns representation
│   ├── optimise.py			# COIL Step 3: optimise
│   ├── ga.py				# Standard GA
│   ├── scheduler			# Scheduler code used by COIL and GA
│   ├── analyse.py              	# Compares results from GA and COIL
│   ├── data             		# Folder containing data generated by generate_data.py
│   ├── vae             		# Folder containing VAEs generated by learn_representation.py
│   └── results             		# Folder containing results generated by optimse.py and ga.py 
└── ...


* To run Experiment 1.1 (all the other experiments are run the same way):
Running COIL:
>> python generate_data.py -c config_t30_r10_d10000_l60 -t
>> python learn_representation.py -c config_t30_r10_d10000 -t
>> python optimise.py -c config_t30_r10_d10000_l60 -t -r 100

Running standard GA:
>> python ga.py -c config_t30_r10_d10000_l60 -t -r 100

Comparing COIL with GA:
>> python analyse.py -c config_t30_r10_d10000_l60

