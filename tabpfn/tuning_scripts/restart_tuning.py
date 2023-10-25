from syne_tune.experiments import load_experiment
import sys
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)

tuner = load_experiment(sys.argv[1], load_tuner=True).tuner 
tuner.run()