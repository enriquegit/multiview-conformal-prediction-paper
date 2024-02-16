# File that contains global variables.

DATASET_NAME = "htad" # Directory name where the data.csv file is located.

PCT_TRAIN = 0.5 # Percent of training data.

PCT_CALIBRATION = 0.50 # From the remaining data (1 - PCT_TRAIN) what percent is for calibration. The rest goes to the test set.

ITERATIONS = 15

ALPHA = 0.05 # Maximum error.

DATA_PATH = "../data/" # Root path where all datasets reside.

DATASET_PATH = DATA_PATH + DATASET_NAME + "/"

FILE_PATH = DATA_PATH + DATASET_NAME + "/data.csv"

NUMCORES = 17 # Number of CPU cores to use depending on your machine.

NTREES = 50
