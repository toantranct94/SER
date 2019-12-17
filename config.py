RATE = '16000' 

BASE_TRAIN = '../Dataset/TrainSet/'

BASE_PUBLIC_TEST = '../Dataset/TestSet/'

TRAINING_GT = '../Dataset/train_label.csv'

TEST_GT = '../Dataset/public_test.csv'

AGU_DATA_PATH = '../Dataset/TrainSet/Aug/'

AUG_GT = AGU_DATA_PATH + 'aug.csv'

TRANSFORM_TIMEWARP = 'TW/'

TRANSFORM_FREQ_MASK = 'FM/'

TRANSFORM_TIME_MASK = 'TM/'

TRANSFORM_COMBINE = 'CB/'

NUM_WORKERS = 16 

INFER_ONLY = False # change this to False to train the model again 

USE_DATA_AUG = True
