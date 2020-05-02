WIDTH = 800
HEIGHT = 600
PLAYERWIDTH = 64
OBSTACLEWIDTH = 30
PRESENTWIDTH = 100
PRESENTHEIGHT = 75
OBJECT_SPEED = 0.3
OBJECTSHOR = 2
OBJECTSVER = 3

SPLIT = [0,
         2,
         4,
         4 + OBJECTSHOR,
         4 + OBJECTSHOR * 2,
         4 + OBJECTSHOR * 2 + OBJECTSVER,
         4 + OBJECTSHOR * 2 + OBJECTSVER * 2]
OBS_SIZE = 4 + OBJECTSHOR * 2 + OBJECTSVER * 2 + 2
REWARD_TYPES = 3
ACTION_SIZE = 2

DATASET_SPLIT = (OBS_SIZE, ACTION_SIZE, OBS_SIZE, REWARD_TYPES)
