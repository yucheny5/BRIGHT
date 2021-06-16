import bright_utils
from new_train import train
import config

"""
get the rwr embedding 
"""
bright_utils.preprocess(0.2, False)
"""
train the model
"""
train(0.2, 250, 0.0001, 128, 500, 10, True, False)
