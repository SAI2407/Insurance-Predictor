import datetime
import logging
import os
import datetime

# Create and configure logger
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename=f"logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)