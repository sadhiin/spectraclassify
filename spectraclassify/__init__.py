'''
@author: Shanjidul Islam Sadhin
Email: sadhin.aiub.cse@gmail.com
Date: 29-dec-2023
'''

import os
import sys
import logging
import argparse

logging_str = "[%(asctime)s]: %(levelname)s: %(module)s: %(message)s"
log_dir = "logs"
log_filesPath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format=logging_str,
                    handlers=[
                        logging.FileHandler(log_filesPath),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger("spectraclassify")
