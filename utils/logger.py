import tensorflow as tf
import numpy as np
import time
import os 
import datetime
import yaml
from dataclasses import dataclass, asdict
import logging


@dataclass
class logger_params:

    academy_name: str = ""
    log_dir: str = ""
    config: str = ""
    date_time_now: str = str(datetime.datetime.now())
    file_name: str = ""
    
    def __post_init__(self):
        c = ":. "
        for char in c: 
            self.date_time_now = self.date_time_now.replace(char, "-")

        if self.academy_name is not "":
            self.file_name = self.config + "_" + self.academy_name


color2code = dict(  warning='\033[1;41m',
                    ok='\033[1;32;40m',
                    info='\033[1;36;40m',
                    grey='\x1b[0;37;40m')


def log(str="", color="info"):
    col = color2code[color]
    end = "\033[0m" + "\n"
    dash = '-' * 65
    print("\n" + col + dash)
    print(str)
    print(dash + end)


def logStr(str="", color="ok"):
    col = color2code[color]
    end = "\033[0m"
    print(col + str + end)


def log2logger(str="", color="ok"):
    col = color2code[color]
    end = "\033[0m"
    logging.getLogger("PPO").info(col + str + end)


class Logger():

    def __init__(self, academy_name="", log_dir="", config=""):

        self.logger_params = logger_params(academy_name=academy_name, log_dir=log_dir, config=config)
        self.metrics = dict()
        self.summary_writer = tf.summary.create_file_writer(self.logger_params.log_dir + '/_Training_Diagnostics/summaries/' + self.logger_params.file_name)

    def store(self, name, value):

        if value is not None:
            if name not in self.metrics.keys():
                self.metrics[name] = tf.keras.metrics.Mean(name=name)
            self.metrics[name].update_state(value)

    def log_metrics(self, epoch):

        log('MEAN METRICS START', color="ok")

        logStr('{:<15s}{:>15}'.format("Epoch", epoch))

        if not self.metrics:
            logStr('NO METRICS')

        else:
            for key, metric in self.metrics.items():
                value = metric.result()
                logStr('{:<15s}{:>15.5f}'.format(key, value))

                with self.summary_writer.as_default():
                    tf.summary.scalar(key, value, step=epoch)

                metric.reset_states()

        log('MEAN METRICS END', color="ok")

    def tf_model_path(self, name='tf_model'):
        return self.logger_params.log_dir + '/_Trained_Network/ckpts/' + str(name)
