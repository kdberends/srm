"""
This module contains common functions
"""
from cycler import cycler
import json
import matplotlib.pyplot as plt 
import numpy as np 
import os
from datetime import datetime
from collections import namedtuple
from typing import List, AnyStr, Dict
import matplotlib as mpl
from logging import Logger, LogRecord
import logging 
import time 

import colorama
from colorama import Fore, Back, Style
from scipy.stats import gaussian_kde

from srm import statsfunc

import matplotlib as mpl 
_colors = ['#0A28A3', '#FFD814', '#00CC96']
mpl.rcParams['axes.prop_cycle'] = cycler(color=[_colors[0],
                                                _colors[1],
                                                _colors[2],
                                                "#D62728",
                                                "#9467BD",
                                                "#8C564B",
                                                "#e377c2",
                                                "#7F7F7F",
                                                "#BCBD22",
                                                "#17BECF",
                                                ])
                                                




class ElapsedFormatter:
    """
    Logger formatting class
    """
    

    def __init__(self):
        self.start_time = time.time()
        self.number_of_iterations = 1
        self.current_iteration = 0
        

        self._new_iteration = False
        self._intro = False
        self._colors = {'INFO': [Back.BLUE, Fore.BLUE],
                    'DEBUG': [Back.CYAN+Fore.BLACK, Fore.CYAN+Back.BLACK],
                    'WARNING': [Back.YELLOW + Fore.BLACK, Fore.YELLOW],
                    'ERROR': [Back.RED, Fore.RED],
                    'RESET': Style.RESET_ALL}
        colorama.init()

    def format(self, record):
        if self._intro:
            return self.__format_intro(record)
        elif self._new_iteration:
            return self.__format_header(record)
        else:
            return self.__format_message(record)

    def get_elapsed_time(self, current_time):
        return current_time - self.start_time
    
    def __current_time(self) -> AnyStr:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def __format_intro(self, record:LogRecord) -> AnyStr:
        level = record.levelname
        color = self._colors[level]
        reset = self._colors['RESET']
        return f"{record.getMessage()}"

    def __format_header(self, record:LogRecord) -> AnyStr:
        # Only one header
        self._new_iteration = False
        color = self._colors['DEBUG'][0]
        reset = self._colors['RESET']
        return f"{color}{self.__current_time()} {record.getMessage()} {reset}"

    def __format_message(self, record) -> AnyStr:
        level = record.levelname
        color = self._colors[level]
        elapsed_seconds = self.get_elapsed_time(record.created)
        return "{color}{now} {level:>7} {reset}{color2}{reset} {progress:4.0f}% T+ {elapsed:.2f}s {message}".format(
                color=color[0],
                color2=color[1],
                now=self.__current_time(),
                level=level,
                reset=self._colors['RESET'],
                elapsed=elapsed_seconds,
                message=record.getMessage(),
                progress=100*self.current_iteration/self.number_of_iterations)

    def __reset_time(self):
        self.start_time = time.time()

    def start_new_iteration(self):
        """
        Use this method to print a header
        """
        self.current_iteration += 1
        self._next_step()

    def _next_step(self):
        self._new_iteration = True
        self.__reset_time()

    def set_number_of_iterations(self, n):
        assert n > 0, 'Total number of iterations should be higher than zero'
        self.number_of_iterations = n

    def set_intro(self, flag: bool=True):
        self._intro=flag


class ElapsedFileFormatter(ElapsedFormatter):
    def __init__(self):
        super().__init__()
        
        self._colors = {'INFO': ['', ''],
                'DEBUG': ['', ''],
                'WARNING': ['', ''],
                'ERROR': ['', ''],
                'RESET': ''}


class SRMBase:
    """
    Base class for SRM classes. Implements methods for logging, project specific parameters
    """
    __logger = None
    __iniFile = None
    __version__ = "0.0.1"
    __contact__ = "koen.berends@deltares.nl"
    __authors__ = "Koen Berends"
    __copyright__ = "Copyright 2020, Deltares"
    __license__ = "LPGL"

    def __init__(self, logger:Logger = None):
        if logger:
            self.set_logger(logger)
        else:
            self._create_logger()

    def _create_logger(self):
        # Create logger
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)

        # create formatter
        self.__logger.__logformatter = ElapsedFormatter()
        self.__logger._Filelogformatter=ElapsedFileFormatter()


        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.__logger.__logformatter)
        self.__logger.addHandler(ch)

    def get_logger(self) -> Logger:
        """ Use this method to return logger object """
        return self.__logger

    def set_logger(self, logger:Logger) -> None:
        """ 
        Use to set logger

        Parameters:
            logger (Logger): Logger instance
        """
        assert isinstance(logger, Logger), '' + \
            'logger should be instance of Logger class'
        self.__logger = logger

    def set_logger_message(self, err_mssg: str, level: str='info', header: bool=False)->None:
        """Sets message to logger if this is set.

        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return

        if header:
            self.get_logformatter().set_intro(True)
            self.get_logger()._Filelogformatter.set_intro(True)
        else:
            self.get_logformatter().set_intro(False)
            self.get_logger()._Filelogformatter.set_intro(False)

        if level.lower() not in [
                'info', 'debug', 'warning', 'error', 'critical']:
            self.__logger.error(
                "{} is not valid logging level.".format(level.lower()))

        if level.lower() == 'info':
            self.__logger.info(err_mssg)
        elif level.lower() == 'debug':
            self.__logger.debug(err_mssg)
        elif level.lower() == 'warning':
            self.__logger.warning(err_mssg)
        elif level.lower() == 'error':
            self.__logger.error(err_mssg)
        elif level.lower() == 'critical':
            self.__logger.critical(err_mssg)

    def start_new_log_task(self, task_name: str="NOT DEFINED") -> None:
        """ 
        Use this method to start a new task. Will reset the internal clock. 

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().start_new_iteration()
        self.set_logger_message(f"Starting new task: {task_name}")

    def get_logformatter(self) -> ElapsedFormatter:
        """ Returns formatter """
        return self.get_logger().__logformatter

    def set_logfile(self, output_dir: str, filename: str='fm2prof.log') -> None:
        # create file handler
        fh = logging.FileHandler(os.path.join(output_dir, filename), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.get_logger()._Filelogformatter)
        self.__logger.addHandler(fh)


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def get_interval(modelresults):
    vals = []
    quantiles = [2.5, 5, 10, 25, 40, 45, 55, 50, 75, 80, 95, 97.5]
    for i in modelresults:
        p, val = statsfunc.get_empirical_cdf(i)
        qs = statsfunc.empirical_ppf(quantiles, p, val)
        vals.append(qs)

    return np.array(vals)

def SetPlotlyStyle(fig, title:str='no title'):
    """ 
    Set plotstyle for plotly plots 
    Call on figure:

    setDeltaresStylePlotly(fig, 
                           title=f'sobek-maas-{model}')

    """
    fig.update_layout(            
    title=dict(
        text=title,
        font = {'color': _colors[0]}
    ),
    xaxis=dict(                 
        title="Rivierkilometer",            
        linewidth=2,
        linecolor='black',
        showticklabels=True,
        tickmode='auto',
        ticklen=5,
        tickwidth=3,
        ticks="inside",
    ),
    yaxis=dict(                 
        title="Waterstandsverschil met w1",  
        linewidth=2,
        linecolor='black',
        showticklabels=True,
        tickmode='auto',
        ticklen=5,
        tickwidth=3,
        ticks="inside",
    ),
    font = {'family': 'Sansa pro'},
    plot_bgcolor =  '#FFF',    
    )

def SetPlotStyle(fig, legend:bool=True,legendbelow:bool=True, singleLegend:bool=False) -> None:
    """ 
    set deltares style for matplotlib plots
e
    Arguments:
        fig: matplotlib figure object
        legendbelow

    """
    if legendbelow:
        legendOptions = dict(fancybox=True, 
                        framealpha=0.5, 
                        edgecolor='None',
                        loc=3,
                        ncol=3,
                        bbox_to_anchor=(-0.02, -0.5))
    else:
        legendOptions = dict(fancybox=True, 
                            framealpha=0.5, 
                            edgecolor='None',
                            loc=2)

    for i, ax in enumerate(fig.axes):
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_edgecolor('#272727')
            ax.spines[spine].set_linewidth(2)
        if singleLegend & (i==len(fig.axes)-1):
            legend = ax.legend(**legendOptions)
            legend.get_frame().set_facecolor('#e5eef2')  # #e5eef2 #92b6c7
            legend.get_frame().set_boxstyle("square", pad=0)
        elif not singleLegend:
            legend = ax.legend(**legendOptions)
            legend.get_frame().set_facecolor('#e5eef2')  # #e5eef2 #92b6c7
            legend.get_frame().set_boxstyle("square", pad=0)

            


        