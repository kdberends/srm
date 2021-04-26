"""
This module contains functions for reading data
"""
# standard library
import json
import os
from typing import List, Iterable
from datetime import datetime, timedelta
from pathlib import Path

# dependency
import numpy as np 
import matplotlib.pyplot as plt 


# package
from srm.common import SetPlotStyle, get_interval, kde_scipy
from srm.common import SRMBase


class SrmParameters:
    def __init__(self, amax:float=0.1, 
                       rg:float=0.18, 
                       tmg:float=120, 
                       nb:float=0.03, 
                       varsigma:float=0.20,
                       n:int=1):


        self.amax = amax
        self.rg = rg 
        self.tmg = tmg 
        self.nb = nb 
        self.varsigma = varsigma 
        self.n = n 
        self.label = 'nolabel'

    def from_trace(self, trace):
        """ To load parameters from a pymc trace """
        self.amax = trace.amax
        self.nb = trace.base_manning 
        self.rg = trace.growth_rate
        self.tmg = trace.day_max_growth
        self.varsigma = trace.sigma 
        self.n = len(trace.amax)
   
    
    def from_dict(self, data):
        data_names = ['amax', 'base_manning', 'growth_rate', 'day_max_growth', 'sigma']
        attributes = ['amax', 'nb', 'rg', 'tmg', 'varsigma']
        for data_name, attribute in zip(data_names, attributes):
            setattr(self, attribute, np.array(data.get(data_name)).T)

        self.n = len(self.amax)

    @property
    def manning(self): return self.nb
    
    @property
    def growth_rate(self): return self.rg

    @property
    def day_max_growth(self): return self.tmg

    
    def plot(self, alt:'SrmParameters'=None):
        """ To plot distributions """
        #fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        #axlist = [item for sublist in axs for item in sublist]
        
        axlist = []
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 2)
        axlist.append(fig.add_subplot(gs[0, 0]))
        axlist.append(fig.add_subplot(gs[0, 1]))
        axlist.append(fig.add_subplot(gs[1, 0]))
        axlist.append(fig.add_subplot(gs[1, 1]))
        axlist.append(fig.add_subplot(gs[2, :]))

        

        self._plot_pdf(axlist, self)
        if alt:
            if isinstance(alt, list):
                for i, al in enumerate(alt):
                    if i > 6:
                        linestyle = '--'
                    else:
                        linestyle = '-'
                    self._plot_pdf(axlist, al, linestyle=linestyle)
            else:
                self._plot_pdf(axlist, alt)

        SetPlotStyle(fig, singleLegend=True)
        plt.tight_layout()
        #plt.show()

    def _plot_pdf(self, axs:List, ob, linestyle:str='-'):
        
        self._plot_hist_kde(ob, axs[0], ob.amax, title=r'$\alpha_{max}$', linestyle=linestyle)
        self._plot_hist_kde(ob, axs[1], ob.nb, 'Base manning', linestyle=linestyle)
        self._plot_hist_kde(ob, axs[2], ob.rg, 'Growth rate', linestyle=linestyle)
        self._plot_hist_kde(ob, axs[3], ob.tmg, 'Day of maximum growth', linestyle=linestyle)
        ax = self._plot_hist_kde(ob, axs[4], ob.varsigma, 'sigma', linestyle=linestyle)

        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])
        
    @staticmethod
    def _plot_hist_kde(ob, ax, values, title:str='', linestyle:str='-'):
        hist_default = dict(density=True, fc='gray', histtype='stepfilled', alpha=0.3, linewidth=2)
        ymin = min(values)
        ymax = max(values)
        x = np.linspace(ymin,ymax)
        bandwidth = (ymax-ymin)/25
        kde = kde_scipy(values, x, bandwidth=bandwidth)
        l = ax.plot(x, kde, label=ob.label, linestyle=linestyle)
        #ax.hist(values, 25, edgecolor=l[0].get_color(), **hist_default)
        
        
        ax.set_title(title)
        
        return ax 


class StreamData(SRMBase):
    __jsonkeys = {'time': 'time',
                  'discharge': 'discharge',
                  'waterlevel_upstream': 'up',
                  'waterlevel_downstream': 'down',
                  'manning': 'computed_friction'
                 }

    def __init__(self, *args, running_daynumbers:bool=False):
        super().__init__()

        self._data = {}

        self._running_daynumbers = running_daynumbers
        
        if isinstance(args[0], str) or isinstance(args[0], Path):
            os.path.isfile(args[0])
            self.__init_from_json(args[0])
        elif isinstance(args[0], dict):
            self.__init_from_dict(args[0])
        else:
            self.__init_from_arrays(*args)

    def __init_from_arrays(self, *args):
        """
        array: [time, discharge, manning, waterlevel downstream]
        """
        self._data['time'] =  args[0] #np.array([datetime(2020, 1, 1, 0, 0, 0) + timedelta(days=int(t)) for t in args[0]])
        self._data['discharge'] = np.array(args[1])
        try:
            self._data['manning'] = np.array(args[2])
            self._data['waterlevel_downstream'] = np.array(args[3])
        except IndexError:
            self._data['manning'] = np.array([np.nan]*len(self._data['discharge']))
            self._data['waterlevel_downstream'] = np.array([np.nan]*len(self._data['discharge']))
        self._mask = [True] * len(self._data['time'])
    
    def __init_from_json(self, inputfile):
        try:
            with open(inputfile, 'r') as f:
                data = json.load(f)
                self._mask = [True] * len(data.get(self.__jsonkeys['time']))
            self.set_logger_message(f'Loaded {inputfile}')
        except FileNotFoundError:
            self.set_logger_message(f'Could not find {inputfile} in {os.getcwd()}', 'error')

        try:
            self._data = self._parsejson(data)
        except Exception as e:
            self.set_logger_message('Unexpected error parsing json data', 'error')

    def __init_from_dict(self, inputdict):
        self._data['time'] = np.array([datetime(2020, 1, 1, 0, 0, 0) + timedelta(days=t) for t in inputdict['time']])
        self._data['discharge'] = np.array(inputdict['discharge'])
        try:
            self._data['manning'] = np.array(inputdict['manning'])
        except KeyError:
            self._data['manning'] = np.array([np.nan]*len(self._data['discharge']))
        self._mask = [True] * len(self._data['time'])
        
    def set_mask(self, years:Iterable=range(2021), months:Iterable=range(13), days:Iterable=range(368)):
        dyears = self.get_year(self._data['time'])
        dmonths = self.get_month(self._data['time'])
        ddaynumbers = self.get_day_of_year(self._data['time'])
        
        yearmask = sum([dyears == i for i in years])
        monthmask = sum([dmonths == i for i in months])
        daymask = sum([ddaynumbers == i for i in days])

        mask = np.argwhere(yearmask & monthmask & daymask &
                        (~np.isnan(self._data['discharge']))&
                        (~np.isnan(self._data['manning']))\
                        ).T[0].astype(int)
        self._mask = mask 


    @property
    def discharge(self):
        return self._data['discharge'][self._mask]

    @property
    def time(self):
        return self._data['time'][self._mask]
        
    @property
    def manning(self):
        return self._data['manning'][self._mask]
    
    @property
    def daynumbers(self):
        if self._running_daynumbers:
            year_base = 366*(self.years - min(self.years))
            return year_base+self.get_day_of_year(self._data['time'])[self._mask]
        else:
            return self.get_day_of_year(self._data['time'])[self._mask]
    
    @property
    def years(self):
        return self.get_year(self._data['time'])[self._mask]

    @property
    def waterlevel_upstream(self):
        return self._data['waterlevel_upstream'][self._mask]

    @property
    def waterlevel_downstream(self):
        return self._data['waterlevel_downstream'][self._mask]

    @property
    def months(self):
        return self.get_month(self._data['time'])[self._mask]
    
    @staticmethod
    def get_day_of_year(t:List[datetime]) -> np.ndarray:
        """
        Returns ndarray of daynumbers for list of datetime objects

        Arguments:
            t: list of datetime objects

        Return:
            An ndarray of daynumbers
        """
        return np.array([(i - datetime(year=i.year, month=1, day=1)).days for i in t])

    @staticmethod
    def get_year(t:List[datetime]) -> np.ndarray:
        """
        Returns ndarray of years for list of datetime objects

        Arguments:
            t: list of datetime objects

        Return:
            An ndarray of daynumbers
        """
        return np.array([i.year for i in t])

    @staticmethod
    def get_month(t:List[datetime]) -> np.ndarray:
        """
        Returns ndarray of years for list of datetime objects

        Arguments:
            t: list of datetime objects

        Return:
            An ndarray of daynumbers
        """
        return np.array([i.month for i in t])

    def _parsejson(self, jsondata):
        data = {}
        for key, value in self.__jsonkeys.items():
            data[key] = np.array(jsondata[value])

        # Parse time to datetime
        data['time'] = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in data['time']])

        
        return data  


class SRMData(SRMBase):
    """
    Class to interact with database
    """
    __streamdata_file = 'data/streamdata.json'
    __interventions_file = 'data/interventions.json'
    __traces_file = "data/traces.json"

    def __init__(self, data_dir:str):
        super().__init__()

        # check if path contains all necessary data
        if self._check_data_dir(Path(data_dir)):
            self.data_dir = Path(data_dir)

        # load intervention dates to memory
        with open(self.data_dir.joinpath(self.__interventions_file), 'r') as f:
            self._interventions = json.load(f)

        # load streamdata
        self.streamdata = StreamData(self.data_dir.joinpath(self.__streamdata_file))
        
        # load traces to memory
        with open(self.data_dir.joinpath(self.__traces_file), 'r') as f:
            self._traces = json.load(f)
        
    def get_growth_parameters_for_daynumber(self, daynumber:int=0, n:int=500, years=None):
        """ return growth rate, maximum growth """
        params = []
        if years==None: years = list(self._interventions.get('years').keys())

        for year, periods in self._interventions.get('years').items():
            for iperiod, period in enumerate(periods):
                if (daynumber >= period[0]) & (daynumber <= period[1]) & (year in years):
                    print (f"{year}: period {period}")
                    params.append(self._load_trace_for_period_and_year(iperiod, int(year)))
        
        draws_per_period = int(n/len(params))
        last_draw = n%len(params)

        output = {"growth_rate": [], "maximum_growth": []}
        for param in params:
            output['growth_rate'].extend(list(param.growth_rate[:draws_per_period]))
            output['maximum_growth'].extend(list(param.amax[:draws_per_period]))
        
        output['growth_rate'].extend(list(param.growth_rate[draws_per_period:draws_per_period+last_draw]))
        output['maximum_growth'].extend(list(param.growth_rate[draws_per_period:draws_per_period+last_draw]))
        
        return output
        
    def _load_trace_for_period_and_year(self, period:int, year=int):
        param = SrmParameters()
        param.from_dict(self._traces[str(year)][period])
        return param

    def _check_data_dir(self, path:Path=''):
        if not path.is_dir():
            raise FileNotFoundError('path is not a directory')
            
        if not path.joinpath(self.__streamdata_file).is_file():
            raise FileNotFoundError('no streamdata file')
            
        if not path.joinpath(self.__interventions_file).is_file():
            raise FileNotFoundError('no interventions file')
            
        if not path.joinpath(self.__traces_file).is_file():
            raise FileNotFoundError("No traces directory")

        return True                     

if __name__ == "__main__":

    data = SRMData('data/leigraaf_211I')
    print (data.get_growth_parameters_for_daynumber(180, years=['2017']).get('growth_rate'))
    

    #p = data._load_trace_for_period_and_year(0, 2015)
    #print(p.manning)

    #data.print_intervention_dates()