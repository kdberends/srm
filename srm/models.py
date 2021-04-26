# standard library
from typing import Callable, List, Union
import json

# dependencies
import matplotlib.pyplot as plt 
import numpy as np 
import pymc3 as pm

# package
from srm.data import StreamData, SrmParameters
from srm.common import SetPlotStyle, get_interval, kde_scipy


class Priors:
    def __init__(self):
        self.amax = pm.Uniform('amax', lower=0.05, upper=0.25)  # Maximum value for a
        self.growth_rate = pm.Uniform('growth_rate', lower=0.0, upper = 0.08) # growth rate
        self.day_max_growth = pm.Uniform('day_max_growth', lower=50, upper=300)
        self.base_manning = pm.Uniform('base_manning', lower=0.01, upper=0.05)
        self.sigma = pm.HalfCauchy('sigma', beta=2, testval=1.)



class FrictionModel:
    def __init__(self, data:StreamData, param:SrmParameters):
        """
        asdf
        """
        self._data = data 
        self._parameters = param 
        self._alphamodel = self._alpha_growth 
        self.__output = None

    @property
    def output(self):
        if self.__output is not None:
            return self.__output
        else:
            self.__output = self.eval()
            return self.__output


    def eval(self):
        if self._parameters.n > 1:
            # assume matrix is supplied
            T = self._make_matrix(self._data.daynumbers, self._parameters.n)
            Q = self._make_matrix(self._data.discharge, self._parameters.n)
        else:
            T = self._data.daynumbers
            Q = self._data.discharge
        
        alpha = self._alpha_growth(T, self._parameters)
        roughness = alpha / Q + self._parameters.nb
        scale = roughness*self._parameters.varsigma
        scale[scale<=0] = np.nan
        roughness_incl_noise = np.random.normal(loc=roughness, scale=scale)
        return [roughness, roughness_incl_noise, alpha]

    def plot(self, breakpoints: tuple=(), show:bool=False, ax=None):
        try: 
            self.output[0].shape[1]
            return self._plot_uncertainty(breakpoints, show, axIn=ax)
        except IndexError:
            return self._plot_deterministic(breakpoints, show,axIn=ax)

    def _plot_deterministic(self, breakpoints: tuple=(), show:bool=False, axIn=None):
        # create figure
        fig, ax = plt.subplots(1, figsize=(12, 5))
        cmap = plt.get_cmap('viridis_r')

        # Plot mean (expected)
        ax.plot(self._data.daynumbers, self.output[0], '--', color='orange')

        
        # Plot data
        ax.plot(self._data.daynumbers, self._data.manning, '.r')
        
        # plot breakpoints
        if breakpoints:
            ylim = ax.get_ylim()
            for breakpoint in breakpoints:
                ax.plot([breakpoint]*2, ylim, '--k')
                
        # Set up labels etc. 
        ax.set_ylim([0, 0.6])
        ax.set_ylabel('Manning coefficient')
        ax.set_xlabel('Dagnummer')
        SetPlotStyle(fig)
        if show:
            plt.show()
        return fig, ax

    def _plot_uncertainty(self, breakpoints: tuple=(), 
                                show:bool=False, 
                                axIn=None,
                                variable='manning'):

        if axIn is None:
            fig, ax = plt.subplots(1, figsize=(12, 5))
        else:
            ax = axIn
        

        cmap = plt.get_cmap('viridis_r')

        # Plot mean (expected)
        ax.plot(self._data.daynumbers, np.mean(self.output[0], 1), '--', color='orange')


        # plot uncertainty
        model_uncertainty = get_interval(self.output[0])
        total_uncertainty = get_interval(self.output[1])
        
        labels = ['Model (97.5%)', 'Model (90%)', 'Model (80%)', 'Model (50%)', 'Model (20%)', 'Model (10%)']
        for i in range(1, 6):
            if axIn is None:
                ax.fill_between(self._data.daynumbers, 
                                model_uncertainty[:, i], 
                                model_uncertainty[:, 11-i], 
                                color=cmap(int(i/5*255)), alpha = i/5, label=labels[i])
            else:
                ax.fill_between(self._data.daynumbers, 
                                model_uncertainty[:, i], 
                                model_uncertainty[:, 11-i], 
                                color=cmap(int(i/5*255)), alpha = i/5)
        # Plot data
        ax.plot(self._data.daynumbers, self._data.manning, '.r')
        ax.plot(self._data.daynumbers, total_uncertainty[:, 1], '--k')
        if axIn is None:
            ax.plot(self._data.daynumbers, total_uncertainty[:, 10], '--k', label='Totale onzekerheid')
        else:
            ax.plot(self._data.daynumbers, total_uncertainty[:, 10], '--k')
        
        # plot breakpoints
        if breakpoints:
            ylim = ax.get_ylim()
            for breakpoint in breakpoints:
                ax.plot([breakpoint]*2, ylim, '--k')
                
        # Set up labels etc. 
        ax.set_ylim([0, 0.6])
        ax.set_ylabel('Manning coefficient')
        ax.set_xlabel('Dagnummer')
        
        if show:
            plt.show()
            
        if axIn is None:
            SetPlotStyle(fig)
            return fig, ax
        else:
            return ax


    @staticmethod
    def _make_matrix(m, n):
        return np.tile(m, (n, 1)).T

    @staticmethod
    def _alpha_constant(t, p):
        """dummy for constant alpha"""
        return p.amax

    @staticmethod
    def _alpha_growth(t, p):
        """only growth"""
        out = np.zeros(t.shape)
        out = p.amax / (1+np.exp(-p.rg*(t-p.tmg)))
        
        return out

    @staticmethod
    def _alpha_growth_mortality(t, p):
        """growth and mortality"""
        out = np.zeros(t.shape)
        out = p.vmax * ((1+np.exp(-p.rg*(t-p.tsg)))**-1 + 
                        (1+np.exp(p.rm*(t-p.tsm)))**-1 - 
                        1)
        
        return out



def get_pmc_model(data, prior_bounds: dict):
    
    days = data.daynumbers
    dataN = data.manning 
    dataQ = data.discharge 
    print (f'NumberOfDataPoints: {len(dataN)}')

    growth_model = pm.Model()
    
    with growth_model:
        # Informative priors
        if prior_bounds:
            amax = pm.Uniform('amax', **prior_bounds.get('amax'))  # Maximum value for a
            r = pm.Uniform('growth_rate', **prior_bounds.get('growth_rate')) # growth rate
            tmg = pm.Uniform('day_max_growth', **prior_bounds.get('day_max_growth'))
            nb = pm.Uniform('base_manning', **prior_bounds.get('base_manning'))

            
        else:
            # uninformed
            amax = pm.Uniform('amax', lower=0.05, upper=0.25)  # Maximum value for a
            r = pm.Uniform('growth_rate', lower=0.0, upper = 0.08) # growth rate
            tmg = pm.Uniform('day_max_growth', lower=50, upper=300)
            nb = pm.Uniform('base_manning', lower=0.01, upper=0.05)

        # Multiplier for predictive uncertainty
        sigma = pm.HalfCauchy('sigma', beta=2, testval=1.)

        # Mean function (=deterministic vegetation model)
        alpha = amax / (1+pm.math.exp(-r*(days-tmg)))
        mu = alpha / dataQ + nb

        # Likelihood function (=probabilistic model)
        likelihood = pm.Normal('y', 
                               mu= mu,
                               sd=mu*sigma, 
                               observed=dataN)
        # MCMC inference
        #trace = pm.sample(2000, cores=1,chains=1, target_accept=0.8)
        #print (model.check_test_point())
    return growth_model