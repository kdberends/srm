from srm.data import StreamData
import srm.models as sm 
import numpy as np 

""" TEST Load data + eval model
data = StreamData('data/leigraaf/211I_output.json')

n = 1000
param = SrmParameters(
           nb=np.random.normal(0.04, scale=0.01, size=n), 
           rg=np.random.normal(0.18, scale=0.01, size=n),
           tmg=np.random.normal(120, scale=20, size=n),
           amax = np.random.normal(0.10, scale=0.05, size=n),
           n=n
           )

fm = FrictionModel(data, param=param)

fm.plot(show=True)
"""
"""
data = StreamData('data/leigraaf/streamdata/211I_output.json')
data.set_mask(years=[2015], months=range(0, 7))
print(data.discharge.shape)
print(data.years.shape)
print(data.daynumbers)
"""

import pymc3 as pm
from srm.data import StreamData
from srm.models import SrmParameters, FrictionModel, get_pmc_model
import srm.models as sm 
import numpy as np 
from scipy.stats import gaussian_kde
# Load data


"""
data = StreamData('../../data/leigraaf/streamdata/211I_output.json')
data.set_mask(years=[2015], months=range(0, 7))

# Get model
model = get_pmc_model(data)

with model:
    trace = pm.load_trace('../../data/leigraaf/traces/2015-1')
"""

"""
n = 2000
param_prior = sm.SrmParameters(
    amax=np.random.normal(0.16, scale=0.05, size=n),                       
    rg=np.random.normal(0.06, scale=0.01, size=n),
    tmg=np.random.normal(150, scale=20, size=n),
    nb=np.random.normal(0.04, scale=0.01, size=n), 
    n=n,
    varsigma=np.random.normal(0.04, scale=0.01, size=n),
                           )

param_alt = sm.SrmParameters(
    amax=np.random.normal(0.10, scale=0.05, size=n),                       
    rg=np.random.normal(0.06, scale=0.01, size=n),
    tmg=np.random.normal(150, scale=20, size=n),
    nb=np.random.normal(0.04, scale=0.01, size=n), 
    n=n,
    varsigma=np.random.normal(0.04, scale=0.01, size=n),
                           )

param_prior.plot(alt=param_alt)
"""

"""
n = 2000
param_prior = sm.SrmParameters(
    amax=np.random.normal(0.16, scale=0.05, size=n),                       
    rg=np.random.normal(0.06, scale=0.01, size=n),
    tmg=np.random.normal(150, scale=20, size=n),
    nb=np.random.normal(0.04, scale=0.01, size=n), 
    n=n,
    varsigma=np.random.normal(0.04, scale=0.01, size=n),
                           )

data = StreamData(list(range(365)), [1]*365)
sf = FrictionModel(data=data, param=param_prior)
sf.plot(show=True)
"""

"""
data = StreamData('data/leigraaf/streamdata/211I_output.json')
data.set_mask(years=[2015], months=range(0, 7))
brmPmc = get_pmc_model(data)
with brmPmc:
    trace = pm.load_trace('data/leigraaf/traces/2015-1')
param = sm.SrmParameters()
param.from_trace(trace)

#sm.FrictionModel(data=data, param=param).plot()
data = StreamData({'time':np.ones(20)*100, 'discharge':np.ones(20)*2})
BRM = sm.FrictionModel(data=data, param=param)
print (BRM.output)
"""

data = StreamData(list(range(365)), np.ones(365))