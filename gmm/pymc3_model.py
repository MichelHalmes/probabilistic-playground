import pymc3 as pm
import numpy as np
import theano.tensor as tt

from data import X

Ndata = len(X)
C = 3
D = 2

with pm.Model() as gmm_model:

    p = pm.Dirichlet('p', a=np.ones(C), shape=C)

    means = tt.stacklists([pm.Normal(f'mean_{c}', mu=0, sd=15, shape=D) for c in range(C)])

    sd = pm.Uniform('sd', lower=0, upper=20)

    category = pm.Categorical('category', p=p, shape=)

    # likelihood for each observed value
    points = pm.Normal('obs', mu=means[category], sd=sd, observed=data)
