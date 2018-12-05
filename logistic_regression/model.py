import numpy as np
import pandas as pd
# import numpy.random as rnd
# import seaborn as sns
# from matplotlib import animation
import pymc3 as pm

from os import path

data = pd.read_csv(path.abspath(path.join(__file__, "../adult_us_postprocessed.csv")))
data['age2'] = np.square(data['age'])
print(data.head())



def manual_logistic_model():
    with pm.Model() as manual_logistic_model:

        alpha = pm.Normal('alpha', 0, 100)
        beta_1 = pm.Normal('beta_1', 0, 100)
        beta_2 = pm.Normal('beta_2', 0, 100)

        p = pm.invlogit(alpha + beta_1*data.age + beta_2*data.educ)

        y_obs = pm.Bernoulli('y_obs', p=p, observed=data.income_more_50K)

        map_estimate = pm.find_MAP()
        print(map_estimate)


def logistic_model():
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('income_more_50K ~ age + educ', data, family=pm.glm.families.Binomial())

        map_estimate = pm.find_MAP()
        print(map_estimate)


def plot_traces(traces, burnin=2000):
    ''' 
    Convenience function:
    Plot traces with overlaid means and values
    '''
    
    ax = pm.traceplot(traces[-burnin:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.summary(traces[-burnin:]).iterrows()})

    for i, mn in enumerate(pm.summary(traces[-burnin:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')


with pm.Model() as logistic_model:

    pm.glm.GLM.from_formula('income_more_50K ~ sex + age + age2 + educ + hours ', data, family=pm.glm.families.Binomial())
    map_estimate = pm.find_MAP()
    step = pm.Metropolis()

    trace = pm.sample(2000,  step, start=map_estimate, chains=1, tune=1000)

plot_traces(trace, burnin=200)

burnin = 100
b = trace['sex[T.Male]'][burnin:]
plt.hist(np.exp(b), bins=20, normed=True)
plt.xlabel("Odds Ratio")
plt.show()

lb, ub = np.percentile(b, 2.5), np.percentile(b, 97.5)
print("P(%.3f < Odds Ratio < %.3f) = 0.95" % (np.exp(lb), np.exp(ub)))




