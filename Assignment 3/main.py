import time
import scipy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("predator-prey-data.csv")
t = df.loc[:,'t'].to_numpy()
data = (df.loc[:,'x'].to_numpy(),df.loc[:,'y'].to_numpy())
hi =2

def get_ODE(data,t,alpha,beta,gamma,delta):
    """Return set of ODE's for given data and params."""


    # data should be array-like of prey and predator quantities respectively
    # params should be array-like of 4 params (alpha, beta, gamma, delta)
    x, y = data
    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt,dydt]

def solve_ODE(data,t,params,ODE=get_ODE):
    """Solve set of ODE's for given data and params."""

    # data should be array-like of prey and predator quantities respectively
    # ODE should be function that returns ODE's
    # params should be array-like of 4 params (alpha, beta, gamma, delta)
    x, y = data
    init = [x[0],y[0]]
    sol = scipy.integrate.odeint(ODE, init, t, args=tuple(params))
    return [sol[:,0],sol[:,1]]

def min_error(solver,data,t,params0=[0,0,0,0],function='MSE'):
    estimation = solve_ODE(data,t,params0)
    if function == 'MSE':
        error = mean_squared_error(est,data)
    else:
        error = mean_absolute_error(est,data)
    scipy.optimize.minimize(mean_squared_error,params0,args=)



params = [1,2,3,4]
est = solve_ODE(data,t,params)
error = mean_squared_error(est,data)

print(time.time())