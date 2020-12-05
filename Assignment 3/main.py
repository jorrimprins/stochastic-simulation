import time
import scipy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

seedy = 10
df = pd.read_csv("predator-prey-data.csv")
t = df.loc[:,'t'].to_numpy()
data = np.transpose(np.array([df.loc[:,'x'].values.tolist(),df.loc[:,'y'].values.tolist()]))

def get_ODE(data,t,alpha,beta,gamma,delta):
    """Return set of ODE's for given data and params."""


    # data should be array-like of prey and predator quantities respectively
    # params should be array-like of 4 params (alpha, beta, gamma, delta)
    x, y = data
    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt,dydt]

def ODE_error(params,data,t,evalfunc='RMSE'):
    """Solve set of ODE's for given data and params."""

    # data should be array-like of prey and predator quantities respectively
    # ODE should be function that returns ODE's
    # params should be array-like of 4 params (alpha, beta, gamma, delta)
    def get_ODE(data, t, alpha, beta, gamma, delta):
        """Return set of ODE's for given data and params."""

        # data should be array-like of prey and predator quantities respectively
        # params should be array-like of 4 params (alpha, beta, gamma, delta)
        x, y = data
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    init = data[0]
    est = scipy.integrate.odeint(get_ODE, init, t, args=tuple(params))
    if evalfunc == 'MSE':
        error = mean_squared_error(est,data)
    elif evalfunc == 'RMSE':
        error = np.sqrt(mean_squared_error(est,data))
    else:
        error = mean_absolute_error(est,data)
    return error

def estimate_params(function, data,t,params0=[0,0,0,0],evalfunc='RMSE'):
    optim = scipy.optimize.minimize(function, params0, args=(data, t,evalfunc),method='BFGS')
    return (optim.fun, optim.x)

def hillclimber(function, data,t,params0=[0,0,0,0],evalfunc='RMSE',n_iter=1500,stepsize=0.1):
    est_params = params0
    est_eval = ODE_error(params0,data,t,evalfunc)
    for iter in range(n_iter):
        new_params = np.array(params0) + np.random.uniform(-1,1,len(params0))*stepsize
        help_array = [new_params[j] < 0 for j in range(len(new_params))]
        while sum(help_array) > 0:
            for i in range(len(help_array)):
                if help_array[i]:
                    new_params[i] = np.array(est_params[i]) + np.random.uniform(-1, 1) * stepsize
                    help_array = [new_params[j] < 0 for j in range(len(new_params))]
        new_eval  = ODE_error(new_params,data,t,evalfunc)
        if new_eval <= est_eval:
            est_params, est_eval = new_params, new_eval
        print(iter)
    return (est_eval, est_params)

def sim_an(function, data,t,params0=[0,0,0,0],evalfunc='RMSE',stepsize=0.1,
                        temprange=(10,0.1),alpha=0.001,n_iter=1000):
    """Peforms simulated annealing to find a solution"""
    init_temp, final_temp = temprange
    prev_temp = init_temp+1
    temp = init_temp
    est_params = params0
    est_eval = ODE_error(params0, data, t, evalfunc)
    epoch = 0

    for i in range(n_iter):
        new_params = np.array(est_params) + np.random.uniform(-1,1,len(est_params))*stepsize
        help_array = [new_params[j] < 0 for j in range(len(new_params))]
        while sum(help_array) > 0:
            for i in range(len(help_array)):
                if help_array[i]:
                    new_params[i] = np.array(est_params[i]) + np.random.uniform(-1, 1) * stepsize
                    help_array = [new_params[j] < 0 for j in range(len(new_params))]
        new_eval = ODE_error(new_params, data, t, evalfunc)
        delta_eval = est_eval - new_eval
        if delta_eval > 0:
            est_params, est_eval = new_params, new_eval
        else:
            if np.random.uniform(0, 1) < np.exp(delta_eval / temp):
                est_params, est_eval = new_params, new_eval
        epoch += 1
        print(epoch)
        print(est_eval)
        prev_temp = temp
        # temp = init_temp / (1 + 5 * np.log(epoch + 1))
        # temp = init_temp/np.log(epoch+2)
        temp *= 0.9


    return (est_eval, est_params,epoch)


# MSE, params_hat = estimate_params(ODE_error,data,t)
np.random.seed(seedy)
# RMSE, params_hat = hillclimber(ODE_error,data,t)
# MAE, params_hat2 = hillclimber(ODE_error,data,t,evalfunc='MAE')
# RMSE2, params_hat3 = hillclimber(ODE_error,data,t,params0=[1.5,1.5,1.5,1.5])
RMSE3, params_hat4, iter = sim_an(ODE_error,data,t)
MAE2, params_hat5, iter = sim_an(ODE_error,data,t,evalfunc='MAE')



print(time.time())