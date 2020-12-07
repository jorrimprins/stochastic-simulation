import time
import scipy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

def get_ODE(data, t, alpha, beta, gamma, delta):
    """Returns set of ODE's for given data and params."""

    # data should be array-like of prey and predator quantities respectively
    # params should be array-like of 4 params (alpha, beta, gamma, delta)
    x, y = data
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def ODE_error(params,data,t,evalfunc='RMSE'):
    """Solves set of ODE's for given data and params."""

    # data should be array-like of prey and predator quantities respectively
    # ODE should be function that returns ODE's
    # params should be array-like of 4 params (alpha, beta, gamma, delta)


    init = data[0]
    est = scipy.integrate.odeint(get_ODE, init, t, args=tuple(params))
    if evalfunc == 'MSE':
        error = mean_squared_error(est,data)
    elif evalfunc == 'RMSE':
        error = mean_squared_error(est,data,squared=False)
    else:
        error = mean_absolute_error(est,data)
    return error

def hillclimber(function, data,t,params0=(0,0,0,0),evalfunc='RMSE',n_iter=1000,stepsize=0.1):
    """Finds optimum by using hill climber algorithm (local search)"""

    est_params = params0
    try:
        est_eval = function(params0, data, t, evalfunc)
    except ValueError:
        est_eval = 100
    eval_list = [est_eval]
    for iter in range(n_iter):
        new_params = np.array(params0) + np.random.uniform(-0.5,1.5,len(params0))*stepsize
        new_params[new_params < 0] = 0
        try:
            new_eval  = function(new_params,data,t,evalfunc)
        except ValueError:
            new_eval = 100
        if new_eval <= est_eval:
            est_params, est_eval = new_params, new_eval
        eval_list.append(est_eval)
        print(iter)
    return (est_eval, est_params, eval_list)

def sim_an(function,data,t,params0=(0,0,0,0),evalfunc='RMSE',stepsize=0.1,
                        init_temp=100,n_iter=1000):
    """Performs simulated annealing to find a global solution"""

    temp = init_temp
    est_params = params0
    try:
        est_eval = function(params0, data, t, evalfunc)
    except ValueError:
        est_eval = 100
    eval_list = [est_eval]
    epoch = 0

    for i in range(n_iter):
        new_params = np.array(est_params) + np.random.uniform(-0.5,1.5,len(est_params))*stepsize
        new_params[new_params < 0] = 0
        try:
            new_eval = function(new_params, data, t, evalfunc)
        except ValueError:
            new_eval = 100
        delta_eval = new_eval - est_eval
        if delta_eval < 0:
            est_params, est_eval = new_params, new_eval
        else:
            if np.random.uniform(0, 1) < np.exp(-delta_eval / temp):
                est_params, est_eval = new_params, new_eval
        eval_list.append(est_eval)
        epoch += 1
        print(epoch)
        # temp = init_temp / (1 + 5 * np.log(epoch + 1))
        # temp = init_temp/np.log(epoch+2)
        temp *= 0.9
        # temp = 100 / (1 + 0.5 * epoch ** 2)

    return (est_eval, est_params, eval_list)

def gen_al(function,data,t,evalfunc='RMSE',popsize=50,n_gen=25,
           n_parents=30,p_mutate=0.3):
    """Performs simulated annealing to find a solution"""
    pop = np.random.uniform(0, 2, (popsize,4))
    pop_eval = np.zeros(popsize)
    for p in range(popsize):
        try:
            pop_eval[p] = function(pop[p], data, t, evalfunc)
        except ValueError:
            pop_eval[p] = 100
    eval_list1 = [min(pop_eval)]
    eval_list2 = [np.mean(pop_eval)]

    epoch = 0

    for i in range(n_gen):
        sort_eval = list(np.sort(pop_eval))
        parents = []
        for j in range(n_parents):
            parents.append(np.where(pop_eval == sort_eval[j])[0][0])
        random.shuffle(parents)

        for j in range(int(len(parents) / 2)):
            for k in range(np.random.randint(1, 4)):
                alpha = np.random.uniform(0, 1)
                offspring = alpha * pop[parents[j]] + (1 - alpha) * pop[parents[j + int(len(parents) / 2)]]
                if np.random.uniform(0, 1) <= p_mutate:
                    offspring += np.random.normal(0, 1, 4)/20
                    offspring[offspring < 0] = 0
                pop = np.vstack((pop, offspring))
        pop_eval = np.zeros(pop.shape[0])
        for p in range(pop.shape[0]):
            try:
                pop_eval[p] = function(pop[p], data, t, evalfunc)
            except ValueError:
                pop_eval[p] = 100
        sort_eval = list(np.sort(pop_eval))
        index = []
        for j in range(popsize):
            index.append(np.where(pop_eval == sort_eval[j])[0][0])
        pop = pop[index]
        pop_eval = sort_eval[0:popsize]
        best_eval = min(pop_eval)
        best_params = pop[np.where(pop_eval == best_eval)[0][0]]
        eval_list1.append(best_eval)
        eval_list2.append(np.mean(pop_eval))
        epoch += 1
        print(epoch)

    return (best_eval, best_params, pop,eval_list1,eval_list2)



print('Start')