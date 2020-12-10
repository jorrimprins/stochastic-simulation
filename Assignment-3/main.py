import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import numpy as np

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
        error_x = mean_squared_error(est[:,0],data[:,0])
        error_y = mean_squared_error(est[:,1],data[:,1])
        error = np.mean([error_x,error_y])
    elif evalfunc == 'RMSE':
        error_x = mean_squared_error(est[:, 0], data[:, 0],squared=False)
        error_y = mean_squared_error(est[:, 1], data[:, 1],squared=False)
        error = np.mean([error_x, error_y])
    else:
        error_x = mean_absolute_error(est[:, 0], data[:, 0])
        error_y = mean_absolute_error(est[:, 1], data[:, 1])
        error = np.mean([error_x, error_y])
    return error

def hillclimber(function, data,t,params0=(0,0,0,0),evalfunc='RMSE',n_iter=1000,stepsize=1):
    """Finds optimum by using hill climber algorithm (local search)"""

    est_params = params0
    try: est_eval = function(params0, data, t, evalfunc)
    except ValueError: est_eval = 100
    eval_list = [est_eval]
    for iter in range(n_iter):
        new_params = np.array(est_params) + np.random.uniform(-1,1,len(params0))*stepsize
        new_params[new_params < 0] = 0
        new_params[new_params > 10] = 10

        try: new_eval  = function(new_params,data,t,evalfunc)
        except ValueError: new_eval = 100
        if new_eval <= est_eval:
            est_params, est_eval = new_params, new_eval
        eval_list.append(est_eval)
    return (est_eval, est_params, eval_list)

def sim_an(function,data,t,params0=(0,0,0,0),evalfunc='RMSE',stepsize=0.25,
                        temprange=(10**0,10**-3),n_iter=5000,cooling='quadratic',n_inner=50):
    """Performs simulated annealing to find a global solution"""
    temp0 = temprange[0]
    temp_end = temprange[1]
    if cooling == 'exponential':
        rate = (temp_end / temp0) ** (1 / (n_iter/n_inner - 1))
    elif cooling == 'linear':
        rate = (temp0-temp_end)/(n_iter/n_inner)
    else:
        alpha = (temp0 / temp_end - 1) / (n_iter/n_inner) ** 2

    est_params = params0
    try: est_eval = function(params0, data, t, evalfunc)
    except ValueError: est_eval = 100
    eval_list = [est_eval]
    epoch = 0
    temp = temp0

    for i in range(int(n_iter/n_inner)):
        inner_params, inner_eval = est_params, est_eval
        for j in range(n_inner):
            new_params = np.array(inner_params) + np.random.uniform(-1,1,len(params0))*stepsize
            new_params[new_params < 0] = 0
            new_params[new_params > 10] = 10
            try: new_eval = function(new_params, data, t, evalfunc)
            except ValueError: new_eval = 100
            delta_eval = new_eval - inner_eval
            if delta_eval < 0:
                inner_params, inner_eval = new_params, new_eval
            elif np.random.uniform(0, 1) < np.exp(-delta_eval / temp):
                inner_params, inner_eval = new_params, new_eval
        est_params, est_eval = inner_params, inner_eval
        eval_list.append(est_eval)
        epoch += 1
        if cooling == 'exponential':
            temp *= rate
        elif cooling == 'linear':
            temp -= rate
        else:
            temp = temp0 / (1 + alpha * i ** 2)

    return (est_eval, est_params, eval_list)

def gen_al(function,data,t,evalfunc='RMSE',popsize=50,n_gen=25,
           n_parents=30,p_mutate=0.3):
    """Performs simulated annealing to find a solution"""
    pop = np.random.uniform(0, 2, (popsize,4))
    pop_eval = np.zeros(popsize)
    for p in range(popsize):
        try: pop_eval[p] = function(pop[p], data, t, evalfunc)
        except ValueError: pop_eval[p] = 100
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
            try: pop_eval[p] = function(pop[p], data, t, evalfunc)
            except ValueError: pop_eval[p] = 100
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
        # print(epoch)

    return (best_eval, best_params, pop,eval_list1,eval_list2)






print('Start')