import time
import numpy as np
import pandas as pd
from main import ODE_error, hillclimber, sim_an, gen_al

seedy = 5
reps = 20
df = pd.read_csv("predator-prey-data.csv")
name = 'fulldata'
t = df.loc[:,'t'].to_numpy()
data = np.transpose(np.array([df.loc[:,'x'].values.tolist(),df.loc[:,'y'].values.tolist()]))
start = time.time()
n_gen=15
popsize=50
n_parents=30
p_mutate=0.3
n_iter = 300

## Estimating the three models and its parameters on full data, see convergence behavior and performance

np.random.seed(seedy)
for eval in ['RMSE','MAE']: # Loop over both evaluation methods
    RMSE_HC, RMSE_SA, RMSE_GA = [], [], []
    RMSE_HC_conv, RMSE_SA_conv, RMSE_GA_conv, RMSE_GA_conv2 = np.repeat(0, n_iter), np.repeat(0, n_iter), \
                                                              np.repeat(0,n_gen), np.repeat(0, n_gen)
    params_HC, params_SA, params_GA = np.repeat(0, 4), np.repeat(0, 4), np.repeat(0, 4)

    #Replicate experiments
    for i in range(reps):
        #1. Hill climber
        print('REPLICATION {}'.format(i))
        RMSE, params, RMSE_list = hillclimber(ODE_error, data, t,n_iter=n_iter,
                                              params0 = np.random.uniform(0, 2, 4))
        RMSE_HC.append(RMSE)
        RMSE_HC_conv = [x + y  for x, y in zip(RMSE_HC_conv, np.array(RMSE_list)/reps)]
        if i == 0:
            params_HC = params
        elif RMSE < RMSE_HC[i-1]:
            params_HC = params
        #2. Simulated Annealing
        RMSE, params, RMSE_list = sim_an(ODE_error, data, t,n_iter=n_iter,
                                         params0=np.random.uniform(0, 2, 4))
        RMSE_SA.append(RMSE)
        RMSE_SA_conv = [x + y  for x, y in zip(RMSE_SA_conv, np.array(RMSE_list)/reps)]
        if i == 0:
            params_SA = params
        elif RMSE < RMSE_SA[i-1]:
            params_SA = params
        #3. Genetic Algorithm
        RMSE, params, pop, RMSE_list_best, RMSE_list_avg = gen_al(ODE_error, data, t, n_gen=n_gen)
        RMSE_GA.append(RMSE)
        RMSE_GA_conv = [x + y for x, y in zip(RMSE_GA_conv, np.array(RMSE_list_best) / reps)]
        RMSE_GA_conv2 = [x + y for x, y in zip(RMSE_GA_conv2, np.array(RMSE_list_avg) / reps)]
        if i == 0:
            params_GA = params
        elif RMSE < RMSE_GA[i - 1]:
            params_GA = params
    #Create dataframes and save
    RMSE_dict = {'HC':[np.mean(RMSE_HC)],'SA':[np.mean(RMSE_SA)],'GA':[np.mean(RMSE_GA)]}
    pd.DataFrame(RMSE_dict).to_csv('Data/{}-overall-{}.csv'.format(name,eval))
    RMSE_dict = {'HC':RMSE_HC_conv,'SA':RMSE_SA_conv}
    pd.DataFrame(RMSE_dict).to_csv('Data/{}-avg-conv-{}.csv'.format(name,eval))
    pd.DataFrame({'GA':RMSE_GA_conv}).to_csv('Data/{}-avg-conv-GA-{}.csv'.format(name,eval))
    param_dict = {'HC':params_HC,'SA':params_SA,'GA':params_GA}
    pd.DataFrame(param_dict).to_csv('Data/{}-best-params-{}.csv'.format(name,eval))

print('Simulations part 1 took {} seconds'.format(time.time()-start))

## Estimating the three models and its parameters on part of data, see convergence behavior and performance
np.random.seed(seedy)
name = 'rndm-frac'
reps = 5

for eval in ['RMSE', 'MAE']:  # Loop over both evaluation methods
    RMSE_HC, RMSE_SA, RMSE_GA = [], [], []
    for s in np.arange(100,0,-10):
        print('SAMPLE SIZE {}'.format(s))
        df_sample = df.sample(s).sort_values(by='t')
        t = df_sample.loc[:, 't'].to_numpy()
        data = np.transpose(np.array([df_sample.loc[:, 'x'].values.tolist(), df_sample.loc[:, 'y'].values.tolist()]))

        RMSE_HC_sample, RMSE_SA_sample, RMSE_GA_sample = [], [], []
        #Replicate experiments
        for i in range(reps):
            #1. Hill climber
            print('REPLICATION {}'.format(i))
            RMSE, params, RMSE_list = hillclimber(ODE_error, data, t,n_iter=n_iter,
                                         params0=np.random.uniform(0, 2, 4))
            RMSE_HC_sample.append(RMSE)

            #2. Simulated Annealing
            RMSE, params, RMSE_list = sim_an(ODE_error, data, t,n_iter=n_iter,
                                         params0=np.random.uniform(0, 2, 4))
            RMSE_SA_sample.append(RMSE)

            #3. Genetic Algorithm
            RMSE, params, pop, RMSE_list_best, RMSE_list_avg = gen_al(ODE_error, data, t, n_gen=n_gen)
            RMSE_GA_sample.append(RMSE)
        RMSE_HC.append(np.mean(RMSE_HC_sample))
        RMSE_SA.append(np.mean(RMSE_SA_sample))
        RMSE_GA.append(np.mean(RMSE_GA_sample))

    #Create dataframes and save
    RMSE_dict = {'HC':RMSE_HC,'SA':RMSE_SA,'GA':RMSE_GA}
    pd.DataFrame(RMSE_dict).to_csv('Data/{}-{}.csv'.format(name,eval))




print('Simulations took {} seconds'.format(time.time()-start))