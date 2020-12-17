import time
import numpy as np
import pandas as pd
import warnings
from functions import get_ODE, ODE_error, hillclimber, sim_an, gen_al
warnings.filterwarnings('ignore')

seedy = 12345
reps = 25
df = pd.read_csv("predator-prey-data.csv")
t = df.values[:,0]
data_full = (df.values[:,1],df.values[:,2])
data = (df.values[:,1],df.values[:,2])
start = time.time()
n_iter = 5000

# Estimating the models and its parameters on full data, see convergence behavior and performance

np.random.seed(seedy)
for eval in ['RMSE','MAE']: # Loop over both evaluation methods
    start1 = time.time()
    HC05, HC10, HC_start1, HC_start2 = [], [], [], []
    HC05_conv, HC10_conv, HC_start1_conv, HC_start2_conv = np.repeat(0, n_iter), np.repeat(0, n_iter),\
                                                                                 np.repeat(0, n_iter), np.repeat(0, n_iter)
    SA01, SA025, SA05, SA10, SA40, SA25, SA100 = [], [], [], [], [], [], []
    SA01_conv, SA025_conv, SA05_conv, SA10_conv, SA40_conv, SA25_conv, SA100_conv = np.repeat(0, n_iter), np.repeat(0,n_iter), \
                                                                                 np.repeat(0, n_iter), np.repeat(0,n_iter), \
                                                                                 np.repeat(0, n_iter), np.repeat(0,n_iter),\
                                                                                    np.repeat(0,n_iter)
    SA_exp, SA_lin, SA_init10, SA_init01, SA_init005, HC20 = [], [], [], [], [], []
    SA_exp_conv, SA_lin_conv, SA_init10_conv, SA_init01_conv, SA_init005_conv, HC20_conv = np.repeat(0, n_iter), np.repeat(0,n_iter),np.repeat(0, n_iter), \
                                                               np.repeat(0,n_iter), np.repeat(0, 20000), np.repeat(0, 20000)
    for i in range(reps):
        print('REPLICATION {}'.format(i))
        #1. Stepsize experiments Hill climber
        RMSE, params, RMSE_list = hillclimber(ODE_error, data, t,n_iter=n_iter,stepsize=1.5,evalfunc=eval)
        HC05.append(RMSE)
        HC05_conv = [x + y  for x, y in zip(HC05_conv, np.array(RMSE_list)/reps)]
        if i == 0: params_HC05 = params
        elif RMSE < HC05[i-1]: params_HC05 = params
        print('HC .5 DONE')
        print('Run {} with {} took {} seconds'.format(i,eval,time.time()-start1))

        RMSE, params, RMSE_list = hillclimber(ODE_error, data, t, n_iter=n_iter, stepsize=1,evalfunc=eval)
        HC10.append(RMSE)
        HC10_conv = [x + y for x, y in zip(HC10_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_HC10 = params
        elif RMSE < HC10[i - 1]: params_HC10 = params
        print('HC 1 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        #2. Starting value experiments HC
        RMSE, params, RMSE_list = sim_an(ODE_error, data, t,n_iter=n_iter,params0=(1,1,1,1),evalfunc=eval)
        HC_start1.append(RMSE)
        HC_start1_conv = [x + y for x, y in zip(HC_start1_conv, np.array(RMSE_list)/reps)]
        if i == 0: params_HC_start1 = params
        elif RMSE < HC_start1[i-1]: params_HC_start1 = params
        print('HC 1,1,1,1 DONE')
        print('Run {} with {} took {} seconds'.format(i,eval,time.time()-start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, params0=(2, 2, 2, 2),evalfunc=eval)
        HC_start2.append(RMSE)
        HC_start2_conv = [x + y for x, y in zip(HC_start2_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_HC_start2 = params
        elif RMSE < HC_start2[i - 1]: params_HC_start2 = params
        print('HC 2,2,2,2 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        #3. Stepsize experiments for Simulated Annealing
        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter,stepsize=0.5,evalfunc=eval)
        SA01.append(RMSE)
        SA01_conv = [x + y for x, y in zip(SA01_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA01 = params
        elif RMSE < SA01[i - 1]: params_SA01 = params
        print('SA .1 DONE')
        print('Run {} with {} took {} seconds'.format(i,eval,time.time()-start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, stepsize=1,evalfunc=eval)
        SA025.append(RMSE)
        SA025_conv = [x + y for x, y in zip(SA025_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA025 = params
        elif RMSE < SA025[i - 1]: params_SA025 = params
        print('SA .25 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, stepsize=1.5,evalfunc=eval)
        SA05.append(RMSE)
        SA05_conv = [x + y for x, y in zip(SA05_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA05 = params
        elif RMSE < SA05[i - 1]: params_SA05 = params
        print('SA .5 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, stepsize=2,evalfunc=eval)
        SA10.append(RMSE)
        SA10_conv = [x + y for x, y in zip(SA10_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA10 = params
        elif RMSE < SA10[i - 1]: params_SA10 = params
        print('SA 1 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        # 4. Markov chain experiments with SA
        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, n_inner=25,evalfunc=eval)
        SA25.append(RMSE)
        SA25_conv = [x + y for x, y in zip(SA25_conv, np.array(RMSE_list) / reps)]
        if i == 0:
            params_SA25 = params
        elif RMSE < SA25[i - 1]:
            params_SA25 = params
        print('SA 25 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, n_inner=40,evalfunc=eval)
        SA40.append(RMSE)
        SA40_conv = [x + y for x, y in zip(SA40_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA40 = params
        elif RMSE < SA40[i - 1]: params_SA40 = params
        print('SA 40 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, n_inner=100,evalfunc=eval)
        SA100.append(RMSE)
        SA100_conv = [x + y for x, y in zip(SA100_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA100 = params
        elif RMSE < SA100[i - 1]: params_SA100 = params
        print('SA 100 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        # 7. Cooling schedule experiments for SA
        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, cooling='linear',evalfunc=eval)
        SA_lin.append(RMSE)
        SA_lin_conv = [x + y for x, y in zip(SA_lin_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA_lin = params
        elif RMSE < SA_lin[i - 1]: params_SA_lin = params
        print('SA linear DONE')
        print('Run {} with {} took {} seconds'.format(i,eval,time.time()-start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, cooling='exponential',evalfunc=eval)
        SA_exp.append(RMSE)
        SA_exp_conv = [x + y for x, y in zip(SA_exp_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA_exp = params
        elif RMSE < SA_exp[i - 1]: params_SA_exp = params
        print('SA exp DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        # 8. Starting temperature experiments for SA
        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter,temprange=(10**1,10**-3),evalfunc=eval)
        SA_init10.append(RMSE)
        SA_init10_conv = [x + y for x, y in zip(SA_init10_conv, np.array(RMSE_list) / reps)]
        if i == 0: params_SA_init10 = params
        elif RMSE < SA_init10[i - 1]: params_SA_init10 = params
        print('SA init10 DONE')
        print('Run {} with {} took {} seconds'.format(i,eval,time.time()-start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=n_iter, temprange=(10 ** -1, 10 ** -3),evalfunc=eval)
        SA_init01.append(RMSE)
        SA_init01_conv = [x + y for x, y in zip(SA_init01_conv, np.array(RMSE_list) / reps)]
        if i == 0:
            params_SA_init01 = params
        elif RMSE < SA_init01[i - 1]:
            params_SA_init01 = params
        print('SA init01 DONE')
        print('Run {} with {} took {} seconds'.format(i, eval, time.time() - start1))

        RMSE, params, RMSE_list = sim_an(ODE_error, data, t, n_iter=20000, evalfunc=eval, stepsize=1, n_inner=40)
        SA_init005.append(RMSE)
        SA_init005_conv = [x + y for x, y in zip(SA_init005_conv, np.array(RMSE_list) / reps)]
        if i == 0:
            params_SA_init005 = params
        elif RMSE < SA_init005[i - 1]:
            params_SA_init005 = params

        RMSE, params, RMSE_list = hillclimber(ODE_error, data, t, n_iter=20000, evalfunc=eval, stepsize=1)
        HC20.append(RMSE)
        HC20_conv = [x + y for x, y in zip(HC20_conv, np.array(RMSE_list) / reps)]
        if i == 0:
            params_HC20 = params
        elif RMSE < HC20[i - 1]:
            params_HC20 = params


    HC_convs = [HC05_conv, HC10_conv, HC_start1_conv, HC_start2_conv,HC20_conv]
    for j in range(len(HC_convs)):
        if len(HC_convs[j]) < n_iter:
            HC_convs[j].extend(list(np.zeros(n_iter-len(HC_convs[j]))))


    SA01_conv = np.array([np.repeat(SA01_conv[i + 1], 50) for i in np.arange(0, len(SA01_conv) - 1)]).reshape(n_iter)
    SA025_conv = np.array([np.repeat(SA025_conv[i + 1], 50) for i in np.arange(0, len(SA025_conv) - 1)]).reshape(n_iter)
    SA05_conv = np.array([np.repeat(SA05_conv[i + 1], 50) for i in np.arange(0, len(SA05_conv) - 1)]).reshape(n_iter)
    SA10_conv = np.array([np.repeat(SA10_conv[i + 1], 50) for i in np.arange(0, len(SA10_conv) - 1)]).reshape(n_iter)
    SA25_conv = np.array([np.repeat(SA25_conv[i + 1], 25) for i in np.arange(0, len(SA25_conv) - 1)]).reshape(n_iter)
    SA40_conv = np.array([np.repeat(SA40_conv[i + 1], 40) for i in np.arange(0, len(SA40_conv) - 1)]).reshape(n_iter)
    SA100_conv = np.array([np.repeat(SA100_conv[i + 1], 100) for i in np.arange(0, len(SA100_conv) - 1)]).reshape(n_iter)
    SA_lin_conv = np.array([np.repeat(SA_lin_conv[i + 1], 50) for i in np.arange(0, len(SA_lin_conv) - 1)]).reshape(n_iter)
    SA_exp_conv = np.array([np.repeat(SA_exp_conv[i + 1], 50) for i in np.arange(0, len(SA_exp_conv) - 1)]).reshape(n_iter)
    SA_init10_conv = np.array([np.repeat(SA_init10_conv[i + 1], 50) for i in np.arange(0, len(SA_init10_conv) - 1)]).reshape(n_iter)
    SA_init01_conv = np.array([np.repeat(SA_init01_conv[i + 1], 50) for i in np.arange(0, len(SA_init01_conv) - 1)]).reshape(n_iter)

    #Create dataframes and save
    error_dict = {'HC05': HC05,'HC10': HC10,'HC_start1': HC_start1, 'HC_start2': HC_start2,
                 'SA01': SA01,'SA025': SA025, 'SA05': SA05,'SA10': SA10,'SA25':SA25,'SA40': SA40, 'SA100': SA100,
                 'SA_exp': SA_exp,'SA_lin': SA_lin, 'SA_init10': SA_init10,'SA_init01':SA_init01,'SA20doez': SA_init005,'Hc20doez': HC20}
    conv_dict = {'HC05': HC05_conv, 'HC10': HC10_conv,'HC_start1': HC_start1_conv,
                 'HC_start2': HC_start2_conv,'SA01': SA01_conv, 'SA025': SA025_conv, 'SA05': SA05_conv, 'SA10': SA10_conv,'SA25': SA25_conv,
                 'SA40': SA40_conv, 'SA100': SA100_conv,'SA_exp': SA_exp_conv, 'SA_lin': SA_lin_conv, 'SA_init10': SA_init10_conv,
                 'SA_init01': SA_init01_conv}
    param_dict = {'HC05': params_HC05, 'HC10': params_HC10,'HC_start1': params_HC_start1,
                  'HC_start2': params_HC_start2,'SA01': params_SA01, 'SA025': params_SA025, 'SA05': params_SA05, 'SA10': params_SA10,
                  'SA25':params_SA25,'SA40': params_SA40, 'SA100': params_SA100,'SA_exp': params_SA_exp, 'SA_lin': params_SA_lin,
                  'SA_init10': params_SA_init10, 'SA_init01': params_SA_init01,'SA20doez': params_SA_init005,'Hc20doez': params_HC20}
    pd.DataFrame(error_dict).to_csv('Data/Final/{}.csv'.format(eval))
    pd.DataFrame(conv_dict).to_csv('Data/Final/{}-conv.csv'.format(eval))
    pd.DataFrame(param_dict).to_csv('Data/Final/{}-params.csv'.format(eval))

    SA_init005_conv = np.array([np.repeat(SA_init005_conv[i + 1], 40) for i in np.arange(0, len(SA_init005_conv) - 1)]).reshape(20000)
    pd.DataFrame({'SA20doez':SA_init005_conv,'HC20doez':HC20_conv}).to_csv('Data/Final/{}-conv-20000.csv'.format(eval))

print('1st simulation experiments took {} seconds'.format(time.time()-start))

# ## Estimating the three models and its parameters on part of data, see convergence behavior and performance
start2 = time.time()
sizes = np.arange(100,0,-10)
shortlist = ['lessx','lessy','lessboth','onlypeaks','nopeaks']

for eval in ['RMSE', 'MAE']:
    print('Runs with {} as evaluation function'.format(eval))
    for short in shortlist:
        HC, SA = [], []
        HC_std, SA_std = [], []
        for s in sizes:
            print('SAMP SIZE {}'.format(s))
            if short == 'lessx':
                index_x = np.sort(np.append(np.random.choice(np.arange(1, 99), s - 2), np.array([0, 99])))
                index_y = np.arange(0, 100)
            elif short == 'lessy':
                index_y = np.sort(np.append(np.random.choice(np.arange(1, 99), s - 2), np.array([0, 99])))
                index_x = np.arange(0, 100)
            elif short == 'lessboth':
                index_x = np.sort(np.append(np.random.choice(np.arange(1, 99), s - 2), np.array([0, 99])))
                index_y = np.sort(np.append(np.random.choice(np.arange(1, 99), s - 2), np.array([0, 99])))
            elif short == 'onlypeaks':
                index_x = np.sort(np.append(np.where(data[0] > 2), np.array([0, 99])))
                index_y = np.sort(np.append(np.where(data[1] > 2), np.array([0, 99])))
            else:
                index_x = np.sort(np.append(np.where(data[0] < 2), np.array([0, 99])))
                index_y = np.sort(np.append(np.where(data[1] < 2), np.array([0, 99])))
            indexdata = (index_x, index_y)
            shortdata = (data[0][index_x], data[1][index_y])

            HC_sample, SA_sample = [], []
            for i in range(reps):
                #1. Hill climber
                print('REP {}'.format(i))
                RMSE, params, RMSE_list = hillclimber(ODE_error, shortdata, t,n_iter=n_iter,indexdata=indexdata,evalfunc=eval)
                error = ODE_error(params,data,t,evalfunc=eval)
                HC_sample.append(error)

                #2. Simulated Annealing
                RMSE, params, RMSE_list = sim_an(ODE_error, shortdata, t, n_iter=n_iter, indexdata=indexdata,
                                                     temprange=(10 ** 0, 10 ** -3),evalfunc=eval,n_inner=40)
                error = ODE_error(params, data, t, evalfunc=eval)
                SA_sample.append(RMSE)

            HC.append(np.mean(HC_sample))
            HC_std.append(np.std(HC_sample))
            SA.append(np.mean(SA_sample))
            SA_std.append(np.std(SA_sample))
            if short in ['onlypeaks', 'nopeaks']:
                break
        RMSE_dict = {'HC_mean':HC,'HC_std':HC_std,'SA_mean':SA,'SA_std':SA_std}
        pd.DataFrame(RMSE_dict).to_csv('Data/Final/fracdata-{}-{}.csv'.format(short,eval))



print('Short data simulations took {} seconds'.format(time.time()-start2))


# Simulate Genetic Algorithm
n_gen=50
popsize=100
n_parents=50
p_mutate=0.3
start = time.time()
GA = []
conv_dict = {}
param_dict = {}
for eval in ['RMSE','MAE']:
    GA_conv_best = np.zeros(n_gen)
    GA_conv_avg = np.zeros(n_gen)
    for i in range(reps):
        print('rep {}'.format(i))
        RMSE, params, pop, RMSE_list_best, RMSE_list_avg = gen_al(ODE_error, data, t, n_gen=n_gen,
                                                                  n_parents=n_parents,popsize=popsize,p_mutate=p_mutate,evalfunc=eval)
        GA.append(RMSE)
        GA_conv_best = [x + y for x, y in zip(GA_conv_best, np.array(RMSE_list_best) / reps)]
        GA_conv_avg = [x + y for x, y in zip(GA_conv_avg, np.array(RMSE_list_avg) / reps)]
        if i == 0:
            params_GA = params
        elif RMSE < GA[i - 1]:
            params_GA = params
        print(start - time.time())
    conv_dict[eval+'-avg'] = GA_conv_avg
    conv_dict[eval+'-best'] = GA_conv_best
    param_dict[eval] = params_GA

pd.DataFrame({'RMSE':GA[0:reps],'MAE':GA[reps:len(GA)]}).to_csv('Data/Final/GA.csv'.format(eval))
pd.DataFrame(conv_dict).to_csv('Data/Final/GA-conv.csv'.format(eval))
pd.DataFrame(param_dict).to_csv('Data/Final/GA-params.csv'.format(eval))

print('Total simulations took {} seconds'.format(time.time()-start))

