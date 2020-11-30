import time
import os
import numpy as np
import simpy
import pandas as pd


class Queue(object):
    def __init__(self, env, num_servers, mu, dist='exp'):
        self.env = env
        self.server = simpy.Resource(env, capacity=num_servers)
        self.prio_server = simpy.PriorityResource(env, capacity=num_servers)
        self.mu = mu
        self.dist = dist
#
#
#     def perform_task(self, task, mu,dist='exp'):
#         if dist == 'exp':
#             t = np.random.exponential(mu)
#         elif dist == 'hyperexp':
#             p = np.random.normal(0,1)
#             if p > 0.75:
#                 t = np.random.exponential(mu/5)
#             else:
#                 t = np.random.exponential(mu)
#         else:
#             t = mu
#         yield self.env.timeout(t)
#         # print("%s performed in %.2f minutes." %(task,t))
#
# def task(env, name, qu, mu, priority=False, dist='exp'):
#     # Arriving task, requesting server, waiting and processing
#     # Either with or without priority
#
#     # req = system.servers.request(priority = t_departure)
#
#
#     t_arrive = env.now
#     if priority:
#         with qu.prio_server.request() as request:
#             yield request
#             waiting_times.append(env.now - t_arrive)
#             yield env.process(qu.perform_task(name,mu,dist))
#             sys_times.append(env.now - t_arrive)
#     else:
#         with qu.server.request() as request:
#             yield request
#             waiting_times.append(env.now - t_arrive)
#             yield env.process(qu.perform_task(name, mu,dist))
#             sys_times.append(env.now - t_arrive)

def task(env, name, qu, mu, priority=False, dist='exp'):
    # Arriving task, requesting server, waiting and processing
    # Either with or without priority
    if dist == 'exp':
        t = np.random.exponential(mu)
    elif dist == 'hyperexp':
        p = np.random.normal(0,1)
        if p > 0.75:
            t = np.random.exponential(mu/5)
        else:
            t = np.random.exponential(mu)
    elif dist == 'erlang':
        t = 0
        for i in range(5):
            t += np.random.exponential(mu)
        t /= 5
    else:
        t = mu

    t_arrive = env.now
    if priority:
        with qu.prio_server.request(t) as request:
            yield request
            waiting_times.append(env.now - t_arrive)
            yield env.timeout(t)
            sys_times.append(env.now - t_arrive)
    else:
        with qu.server.request() as request:
            yield request
            waiting_times.append(env.now - t_arrive)
            yield env.timeout(t)
            sys_times.append(env.now - t_arrive)


def setup(env, num_servers, mu, lam,priority=False, dist='exp'):
    # Setting up the arrival of tasks during simulation

    qu = Queue(env, num_servers, mu, dist)
    i = 0
    while True:
        yield env.timeout(np.random.exponential(1/lam))
        i += 1
        env.process(task(env, 'Task %d' % i, qu, mu, priority=priority, dist=dist))

#Set params
seedy = 42
np.random.seed(seedy)
NUM_SERVERS_LIST = [1, 2, 4, 8]                            # Number of servers
NUM_SERVERS_LIST2 = [1, 2]                            # Short number of servers
MU = 1                                                 # Mu is set to 1 for clarity
LAM_LIST = [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]   # Lambda varies to vary rho
DISTRIBUTIONS = ['deterministic','hyperexp','erlang']      # Service distributions
SIM_TIME = 20000                                       # Simulation time
NUM_SIM = 300                                     # Number of replications


start = time.time()
df_wait = pd.DataFrame({}) # Create empty dataframe

# 1. Loop over various lambda and n for exponential distribution
for NUM_SERVERS in NUM_SERVERS_LIST:
    for LAM in LAM_LIST:
        avg_wait = []
        avg_sys = []
        for i in range(NUM_SIM):
            waiting_times = []
            sys_times = []
            env = simpy.Environment()
            env.process(setup(env, NUM_SERVERS, MU, LAM*NUM_SERVERS))
            env.run(until=SIM_TIME)
            avg_wait.append(np.mean(waiting_times))
        print(time.time()-start)
        df_wait[str(('exp',NUM_SERVERS,LAM))] = avg_wait
print('First loop done in '+str((time.time()-start)/60)+' minutes')

# 2. Loop over all lambdas for n=1 and priority scheduling
NUM_SERVERS = 1
for LAM in LAM_LIST:
    avg_wait = []
    avg_sys = []
    for i in range(NUM_SIM):
        waiting_times = []
        sys_times = []
        env = simpy.Environment()
        env.process(setup(env, NUM_SERVERS, MU, LAM*NUM_SERVERS,priority=True))
        env.run(until=SIM_TIME)
        avg_wait.append(np.mean(waiting_times))
    print(time.time()-start)
    df_wait[str(('priority',NUM_SERVERS,LAM))] = avg_wait
print('Second loop done after '+str((time.time()-start)/60)+' minutes')


# 3. Loop over all lambdas, n=[1,2] and various distributions
for DIST in DISTRIBUTIONS:
    for NUM_SERVERS in NUM_SERVERS_LIST2:
        for LAM in LAM_LIST:
            avg_wait = []
            avg_sys = []
            for i in range(NUM_SIM):
                waiting_times = []
                sys_times = []
                env = simpy.Environment()
                env.process(setup(env, NUM_SERVERS, MU, LAM*NUM_SERVERS,dist=DIST))
                env.run(until=SIM_TIME)
                avg_wait.append(np.mean(waiting_times))
            print(time.time()-start)
            df_wait[str((DIST,NUM_SERVERS,LAM))] = avg_wait

print('Third loop done after '+str((time.time()-start)/60)+' minutes')


#Save dataframe of simulations to csv, to make graphs in jupyter
time = time.time() - start
path = os.getcwd()+'/Data/'
os.chdir(path)
df_wait.to_csv(r'df_waittime.csv')




