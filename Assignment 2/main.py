import time
import simpy
import numpy as np


class Queue(object):
    """A carwash has a limited number of machines (``NUM_MACHINES``) to
    clean cars in parallel.

    Cars have to request one of the machines. When they got one, they
    can start the washing processes and wait for it to finish (which
    takes ``washtime`` minutes).

    """
    def __init__(self, env, num_servers, mu):
        self.env = env
        self.server = simpy.Resource(env, capacity=num_servers)
        self.prio_server = simpy.PriorityResource(env, capacity=num_servers)
        self.mu = mu

    def perform_task(self, task, mu):
        t = np.random.exponential(mu)
        yield self.env.timeout(t)
        # print("%s performed in %.2f minutes." %(task,t))

def task(env, name, qu,mu,priority=False):
    t_arrive = env.now
    if priority:
        with qu.prio_server.request() as request:
            yield request
            waiting_times.append(env.now - t_arrive)
            yield env.process(qu.perform_task(name,mu))
            system_times.append(env.now - t_arrive)
    else:
        with qu.server.request() as request:
            yield request
            waiting_times.append(env.now - t_arrive)
            yield env.process(qu.perform_task(name, mu))
            system_times.append(env.now - t_arrive)


def setup(env, num_servers, mu, lam,priority=False):
    """Create a carwash, a number of initial cars and keep creating cars
    approx. every ``t_inter`` minutes."""
    qu = Queue(env, num_servers, mu)

    i = 0
    while True:
        yield env.timeout(np.random.exponential(1/lam))
        i += 1
        env.process(task(env, 'Task %d' % i, qu,mu,priority=priority))


# Setup and start the simulation
print('Queue')
#Set params
seedy = 42
NUM_SERVERS_LIST = [1, 2, 4]  # Number of servers
MU = 1
LAM_LIST = [0.95,0.9, 0.85, 0.8, 0.75]
SIM_TIME = 1000    # Simulation time

# Create an environment and start the setup process
np.random.seed(seedy)  # This helps reproducing the results

start = time.time()
results_wait = {}
results_system = {}

for NUM_SERVERS in NUM_SERVERS_LIST:
    for LAM in LAM_LIST:
        avg_wait = []
        avg_system = []
        for i in range(100):
            waiting_times = []
            system_times = []

            env = simpy.Environment()
            env.process(setup(env, NUM_SERVERS, MU, LAM*NUM_SERVERS,priority=True))
            env.run(until=SIM_TIME)
            avg_wait.append(np.mean(waiting_times))
            avg_system.append(np.mean(system_times))
        print(time.time()-start)
        results_wait[(str(NUM_SERVERS),str(LAM))] = avg_wait
        results_system[(str(NUM_SERVERS),str(LAM))] = avg_system

time = time.time() - start

print(start)


