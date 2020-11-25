# from simpy import *
# from random import expovariate, seed
#
# class Generate(Process):
#     """ Generates tasks randomly """
#
#     def generate(self, num_tasks, lam, server):
#         for i in range(num_tasks):
#             c = Task(name="Task %d" % (i,))
#             activate(c, c.perform(server=server))
#             t = expovariate(lam)
#             yield hold, self, t
#
# class Task(Process):
#     """ Task arrives and is performed """
#
#     def perform(self, server):
#         arrive = now()
#         print
#         "%8.4f %s: Here I am     " % (now(), self.name)
#         yield request, self, server
#         wait = now() - arrive
#         print
#         "%8.4f %s: Waited %6.3f" % (now(), self.name, wait)
#         tib = expovariate(mu)
#         yield hold, self, tib
#         yield release, self, server
#         print
#         "%8.4f %s: Finished      " % (now(), self.name)
#
# ## Experiment data -------------------------
#
# num_tasks = 5
# maxTime = 400.0  # minutes
# mu = 12.0  # mean, minutes
# lam = 10.0  # mean, minutes
# theseed = 12345
#
# ## Model/Experiment ------------------------------
#
# seed(theseed)
# k = Resource(capacity=1)#name="Servers", unitName="Server1")
#
# initialize()
# s = Generate('Source')
# activate(s, s.generate(num_tasks=maxNumber, lambda=ARRint,server=k), at=0.0)
# simulate(until=maxTime)