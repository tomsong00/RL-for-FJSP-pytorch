class Parameters(object):
    def __init__(self):
        self.beta = None
        self.alpha =None
        self.population_size = None
        self.max_generation = None
        self.length=None
        self.gobal_best_fitness=None


    def parameter_setting(self):
        self.max_generation = 10
        self.population_size = 80
        self.alpha = 0.95
        self.beta = 0.05
        self.length=3
        self.gobal_best_fitness=999999
        return self.max_generation, self.population_size, self.alpha, self.beta, self.gobal_best_fitness,self.length
