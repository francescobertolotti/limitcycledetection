import random
import numpy as np

def model_run(prey_growth, predator_growth, prey_chance, energy_cons, energy_per_prey, size_world, init_energy, init_prey, init_predator, max_time):

    def toroidal_fix(x,y):
        if x > size_world: x = x - size_world
        if x < 0: x = size_world + x
        if y > size_world: y = y - size_world
        if y < 0: y = size_world + y
        return x, y

    # Prey class
    class Prey:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def update(self):
            self.x += random.uniform(-1, 1)
            self.y += random.uniform(-1, 1)
            self.x, self.y = toroidal_fix(self.x, self.y)


    # Predator class
    class Predator:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.energy = 0

        def update(self, prey_list, predator_list):
            #movement
            self.x += random.uniform(-1, 1)
            self.y += random.uniform(-1, 1)
            self.x, self.y = toroidal_fix(self.x, self.y)

            #hunt
            prey_in_range = [p for p in prey_list if abs(p.x - self.x) < 2 and abs(p.y - self.y) < 2]
            if len(prey_in_range) > 0 and np.random.random() > prey_chance:
                prey_to_hunt = random.choice(prey_in_range)
                prey_list.remove(prey_to_hunt)
                self.energy += energy_per_prey

            #starve
            self.energy -= energy_cons
            if self.energy < 0: 
                predator_list.remove(self)

    # Setup
    prey_population = [Prey(random.randint(0, 100), random.randint(0, 100)) for i in range(init_prey)]
    predator_population = [Predator(random.randint(0, 100), random.randint(0, 100)) for i in range(init_predator)]
    for predator in predator_population: predator.energy = init_energy

    # Metrics
    prey_ts, predator_ts = [],[]

    # Simulation
    for t in range(max_time):
        # Update populations
        for prey in prey_population: prey.update()
        for predator in predator_population: predator.update(prey_population, predator_population)

        # Reproduce
        new_prey = [Prey(random.randint(0, 100), random.randint(0, 100)) for i in range(int(len(prey_population) * prey_growth))]
        prey_population += new_prey

        n_new_predators = np.random.poisson(len(predator_population) * predator_growth)
        for _ in range(n_new_predators):
            father = random.choice(predator_population)
            new_predator = Predator(random.randint(0, 100), random.randint(0, 100))
            # Split energy of the father
            new_predator.energy = father.energy / 2
            father.energy = father.energy / 2
            # Update predators population
            predator_population.append(new_predator)

        # Update Globals
        prey_ts.append(len(prey_population))
        predator_ts.append(len(predator_population))

        # Exit
        if len(prey_population) > 10000: return prey_ts, predator_ts
        if len(predator_population) == 0: return prey_ts, predator_ts
        if len(prey_population) == 0: return prey_ts, predator_ts


    return prey_ts, predator_ts