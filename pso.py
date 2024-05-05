import pyswarms as ps
import random
from math import cos, sin, cosh, sinh, sqrt

import numpy as np

from pyswarms.backend.topology import Pyramid, Ring, Star, Random, VonNeumann


class PSOModel:

    def __init__(self, target):
        self.topology = Ring()
        self.k = -5
        self.target = target
        self.nop = 100
        self.nop_best_of_best = 200
        self.iters = 1500
        self.iters_best_of_best = 1300
        self.iters_modele_locatii = 600

        self.options = {'c1': 3, 'c2': 0.25, 'w': 0.5, 'k': 50, 'p': 2}  # cel mai bun pana acum
        self.options_best_of_best = {'c1': 3, 'c2': 0.20, 'w': 0.4, 'k': 40, 'p': 2}  # cel mai bun pana acum

        self.bounds = ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.6, 0.6])  # 0.3
        self.boundsLocMica = ([0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.6, 0.6])  # 0.3

        self.intervale_cautare_locatii = [0.0, 0.1, 0.5, 0.9, 1.0]

        self.nr_optimizers = 3
        self.nr_optimizers_loc_mica = 1
        self.nr_optimizers_general = 1

        self.rezPartiale = []
        self.costPartiale = []

    def get_severity(self, adancime):
        return 0.0009 * adancime ** 6 - 0.0047 * adancime ** 6 + 0.0096 * adancime ** 4 - 0.0083 * adancime ** 3 + 0.0063 * adancime ** 2 - 0.0003 * adancime

    def get_incastrare(self, adancime2):
        return (
                0.0009 * adancime2 ** 6 - 0.0047 * adancime2 ** 6 + 0.0096 * adancime2 ** 4 - 0.0083 * adancime2 ** 3 + 0.0063 * adancime2 ** 2 - 0.0003 * adancime2)

    def get_lambda(self, mode):
        return [1.875104, 4.694091, 7.854757, 10.99554, 14.137168, 17.278759, 20.420352,
                23.561944, 26.703537556, 29.845130209][mode - 1]

    def phi(self, mode, x):
        li = self.get_lambda(mode)

        return 0.5 * (cos(li * x) + cosh(li * x) - ((cos(li) + cosh(li)) / (sin(li) + sinh(li))) * (
                sin(li * x) + sinh(li * x)))

    def rfs(self, location1, location2, adancime1, adancime2):
        return [
            self.get_incastrare(adancime1) * self.phi(mode, location1) ** 2 + self.get_severity(adancime2) * self.phi(
                mode, location2) ** 2
            for mode in range(1, 9)]  # mode era pana la 9

    def fitness(self, x):
        vect = np.array([])
        for p in x:
            # pozitie, adancime = p
            r = self.rfs(*p)

            vect = np.append(vect, -(sum((r[mod] - self.target[mod]) ** 2 for mod in range(8)) ** self.k))

        return vect

    def generate_diverse_particles_with_best(self, best_solutions, n_particles=200,
                                             bounds=([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.6, 0.6])):

        new_particles = best_solutions.copy()

        n_best_solutions = len(best_solutions)

        additional_particles_needed = n_particles - n_best_solutions

        print("add", additional_particles_needed)
        print("n_best", n_best_solutions)

        step_sizes = [(bounds[1][i] - bounds[0][i] + 3) / (additional_particles_needed // n_best_solutions) for i in
                      range(4)]

        for solution in best_solutions:
            for _ in range(additional_particles_needed // n_best_solutions):
                new_solution = [
                    np.clip(solution[i] + np.random.uniform(-step_sizes[i], step_sizes[i]), bounds[0][i], bounds[1][i])
                    for
                    i in range(4)]
                new_particles.append(new_solution)

        while len(new_particles) < n_particles:
            base_solution = random.choice(best_solutions)
            new_solution = [
                np.clip(base_solution[i] + np.random.uniform(-step_sizes[i], step_sizes[i]), bounds[0][i], bounds[1][i])
                for
                i in range(4)]
            new_particles.append(new_solution)

        for i in range(len(new_particles)):
            new_particles[i] = np.array(new_particles[i])
        return np.array(new_particles)

    def get_bounds(self, new_population):
        minim = [min([pop[i] for pop in new_population]) for i in range(4)]
        maxim = [max([pop[i] for pop in new_population]) for i in range(4)]
        new_bounds = (minim, maxim)
        return new_bounds

    def run_optimizer(self, optimizer):
        cost, pos = optimizer.optimize(self.fitness, iters=self.iters)
        self.rezPartiale.append(pos)
        self.costPartiale.append(cost)

    def predict(self):
        bestcost = 0
        bestBoundsLoc = None

        for loc1 in range(len(self.intervale_cautare_locatii) - 1):
            for loc2 in range(loc1, len(self.intervale_cautare_locatii) - 1):
                bounds_partiale = ([self.intervale_cautare_locatii[loc1], self.intervale_cautare_locatii[loc2], 0.0, 0.0],
                                   [self.intervale_cautare_locatii[loc1 + 1], self.intervale_cautare_locatii[loc2 + 1], 0.6, 0.6])
                print(bounds_partiale)
                opt_part = ps.single.GeneralOptimizerPSO(n_particles=self.nop, dimensions=4, options=self.options,
                                                         bounds=bounds_partiale, topology=self.topology)

                cost, pos = opt_part.optimize(self.fitness, iters=self.iters_modele_locatii)
                if cost < bestcost:
                    bestBoundsLoc = bounds_partiale
                    bestcost = cost

        optimizers = []

        for _ in range(self.nr_optimizers):
            optimizers.append(ps.single.GeneralOptimizerPSO(n_particles=self.nop, dimensions=4, options=self.options,
                                                            bounds=bestBoundsLoc, topology=self.topology))
        for _ in range(self.nr_optimizers_general):
            optimizers.append(ps.single.GeneralOptimizerPSO(n_particles=self.nop, dimensions=4, options=self.options,
                                                            bounds=self.bounds, topology=self.topology))

        if (bestBoundsLoc[0][0] == 0.0 and bestBoundsLoc[0][1] != 0.0) or (
                bestBoundsLoc[0][0] != 0.0 and bestBoundsLoc[0][1] == 0.0):
            if bestBoundsLoc[0][0] >= 0.5 or bestBoundsLoc[0][1] >= 0.5:
                for _ in range(self.nr_optimizers_loc_mica):
                    optimizers.append(
                        ps.single.GeneralOptimizerPSO(n_particles=self.nop, dimensions=4, options=self.options,
                                                      bounds=self.boundsLocMica, topology=self.topology))

        for indice in range(len(optimizers)):
            self.run_optimizer(optimizers[indice])

        new_generation = self.generate_diverse_particles_with_best(self.rezPartiale, bounds=self.bounds)
        new_bounds = self.get_bounds(new_generation)

        final_optimizer = ps.single.GeneralOptimizerPSO(n_particles=self.nop_best_of_best, dimensions=4, options=self.options_best_of_best,
                                                        bounds=new_bounds, topology=self.topology, init_pos=new_generation)

        cost_best_of_best, pos_best_of_best = final_optimizer.optimize(self.fitness, iters=self.iters_best_of_best)
        return cost_best_of_best, pos_best_of_best
