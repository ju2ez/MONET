#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
#

import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


class AUCTracker:
    """Running trapezoidal AUC of a metric against n_evals.

    Call update(n_eval, value) whenever a new data point is available.
    The `auc` attribute holds the current integral ∫ value dn_eval.
    """
    def __init__(self):
        self._prev_n = None
        self._prev_v = None
        self.auc = 0.0

    def update(self, n_eval, value):
        if self._prev_n is not None and n_eval > self._prev_n:
            self.auc += 0.5 * (value + self._prev_v) * (n_eval - self._prev_n)
        self._prev_n = n_eval
        self._prev_v = value
        return self.auc


def gaussian_mutation(x, mutation_std, p_min=0.0, p_max=1.0):
    '''
    Gaussian mutation
    '''
    y = x.copy()
    a = np.random.normal(0, mutation_std * (p_max - p_min), size=len(x))
    y = y + a
    return np.clip(y, p_min, p_max)

def polynomial_mutation(x, p_min=0.0, p_max=1.0):
    '''
    Cf Deb 2001, p 124 ; param: eta_m
    '''
    y = x.copy()
    eta_m = 5.0
    r = np.random.random(size=len(x))
    for i in range(0, len(x)):
        if r[i] < 0.5:
            delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
        else:
            delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
        y[i] += delta_i
    return np.clip(y, p_min, p_max)

def sbx(x, y, p_min=0.0, p_max=1.0):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10.0
    xl = p_min
    xu = p_max
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z


def iso_dd(x, y, p_min=0.0, p_max=1.0, iso_sigma=1./300, line_sigma=20./300):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    assert(x.shape == y.shape)
    a = np.random.normal(0, iso_sigma, size=len(x))
    b = np.random.normal(0, line_sigma)
    z = x.copy() + a + b * (x - y)
    return np.clip(z, p_min, p_max)

def plane_dd(node_0, nodes: list, p_min=0.0, p_max=1.0, iso_sigma=1./300, line_sigma=20./300):
    '''
    Plane-DD variation operator
    Samples on the plane spanned by three points (x, y, z)
    '''

    dim = len(node_0['solution'])
    for node in nodes:
        assert(len(node['solution']) == dim)

    x = node_0['solution']
    y = nodes[0]['solution']
    z = nodes[1]['solution']

    x_fitness = node_0['fitness']
    y_fitness = nodes[0]['fitness']
    z_fitness = nodes[1]['fitness']

    # Compute two basis vectors that span the plane
    v1 = y - x  # First basis vector
    v2 = z - x  # Second basis vector

    # Sample random coefficients for the plane
    # Use fitness weights to bias toward better solutions
    total_fitness = x_fitness + y_fitness + z_fitness
    alpha = np.random.normal(y_fitness / total_fitness, line_sigma)
    beta = np.random.normal(z_fitness / total_fitness, line_sigma)

    # Create a point on the plane: x + alpha*v1 + beta*v2
    plane_point = x + alpha * v1 + beta * v2

    # Add isotropic noise for exploration
    noise = np.random.normal(0, iso_sigma, size=dim)
    offspring = plane_point + noise

    return np.clip(offspring, p_min, p_max)


def iso_dd_mtme(x, y, p_min=0.0, p_max=1.0, iso_sigma=1./300, line_sigma=20./300):
    '''
    Iso+ Noisy-Line
    Ref:
    Multi-task Map-Elites
    GECCO 2020
    '''
    assert(x.shape == y.shape)
    a = np.random.normal(0, iso_sigma, size=len(x))
    b = np.random.normal(0, line_sigma, size=len(x))
    z = x.copy() + a + b * (x - y)
    return np.clip(z, p_min, p_max)

def regression(s, archive, config):
    """ local linear regression  """
    _, idx = archive.tree.query(s, k=1)
    indexes = archive.centroid_neighbors[idx]  # find the direct neighbors using the precomputed delauney from the centroids
    X = [archive.elites[i]["situation"] for i in indexes]
    Y = [archive.elites[i]["command"] for i in indexes]
    reg = LinearRegression().fit(X, Y)
    c = reg.predict(np.array([s]))[0]
    dim = len(c)
    return np.clip(c + np.random.normal(0, config["linreg_sigma"]) * np.std(Y, axis=0), np.zeros(dim), np.ones(dim))

def regression_monet(task, neighbors):
    """ local linear regression for MONET  """
    X = [n["task_config"]["task_vec"] for n in neighbors]
    Y = [n["solution"] for n in neighbors]
    reg = LinearRegression().fit(X, Y)
    c = reg.predict(np.array([task]))[0]
    dim = len(c)
    noise = np.random.normal(0, 1.) * np.std(Y, axis=0)
    return np.clip(c + noise, np.zeros(dim), np.ones(dim))