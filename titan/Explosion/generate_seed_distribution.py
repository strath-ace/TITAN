#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can revolumesbute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is volumesbuted in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""generate_seed_distribution module."""
import numpy as np
import psutil
from scipy.spatial import KDTree
from scipy.stats import uniform, linregress, multivariate_normal, norm
from scipy.optimize import minimize, dual_annealing, Bounds, basinhopping, direct, differential_evolution
from datetime import datetime as dt
from matplotlib import pyplot as plt
from functools import partial

class SpiralSampler():
    """SpiralSampler."""
    def __init__(self, a=1.0,b=1.0,k=1.0,m=1.0, rng=None):
        """Documentation for the function.
:param a: Value for a.
:type a: Any
:param b: Value for b.
:type b: Any
:param k: Value for k.
:type k: Any
:param m: Value for m.
:type m: Any
:param rng: Value for rng.
:type rng: Any"""
        self.a=a
        self.b=b
        self.k=k
        self.m=m
        if rng is None: self.rng = np.random.RandomState(dt.now().microsecond)
        else: self.rng = rng
        self.parameter = uniform()
        self.parameter.random_state = self.rng
    def rvs(self, n_samples, kind='xyz'):
        """Documentation for the function.
:param n_samples: Integer value for n samples.
:type n_samples: int
:param kind: Value for kind.
:type kind: Any
:return: Return value.
:rtype: Any"""
        theta = self.parameter.rvs(n_samples)
        r = self.a * np.exp( self.b * theta)
        phi = self.k * theta
        psi = self.m * theta
        if kind=='polar': return np.array([r, phi, psi])
        else:
            x = r * np.cos(psi) * np.cos(phi)
            y = r * np.cos(psi) * np.sin(phi)
            z = r * np.sin(phi)
            return np.array([x,y,z]).T

def generate_seed_points(obj_half_len=2, parameters = [0, 0, 0], law = [6, -1.6]):
    """Documentation for the function.
:param obj_half_len: Value for obj half len.
:type obj_half_len: Any
:param parameters: Value for parameters.
:type parameters: Any
:param law: Value for law.
:type law: Any
:return: Return value.
:rtype: Any"""
    n_points, n_shells, min_len = parameters
    shells = np.logspace(min_len,np.log(obj_half_len),int(n_shells))
    points = np.zeros([1,3])
    scale = 50
    shell_points = np.floor(scale*np.exp(-shells))
    while not np.sum(shell_points)==n_points:
        if np.sum(shell_points)<n_points: scale*=1.1
        else: scale*=0.9
        shell_points = np.floor(scale*law[0]*shells**law[1])
    for shell_rad, n_at_shell in zip(shells, shell_points):
        #n_at_shell = int(50*np.exp(-shell_rad))
        points_on_shell = trigonometric_2sphere_sampling(shell_rad, int(n_at_shell))
        points = np.vstack([points, points_on_shell])
    return points[1:,:]
    # tree = KDTree(points)
    # dists, _ = tree.query(points,k=2)
    # dists = dists[:,1]

    # cum = []
    
    # bins = np.logspace(np.log10(np.min(dists)),np.log10(np.max(dists)),250, endpoint=False)
    # #emp = law_coefficient * bins ** law_exponent
    # for q_dist in bins: cum.append(len(np.where(dists>q_dist)[0]))
    # law = linregress(np.log10(bins),np.log10(cum))
    # print(10**law.intercept, law.slope)
    # print(abs(law.slope-desired_slope))
    # if not give_points: return abs(law.slope-desired_slope)
    # else:
    #     plot_result(cum, points, bins, desired_slope, law)
    #     return points
    
    

    
def get_seed_points(expl_dir, obj_len=2, law = [6, -1.6]):
    """Documentation for the function.
:param expl_dir: Value for expl dir.
:type expl_dir: str
:param obj_len: Value for obj len.
:type obj_len: Any
:param law: Value for law.
:type law: Any"""
    seed_func = partial(generate_seed_points, obj_len, law[1], False)
    opt = minimize(seed_func,[64,20,-3],bounds=[[32,128],[3,32],[-5,-1]],tol=1e-2)
    print(opt.message)
    print(opt.x)
    points = generate_seed_points(obj_len, law[1], give_points=True, parameters=opt.x)
    np.savetxt(expl_dir+'/points.csv',points,delimiter=',')
def plot_result(points, desired_law, regress_law):
    """Documentation for the function.
:param points: Value for points.
:type points: list[float]
:param desired_law: Value for desired law.
:type desired_law: Any
:param regress_law: Value for regress law.
:type regress_law: Any"""
    k = min(points.shape[0], 16) #int(np.min([100,np.floor(0.9*points.shape[0])]))
    dists = get_dists(points, k=max([k,2]))

    cum = []
    bins = np.logspace(np.log10(np.min(dists)),np.log10(np.max(dists)),250, endpoint=False)
    for q_dist in bins: cum.append(len(np.where(dists>q_dist)[0]))


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(bins, cum, label='Generated Points')
    ax.plot(bins, desired_law[0]*bins**desired_law[1], label='Desired Law')

    ax.plot(bins, regress_law[0]*bins**regress_law[1], label='Fitted Law')
    ax.legend()
    fig.suptitle('kNN Distance Law k={}'.format(k))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(np.max(points, axis=0) - np.min(points, axis=0))
    #ax.set_aspect('equal')
    ax.scatter3D(points[:,0], points[:,1], points[:,2], marker='*')
    plt.show()


def trigonometric_2sphere_sampling(rad, n_points, rng=None):
    """Documentation for the function.
:param rad: Value for rad.
:type rad: Any
:param n_points: Integer value for n points.
:type n_points: int
:param rng: Value for rng.
:type rng: Any
:return: Return value.
:rtype: Any"""
    if rng is None: rng = np.random.RandomState(dt.now().microsecond)
    z = uniform(loc=-1,scale=2).rvs(size=n_points,random_state=rng)
    t = uniform(loc=0,scale=2*np.pi).rvs(size=n_points,random_state=rng)

    r = np.sqrt(1-z**2)
    x = r * np.cos(t)
    y = r * np.sin(t)
    points = rad * np.vstack([x,y,z]).T
    return points


def get_law_from_points(ref_law = [6,-1.6], return_law=False, weights=[0.1,1,0.1],X=None):
    """Documentation for the function.
:param ref_law: Value for ref law.
:type ref_law: Any
:param return_law: Value for return law.
:type return_law: Any
:param weights: Value for weights.
:type weights: Any
:param X: Numeric value for x.
:type X: float
:return: Return value.
:rtype: Any"""
    X = X.reshape([-1,3])
    k = min(X.shape[0],16)#int(np.min([100,np.floor(0.9*X.shape[0])]))
    dists = get_dists(X,k=max([k,2]))
    
    law = law_linregress(dists)
    print('{}N^[{}] R2={}'.format(np.round(10**law.intercept,4),np.round(law.slope,4),np.round(law.rvalue**2,4)))
    delta = [weights[0] * (ref_law[0]-10**law.intercept),
             weights[1] * (ref_law[1]-law.slope), 
             weights[2] * (1-law.rvalue**2)]
    if return_law: return [10**law.intercept, law.slope, law.rvalue**2]
    if np.any(np.isnan(delta)): return 1e5
    return np.linalg.norm(delta)

def law_linregress(dists):
    """Documentation for the function.
:param dists: Value for dists.
:type dists: Any
:return: Return value.
:rtype: Any"""
    cum = []
    bins = np.logspace(np.log10(np.min(dists)),np.log10(np.max(dists)),250, endpoint=False)
    for q_dist in bins: cum.append(len(np.where(dists>q_dist)[0]))
    law = linregress(np.log10(bins),np.log10(cum),nan_policy='omit')
    return law

def get_dists(X, k=2):
    """Documentation for the function.
:param X: Numeric value for x.
:type X: float
:param k: Value for k.
:type k: Any
:return: Return value.
:rtype: Any"""
    tree = KDTree(X)
    dists, _ = tree.query(X,k=k)
    dists = np.mean(dists[:,1:],axis=1)
    return dists

def distribution_generator(desired_law, return_law, weights = [0.1,1,0.1], n_frags=None, n_tests = 10, parameter_vector = None):
    """Documentation for the function.
:param desired_law: Value for desired law.
:type desired_law: Any
:param return_law: Value for return law.
:type return_law: Any
:param weights: Value for weights.
:type weights: Any
:param n_frags: Integer value for n frags.
:type n_frags: int
:param n_tests: Integer value for n tests.
:type n_tests: int
:param parameter_vector: Value for parameter vector.
:type parameter_vector: Any
:return: Return value.
:rtype: Any"""
    if parameter_vector is None or n_frags is None: raise Exception('Must provide parameters!')
    mean = np.zeros(3)
    Lower = np.diag(parameter_vector[:3])
    Lower[1,0] = parameter_vector[3]
    Lower[2,:2] = parameter_vector[4:]
    COV = Lower @ Lower.T
    mvn = multivariate_normal(mean=mean, cov=COV, allow_singular=True)
    test_points = [mvn.rvs(n_frags) for _ in range(n_tests)]
    deltas = []
    for i_test in range(n_tests):
        deltas.append(get_law_from_points(ref_law=desired_law, return_law=return_law, weights=weights, X=test_points[i_test]))
    return np.mean(deltas)

def log_spiral_3d(desired_law, return_law, weights = [0.1,1,0.1], n_frags=None, n_tests = 10, rng = None, parameter_vector = None,):
    """Documentation for the function.
:param desired_law: Value for desired law.
:type desired_law: Any
:param return_law: Value for return law.
:type return_law: Any
:param weights: Value for weights.
:type weights: Any
:param n_frags: Integer value for n frags.
:type n_frags: int
:param n_tests: Integer value for n tests.
:type n_tests: int
:param rng: Value for rng.
:type rng: Any
:param parameter_vector: Value for parameter vector.
:type parameter_vector: Any
:return: Return value.
:rtype: Any"""
    b, k, m = parameter_vector
    print(b,k,m)
    a = 1
    sampler = SpiralSampler(a, b, k, m, rng=rng)
    test_points = [sampler.rvs(n_frags) for _ in range(n_tests)]
    deltas = []
    for i_test in range(n_tests):
        deltas.append(get_law_from_points(ref_law=desired_law, return_law=return_law, weights=weights, X=test_points[i_test]))
    return np.mean(deltas)

def optimal_seeds(n_fragments=24, method='anneal', desired_law = [6, -1.6], plot=True, obj_len=2, CoG=[0,0,0], compute_budget=2e5, rng=None):
    """Documentation for the function.
:param n_fragments: Integer value for n fragments.
:type n_fragments: int
:param method: Value for method.
:type method: Any
:param desired_law: Value for desired law.
:type desired_law: Any
:param plot: Value for plot.
:type plot: Any
:param obj_len: Value for obj len.
:type obj_len: Any
:param CoG: Value for cog.
:type CoG: Any
:param compute_budget: Value for compute budget.
:type compute_budget: Any
:param rng: Value for rng.
:type rng: Any
:return: Return value.
:rtype: Any"""
    
    match method:
        case 'anneal':
            points = generate_seed_points(obj_half_len=0.5*obj_len, parameters=[n_fragments,5,-3])
            n_points = points.shape[0]
            lb = np.full_like(points, -0.5*obj_len).flatten()
            ub = np.full_like(points, 0.5*obj_len).flatten()
            optfunc = partial(get_law_from_points, desired_law, False, [0.1,1.2,2])
            opt = dual_annealing(func=optfunc,bounds=Bounds(lb=lb,ub=ub),
                                 x0=points.flatten(), maxfun=compute_budget)
            print(opt.message)
            print(opt.x)
            out_points = np.reshape(opt.x,[-1,3])
        case 'direct':
            opt = direct(func=optfunc,bounds=Bounds(lb=lb,ub=ub), maxfun=int(compute_budget))
            print(opt.message)
            print(opt.x)
            out_points = np.reshape(opt.x,[-1,3])
        case 'distri':
            optfunc = partial(distribution_generator, desired_law, False, [0.0,0.0,1.0], n_fragments, 100)
            x0 = [1.0,1.0,1.0,0.0,0.0,0.0]
            bounds = [ (1e-3, 10.0),
                       (1e-3, 10.0),
                       (1e-3, 10.0),
                       (-10.0, 10.0),
                       (-10.0, 10.0),
                       (-10.0, 10.0)]
            #opt = dual_annealing(func =optfunc, bounds = bounds, x0=x0,maxfun=compute_budget)
            opt = direct(func=optfunc, bounds=bounds, maxfun=int(compute_budget))
            print(opt.message)
            print(opt.x)
            parameter_vector = opt.x
            Lower = np.diag(parameter_vector[:3])
            Lower[1,0] = parameter_vector[3]
            Lower[2,:2] = parameter_vector[4:]
            COV = Lower @ Lower.T
            print('Output Covariance as... {}'.format(COV))
            out_points = multivariate_normal([0,0,0],COV).rvs(n_fragments)
        case 'spiral':
            optfunc = partial(log_spiral_3d, desired_law, False, [2,5,9], n_fragments, 50, rng)
            bounds = [ (0.1, 10),
                       (20.0, 100.0),
                       (20.0, 100.0)]
            x0 = [5,50,50]
            #opt = direct(func=optfunc, bounds=bounds, maxfun=int(compute_budget), vol_tol=1e-24)
            opt = dual_annealing(func =optfunc, bounds = bounds, x0=x0,maxfun=compute_budget, rng=rng)
            #opt = differential_evolution(func =optfunc, bounds = bounds, maxiter=compute_budget, rng=rng, workers=psutil.cpu_count())
            print(opt.message)
            print(opt.x)
            out_points = SpiralSampler(1,opt.x[0],opt.x[1],opt.x[2]).rvs(1000)
            
    
    law_out = get_law_from_points(X=out_points, return_law=True, ref_law=desired_law)


    print('Final law value of {}N^[{}] R2={}'.format(law_out[0],law_out[1],law_out[2]))
    gen_points = out_points + np.full_like(out_points, CoG)

    SF = np.max(np.linalg.norm(gen_points, axis=1))/obj_len

    if plot: plot_result(out_points, desired_law, law_out)
    return [1/SF, opt.x[0], opt.x[1], opt.x[2]]
