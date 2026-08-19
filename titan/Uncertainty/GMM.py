#
# Copyright (c) 2026 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN 
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""GMM module."""
import numpy as np
from copy import copy
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
from datetime import datetime as dt

class recursive_gaussian_mixture():
    """A recursive Gaussian Mixture Model (GMM) defined based upon splitting libraries of Gaussians recursively. 
    """
    def __init__(self, mean : np.ndarray, cov : np.ndarray, weight = 1.0, is_leaf = True, library_size = 3, tree_size = 1, rng = np.random.RandomState(dt.now().microsecond), sigma_parameters = [1e-3,2,0]):
        """Create a new GMM node, by default this creates a single node GMM, equivalent to a standard normal distribution but equipped with the split_leaf() method
        :param mean: Vector describing the mean of the distribution
        :type mean: np.ndarray
        :param cov: Matrix describing the covariance of the distribution
        :type cov: np.ndarray
        :param weight: Weight of the Gaussian inside the GMM, defaults to 1.0
        :type weight: float, optional
        :param is_leaf: Does this GMM have no children, i.e. is a leaf node? Defaults to True
        :type is_leaf: bool, optional
        :param library_size: Number of Gaussians to split nodes into, defaults to 3
        :type library_size: int, optional
        :param tree_size: Number of nodes in the tree, defaults to 1
        :type tree_size: int, optional
        :param rng: Random state, defaults to  np.random.RandomState(dt.now().microsecond)
        :type rng: np.random.RandomState, optional
        :param sigma_parameters: Sigma point generator parameters (Alpha, Beta, Kappa), defaults to [1e-3,2,0]
        :type sigma_parameters: list, optional
"""
        ## Statistical Parameters of Gaussian
        self.mean = np.array(mean)
        self.cov = cov
        self.empirical_cov = None
        self.dim = len(self.mean.flatten())
        self.distribution = None

        ## Node Parameters of Tree
        self.n_leaf_nodes = tree_size
        self.rng = rng
        self.leaf_list = [] if not is_leaf else [self]
        self.is_leaf = is_leaf
        self.children = []

        ## Parameters of Mixture
        self.library_size = library_size
        self.weight = weight
        self.shannon_entropy = []
        self.dynamical_entropy = None

        #: Standard splitting libraries with 3 and 5 from DeMars et al, doi.org/10.2514/1.58987,
        #: 4 element library from Huber et al, doi.org/10.1109/MFI.2008.4648062
        self.libraries = {3:{'weights' : np.array([ 0.2252246249,  0.5495507502,  0.2252246249]),
                             'means'   : np.array([-1.0575154615,  0.0,           1.0575154615]),
                             'std'     :   0.6715662887},
                          4:{'weights' : np.array([ 0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098]),
                             'means'   : np.array([-1.4131205233, -0.44973059608, 0.44973059608, 1.4131205233]),
                             'std'     :   0.51751260421},
                          5:{'weights' : np.array([ 0.0763216491,  0.2474417860,  0.3524731300,  0.2474417860, 0.0763216491]),
                             'means'  :  np.array([-1.6899729111, -0.8009283834,  0.0,           0.8009283834, 1.6899729111]),
                             'std'    :   0.4422555386}}
        # Sigma point parameters
        self.sig_params = sigma_parameters
        self.point_generator = MerweScaledSigmaPoints(self.dim, 
                                                      self.sig_params[0], 
                                                      self.sig_params[1], 
                                                      self.sig_params[2])
        if not is_leaf:
            self.split_self()
        else: self.update_distribution()

    def get_mixture_parameters(self,rootcov : np.ndarray):
        """Computes new mixture parameters that would result from splitting the gaussian along
        :param rootcov: Covariance of the root node to be split
        :type rootcov: np.ndarray
:return: Weights, Means, Covariance
:rtype: np.ndarray, np.ndarray, np.ndarray
"""

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(rootcov)
        split_axis = len(self.eigenvalues)-1
        weights = self.weight * self.libraries[self.library_size]['weights']
        component_eigenvalues= copy(self.eigenvalues)
        component_eigenvalues[-1] = component_eigenvalues[-1]*(self.libraries[self.library_size]['std'])**2
        cov = self.eigenvectors @ np.diag(component_eigenvalues) @ self.eigenvectors.transpose()
        means = []
        for i_component in range(self.library_size):
            means.append(self.mean + np.sqrt(self.eigenvalues[-1]) * self.libraries[self.library_size]['means'][i_component] * self.eigenvectors[:,split_axis])
        return weights, means, cov
    
    def get_shannon_entropy_change(self) -> float:
        """Computes the Shannon Entropy of the distribution, appends it to the list of entropies and returns the delta from the previous entropy
:return: dH, the change in Shannon Entropy
:rtype: float
"""
        self.shannon_entropy.append(0.5 * self.dim * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.cov)) + self.dim/2)
        dH = self.shannon_entropy[-1] - self.shannon_entropy[-2] if len(self.shannon_entropy)>1 else None
        return dH
    
    def update_distribution(self):
        """Recomputes the base Gaussian of the node and does an empirical covariance calculation"""
        self.distribution = multivariate_normal(self.mean,self.cov, allow_singular = True)
        self.empirical_cov = self.run_empirical_cov()

    def rvs(self,n=1) -> np.ndarray:
        """Sample the GMM recursively to get an array of n samples
        :param n: Number of samples, defaults to 1
        :type n: int, optional
:return: Resultant sample array
:rtype: np.ndarray
"""
        if self.is_leaf: 
            result = self.distribution.rvs(n)
        else:
            result = np.zeros_like(self.mean)
            probabilities = [leaf.weight for leaf in self.leaf_list]
            probabilities = np.divide(probabilities,np.sum(probabilities))
            leaf_selection = np.random.choice(a=len(self.leaf_list),size=n,p=probabilities)
            selected_leaf, n_per_leaf = np.unique(leaf_selection, return_counts=True)

            for i_leaf, leaf_index in enumerate(selected_leaf):
                leaf_result = self.get_leaf_by_index(leaf_index).rvs(n=n_per_leaf[i_leaf])
                result = np.vstack((result,leaf_result))
            result = result[1:,:]
        return result

    def get_leaf_by_index(self,index:int):
        """Retrieve a node from the tree by its index in the leaf list
        :param index: Index of target leaf
        :type index: int
:return: Leaf
:rtype: recursive_gaussian_mixture
"""
        if index < len(self.leaf_list): return self.leaf_list[index]
        i_leaf = 0
        while i_leaf <= index:
            if len(self.children)<1: return self
            for child in self.children:
                if child.is_leaf: leaf = child
                else: leaf = child.get_leaf_by_index(i_leaf) 
                if i_leaf==index: return leaf
                i_leaf += 1

    def build_leaf_list(self, recursive = False):
        """Construct a list of leaf nodes of this tree
        :param recursive: Recurse down the tree? Defaults to False
        :type recursive: bool, optional
"""
        self.n_leaf_nodes = 0
        if self.is_leaf: 
            self.leaf_list = [self]
            self.n_leaf_nodes = 1
        else:
            self.leaf_list = []
            for child in self.children: 
                if recursive: child.build_leaf_list(recursive=True)
                self.n_leaf_nodes += child.n_leaf_nodes
                [self.leaf_list.append(leaf) for leaf in child.leaf_list]



    def split_self(self):
        """Splits this node into a number of children defined by the library size"""
        rootcov = self.run_empirical_cov()
        # This node is no longer a leaf node
        self.is_leaf = False
        self.n_leaf_nodes -= 1

        self.n_leaf_nodes += self.library_size
        self.children = []
        
        weights, means, cov = self.get_mixture_parameters(rootcov)
        for i_child in range(self.library_size):
            self.children.append(recursive_gaussan_mixture(mean         =  means[i_child],
                                                           cov          =  cov_nearest(cov),
                                                           weight       =  weights[i_child],
                                                           is_leaf      =  True,
                                                           library_size =  self.library_size,
                                                           tree_size    =  1,
                                                           rng          =  self.rng,
                                                           sigma_parameters = self.sig_params))
    
    def split_leaf(self):
        """Selects a leaf of this tree according to the principle of maximal uncertainty and splits it"""
        max_uncertainty = 0.0
        split_candidate = None
        i_candidate = None
        for i_leaf, leaf in enumerate(self.leaf_list):
            leaf_cov = leaf.empirical_cov
            uncertainty, direction = np.linalg.eigh(leaf_cov)
            if uncertainty[-1] > max_uncertainty:
                max_uncertainty = uncertainty[-1]
                eigvec = direction[-1]
                split_candidate = leaf
                i_candidate = i_leaf
        print('Splitting Leaf {} (with max variance {} at {})'.format(i_candidate,max_uncertainty,eigvec))
        split_candidate.split_self()
        self.build_leaf_list(recursive=True)
    
    def recalculate_mean(self) -> np.ndarray:
        """Compute the mean of the GMM recursively down the tree
:return: Mean of this node
:rtype: np.ndarray
"""
        if self.is_leaf: return self.weight*self.mean
        else:
            mean = np.zeros_like(self.mean)
            for child in self.children: mean += child.recalculate_mean()
            return mean
    
    def run_empirical_cov(self,fidelity=10000) -> np.ndarray:
        """Computes the empirical covariance of the GMM with a specified fidelity
        :param fidelity: Number of samples to use for covariance calculation, defaults to 10000
        :type fidelity: int, optional
:return: Empirical covariance
:rtype: np.ndarray
"""
        return np.cov(self.rvs(fidelity),rowvar=False)
    
    def generate_points(self) -> np.ndarray:
        """Computes the set of sigma points to propagate all Gaussians of this GMM
:return: Set of sigma points, of size equal to n_Gaussians * ( 2 * Dimensions + 1 )
:rtype: np.ndarray
"""
        if self.is_leaf:
            return self.point_generator.sigma_points(self.mean, self.cov)
        else:
            sigmas = np.zeros([1, self.dim])
            for child in self.children:
                sigmas = np.vstack([sigmas, child.generate_points()])
            return sigmas[1:, :]
    
    def transform(self, points : np.ndarray, overwrite=True):
        """Computes an unscented transform for each Gaussian using the passed points array
        :param points: Array of points post-propagation
        :type points: np.ndarray
        :param overwrite: Whether to adjust the gaussians using the transform, defaults to True
        :type overwrite: bool, optional
:return: Means, Covariances
:rtype: np.ndarray, np.ndarray
"""
        if self.is_leaf:
            mu, cov = unscented_transform(points, self.point_generator.Wm, self.point_generator.Wc)
            if overwrite:
                self.mean = mu
                self.cov = cov_nearest(cov)
                self.update_distribution()
                return [self.mean], [self.cov]
            else:
                self.mean_out = mu
                self.cov_out = cov_nearest(cov)
                return [self.mean_out], [self.cov_out]
        else:
            pointer = 0
            mus = []
            covs = []
            for child in self.children:
                child_mus, child_covs = child.transform(points[pointer:pointer+child.n_leaf_nodes*(2*child.dim+1),:],
                                overwrite=overwrite)
                pointer+=child.n_leaf_nodes*(2*child.dim+1)
                [mus.append(mu) for mu in child_mus]
                [covs.append(cov) for cov in child_covs]

            if overwrite:
                self.recalculate_mean()
                self.run_empirical_cov()
            return mus, covs

    # def verify_passing(self, nums : list) -> list:
    #     """Verify that numbers can be passed down and up the GMM

    #     :param nums: List of numbers, must have len = num_leaf_nodes
    #     :type nums: list
    #     :return: Same list after being passed all the way to the leaves of the GMM and back up
    #     :rtype: list
    #     """
    #     if self.is_leaf: 
    #         return nums
    #     else:
    #         pointer = 0
    #         out_nums = []
    #         for child in self.children:
    #             child_nums = child.verify_passing(nums[pointer:pointer+child.n_leaf_nodes])
    #             pointer+=child.n_leaf_nodes
    #             [out_nums.append(num) for num in child_nums]
    #     assert out_nums == nums
    #     return nums

if __name__=='__main__':
    gmm = recursive_gaussian_mixture(np.zeros(13),np.diag(np.ones(13)+np.random.random(13)),is_leaf=True)

    print(gmm.generate_points().shape)

    for _ in range(10): gmm.split_leaf()
    gmm.verify_passing(list(range(gmm.n_leaf_nodes)))
    print(gmm.generate_points().shape)
    gmm.transform(gmm.generate_points())
