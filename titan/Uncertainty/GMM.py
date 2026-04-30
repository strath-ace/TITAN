import numpy as np
from copy import copy
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform

class recursive_gaussan_mixture():
    def __init__(self, mean, cov, weight = 1.0, is_leaf = True, library_size = 3, tree_size = 1, rng = np.random.RandomState(seed=42069), sigma_parameters = [1e-3,2,0]):

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

        # Standard splitting libraries with 3 and 5 from DeMars et al, doi.org/10.2514/1.58987
        # 4 element library from Huber et al, doi.org/10.1109/MFI.2008.4648062
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

    def get_mixture_parameters(self,rootcov):
        # Split component along principal axis of covariance hyper-ellipsoid
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
    
    def get_shannon_entropy_change(self):
        self.shannon_entropy.append(0.5 * self.dim * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.cov)) + self.dim/2)
        dH = self.shannon_entropy[-1] - self.shannon_entropy[-2] if len(self.shannon_entropy)>1 else None
        return dH
    
    def update_distribution(self):
        self.distribution = multivariate_normal(self.mean,self.cov, allow_singular = True)
        self.empirical_cov = self.run_empirical_cov()

    def rvs(self,n=1):
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

    def get_leaf_by_index(self,index):
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
    
    def recalculate_mean(self):
        if self.is_leaf: return self.weight*self.mean
        else:
            mean = np.zeros_like(self.mean)
            for child in self.children: mean += child.recalculate_mean()
            return mean
    
    def run_empirical_cov(self,fidelity=10000):
        return np.cov(self.rvs(fidelity),rowvar=False)
    
    def generate_points(self):
        if self.is_leaf:
            return self.point_generator.sigma_points(self.mean, self.cov)
        else:
            sigmas = np.zeros([1, self.dim])
            for child in self.children:
                sigmas = np.vstack([sigmas, child.generate_points()])
            return sigmas[1:, :]
    
    def transform(self, points, overwrite=True):
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

    def verify_passing(self, nums):
            if self.is_leaf: 
                return nums
            else:
                pointer = 0
                out_nums = []
                for child in self.children:
                    child_nums = child.verify_passing(nums[pointer:pointer+child.n_leaf_nodes])
                    pointer+=child.n_leaf_nodes
                    [out_nums.append(num) for num in child_nums]
            assert out_nums == nums
            return nums

if __name__=='__main__':
    gmm = recursive_gaussan_mixture(np.zeros(13),np.diag(np.ones(13)+np.random.random(13)),is_leaf=True)

    print(gmm.generate_points().shape)

    for _ in range(10): gmm.split_leaf()
    gmm.verify_passing(list(range(gmm.n_leaf_nodes)))
    print(gmm.generate_points().shape)
    gmm.transform(gmm.generate_points())