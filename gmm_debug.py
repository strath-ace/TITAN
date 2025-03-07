from Uncertainty.UT import recursive_gaussan_mixture
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
cov = np.diag([0.3,0.2,0.5,0.7])
mean = [1.0,10.0,50.0,10.0]
n_samples = 10000

def banana(v): 
    return np.exp(v)

def f_nonlinear(v):
    return np.cross([v[0],v[1]],[v[1],v[2]])

def spherical_transformation(v):
    r = np.linalg.norm(v)
    theta = np.arccos(v[2]/r)
    phi = np.sign(v[1])*np.arccos(v[0]/np.sqrt(v[0]**2+v[1]**2))
    return [r,theta,phi]

def trigonometric_transformation(v):
    v0 = np.sin(v[0])
    v1 = np.cos(v[1]+v[2])
    v2 = np.sin(v[2])
    v3 = np.cos(v[0]-v[3])
    return [v0,v1,v2,v3]

def projectile_motion(s):
    dt = 0.1
    a = 9.81
    cd = 0.1
    for iter in range(50):
        x = s[0]
        y = s[1]
        u = s[2]
        v = s[3]

        s[0] = x + u*dt
        s[1] = y + v*dt
        s[2] = u - np.sign(u)*cd*u**2*dt
        s[3] = v - np.sign(v)*cd*v**2*dt - a*dt
    return s

#true_distri = stats.normal_inverse_gamma(mean,cov)
rvs_true = stats.multivariate_normal(mean,cov).rvs(n_samples)
nl_func = trigonometric_transformation
true_sphere = np.array([nl_func(point) for point in rvs_true])

base_node = recursive_gaussan_mixture(mean, cov, 
                                      weight = 1.0, 
                                      is_leaf = True, 
                                      library_size = 5, 
                                      tree_size = 1, 
                                      rng = np.random.RandomState(seed=42069))



points = MerweScaledSigmaPoints(n=len(mean),alpha=1e-3,beta=2,kappa=0.0)
sigmas = points.sigma_points(mean,cov)
new_sigmas = np.array([nl_func(point) for point in sigmas])
new_mu, new_cov = unscented_transform(new_sigmas,points.Wm,points.Wc)

base_node.mean = new_mu
base_node.cov = new_cov
base_node.update_distribution()
base_sphere = base_node.rvs(n_samples)

mu_true = np.mean(true_sphere,axis=0)
mm_gmm = np.mean(base_sphere,axis=0)
print('Mean is {} (True value is {})'.format(mm_gmm,mu_true))
#assert np.all(np.isclose(np.mean(rvs_true),np.mean(rvs_gmm),rtol=0.1))

base_node = recursive_gaussan_mixture(mean, cov, 
                                      weight = 1.0, 
                                      is_leaf = True, 
                                      library_size = 5, 
                                      tree_size = 1, 
                                      rng = np.random.RandomState(seed=42069))

n_splits = 2000
rvs_split1 = base_node.rvs(n_samples)
for i in range(n_splits): 
    print(i)
    base_node.split_leaf()
for leaf in base_node.leaf_list:
    sigmas = points.sigma_points(leaf.mean,leaf.cov)
    new_sigmas = np.array([nl_func(point) for point in sigmas])
    new_mu, new_cov = unscented_transform(new_sigmas,points.Wm,points.Wc)
    leaf.mean = new_mu
    leaf.cov = new_cov
    leaf.update_distribution()
split_sphere = base_node.rvs(n_samples)
print('Split mean is {}, (true is {})'.format(np.mean(split_sphere,axis=0),mu_true))
fig = plt.figure()
fig2 = plt.figure()
axes = [fig.add_subplot(1,3,i+1) for i in range(3)]
xy_ax = [fig2.add_subplot(1,3,i+1) for i in range(3)]
labels = ['xy','yz','xz']
# for i_leaf in range(base_node.n_leaf_nodes):
#     dat = base_node.get_leaf_by_index(i_leaf).rvs(n_samples)
#     [sns.histplot(x=dat[:,i],ax=axes[i]) for i in range(3)]
#     [sns.histplot(x=dat[:,i],y=dat[:,i+1], ax=xy_ax[i],label=labels[i]) for i in range(2)]
#     sns.histplot(x=dat[:,0],y=dat[:,2], ax=xy_ax[2],label=labels[2])
id_list  = np.ones([n_samples,1])
id_list2 = 2*id_list
id_list3 = 3*id_list
data = pd.DataFrame(np.vstack((np.hstack((true_sphere, id_list)),np.hstack((base_sphere, id_list2)),np.hstack((split_sphere, id_list3)))),columns=['X','Y','U','V','distri'])
#data = pd.DataFrame(np.vstack((np.hstack((true_sphere, id_list)),np.hstack((base_sphere, id_list2)))),columns=['X','Y','Z','distri'])
sns.pairplot(data,hue='distri',kind='kde',plot_kws=dict(levels=[0.1,0.9]))
#assert np.all(np.isclose(np.mean(rvs_split1,axis=1),np.mean(rvs_gmm),rtol=0.2))
plt.show()