from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.spatial import Voronoi
import pathlib, glob, traceback
import trimesh

def plot_result(data, directory='.'):
    for parameter, title in zip(['volume','surf_area','mass','area_mass_ratio','reference_length'],['Volume', 'Surface Area', 'Mass', 'Area to Mass Ratio', 'Reference Length']):
        series = data[parameter].to_numpy()
        cum = []
        bins = np.logspace(np.log10(np.min(series)),np.log10(np.max(series)),250, endpoint=False)
        for q_dist in bins: cum.append(len(np.where(series>q_dist)[0]))
        law = linregress(np.log10(bins),np.log10(cum))

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(bins, cum, label='Fragments', linestyle=':',marker='x',color='black')
        ax.plot(bins, 10**law.intercept*bins**law.slope, label='{}N^[{}]'.format(np.round(10**law.intercept,4),np.round(law.slope,4)), color='blue')
        if parameter=='reference_length':
            ax.plot(bins, 6*bins**-1.6, label='6N^[-1.6] (NASA-SBM)'.format(np.round(10**law.intercept,4),np.round(law.slope,4)), color='red')
        ax.legend()
        ax.grid(which='major')
        ax.set_xlabel(title)
        ax.set_ylabel('Number of Fragments')
        fig.suptitle(title+' Statistics (R²={})'.format(law.rvalue**2))
        fig.savefig(directory+'/'+title+'_stats.png', dpi=600)

if __name__=='__main__':
    if pathlib.Path('./stats.csv').resolve().exists():
        data = pd.read_csv('./stats.csv')
        plot_result(data)
        exit()
    elif len(glob.glob('./*.stl'))==1:
        import sys
        sys.path.append(str(pathlib.Path.home())+'/TITAN')
        from Explosion.manifold_voronoi_fracture import generate_fragment_meshes, mesh_check
        from Explosion.generate_seed_distribution import SpiralSampler
        stlfile = glob.glob('./*.stl')[0]
        mesh = trimesh.load(stlfile)
        cog_bias = 0.1
        crack_width = 1e-3
        n_frags = 64
        pathlib.Path('./voronoi_seeded/').resolve().mkdir(exist_ok=True)
        expl_dir = './voronoi_seeded'
        spiral_weights = np.array([0.75 * 0.6064932580341802, 
                                         0.7066237981471332, 
                                        35.36336186280215, 
                                        34.24156856002203])
        max_attempts = 30
        for i_attempt in range(max_attempts):
            
            rng = np.random.RandomState(6+i_attempt)

            f_id = rng.choice(np.arange(mesh.faces.shape[0]))

            facet = np.mean(mesh.vertices[mesh.faces],axis=1)[f_id,:]
            nucleus = cog_bias*mesh.center_mass+ (1-cog_bias)*facet
            points  = SpiralSampler(spiral_weights[0],spiral_weights[1],spiral_weights[2],spiral_weights[3], rng=rng).rvs(n_frags)
            points += np.full_like(points,nucleus)
            vor = Voronoi(points)
            try:
                generate_fragment_meshes(stlfile,vor, expl_dir, extrude=crack_width, rng=rng, n=n_frags, nucleus=nucleus)
                pathlib.Path('./prisms0.stl').resolve().rename('voronoi_seeded{}.stl'.format(i_attempt))
                if mesh_check(expl_dir,1000, mesh.volume, 
                            threshold=15, delete_bad=True, 
                            quiet=False): break
            except Exception as e: 
                [print(tr) for tr in traceback.format_exception(e)]
            if i_attempt<max_attempts-1:
                print('Voronoi {} failed mesh check! Recalculating...'.format(i_attempt))
                if pathlib.Path(expl_dir+'/stats.csv').resolve().exists():
                    pathlib.Path(expl_dir+'/stats.csv').resolve().unlink()
                for frag in glob.glob("{}/*.stl".format(expl_dir)): pathlib.Path(frag).unlink()
            else: raise Exception('Could not build voronoi fragments after {} attempts'.format(i_attempt+1))
        plot_result(pd.read_csv(expl_dir+'/stats.csv'),expl_dir)
        
        pathlib.Path('./voronoi_uniform/').resolve().mkdir(exist_ok=True)
        expl_dir = './voronoi_uniform'
        max_attempts = 30
        n_frags = 48
        for i_attempt in range(max_attempts):
            rng = np.random.RandomState(6+i_attempt)
       
            f_id = rng.choice(np.arange(mesh.faces.shape[0]))

            facet = np.mean(mesh.vertices[mesh.faces],axis=1)[f_id,:]
            nucleus = cog_bias*mesh.center_mass+ (1-cog_bias)*facet
            points = rng.random([n_frags, 3])
            points = np.full_like(points, mesh.bounds[0])+(np.full_like(points, mesh.bounds[1])-np.full_like(points, mesh.bounds[0]))*points
            try:
                generate_fragment_meshes(stlfile,Voronoi(points), expl_dir, extrude=crack_width, rng=rng, n=n_frags, nucleus=nucleus)
                pathlib.Path('./prisms0.stl').resolve().rename('voronoi_uniform{}.stl'.format(i_attempt))
                if mesh_check(expl_dir,1000, mesh.volume, 
                            threshold=15, delete_bad=True, 
                            quiet=False): break
            except Exception as e: 
                [print(tr) for tr in traceback.format_exception(e)]
            if i_attempt<max_attempts-1:
                print('Voronoi {} failed mesh check! Recalculating...'.format(i_attempt))
                if pathlib.Path(expl_dir+'/stats.csv').resolve().exists():
                    pathlib.Path(expl_dir+'/stats.csv').resolve().unlink()
                for frag in glob.glob("{}/*.stl".format(expl_dir)): pathlib.Path(frag).unlink()
            else: raise Exception('Could not build voronoi fragments after {} attempts'.format(i_attempt+1))
            
        plot_result(pd.read_csv(expl_dir+'/stats.csv'),expl_dir)
        
        pathlib.Path('./orthogonal_nucleus/').resolve().mkdir(exist_ok=True)
        expl_dir = './orthogonal_nucleus'
        max_attempts = 30
        n_frags = 14
        for i_attempt in range(max_attempts):
            rng = np.random.RandomState(6+i_attempt)
       
            f_id = rng.choice(np.arange(mesh.faces.shape[0]))

            facet = np.mean(mesh.vertices[mesh.faces],axis=1)[f_id,:]
            nucleus = cog_bias*mesh.center_mass+ (1-cog_bias)*facet

            try:
                generate_fragment_meshes(stlfile,None, expl_dir, extrude=crack_width, rng=rng, n=n_frags, nucleus=nucleus)
                pathlib.Path('./prisms0.stl').resolve().rename('nucleus{}.stl'.format(i_attempt))
                if mesh_check(expl_dir,1000, mesh.volume, 
                            threshold=15, delete_bad=True, 
                            quiet=False): break
            except Exception as e: 
                [print(tr) for tr in traceback.format_exception(e)]
            if i_attempt<max_attempts-1:
                print('Voronoi {} failed mesh check! Recalculating...'.format(i_attempt))
                if pathlib.Path(expl_dir+'/stats.csv').resolve().exists():
                    pathlib.Path(expl_dir+'/stats.csv').resolve().unlink()
                for frag in glob.glob("{}/*.stl".format(expl_dir)): pathlib.Path(frag).unlink()
            else: raise Exception('Could not build voronoi fragments after {} attempts'.format(i_attempt+1))
        plot_result(pd.read_csv(expl_dir+'/stats.csv'),expl_dir)
        pathlib.Path('./orthogonal_no_nucleus/').resolve().mkdir(exist_ok=True)
        expl_dir = './orthogonal_no_nucleus'
        n_frags = 12
        max_attempts = 30
        for i_attempt in range(max_attempts):
            rng = np.random.RandomState(6+i_attempt)
       
            f_id = rng.choice(np.arange(mesh.faces.shape[0]))

            facet = np.mean(mesh.vertices[mesh.faces],axis=1)[f_id,:]
            nucleus = cog_bias*mesh.center_mass+ (1-cog_bias)*facet

            try:
                generate_fragment_meshes(stlfile,None, expl_dir, extrude=crack_width, rng=rng, n=n_frags)
                pathlib.Path('./prisms0.stl').resolve().rename('no_nucleus{}.stl'.format(i_attempt))
                if mesh_check(expl_dir,1000, mesh.volume, 
                            threshold=15, delete_bad=True, 
                            quiet=False): break
            except Exception as e: 
                [print(tr) for tr in traceback.format_exception(e)]
            if i_attempt<max_attempts-1:
                print('Voronoi {} failed mesh check! Recalculating...'.format(i_attempt))
                if pathlib.Path(expl_dir+'/stats.csv').resolve().exists():
                    pathlib.Path(expl_dir+'/stats.csv').resolve().unlink()
                for frag in glob.glob("{}/*.stl".format(expl_dir)): pathlib.Path(frag).unlink()
            else: raise Exception('Could not build voronoi fragments after {} attempts'.format(i_attempt+1))
        plot_result(pd.read_csv(expl_dir+'/stats.csv'),expl_dir)
        
