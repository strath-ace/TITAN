import mutationpp as mpp
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import pathlib
from itertools import combinations
import multiprocessing


def make_mixfile_if_needed(mix_elems:list, max_reductions=4) ->str:
    mix_elems.sort()
    name = '_'.join(mix_elems)
    mpp_data_dir = mpp.GlobalOptions.dataDirectory()
    if pathlib.Path(mpp_data_dir+'/mixtures/'+name+'.xml').exists():
        print('Mix already exists!');exit()
    print('Making mixture {}: {}'.format(name,mix_elems))
    if not 'N' in mix_elems: mix_elems.append('N')
    if not 'O' in mix_elems: mix_elems.append('O')

    
    with open(mpp_data_dir+'/mixtures/mix_'+name+'.xml', 'w') as f:
        f.write('<mixture thermo_db="NASA-9">\n')
        f.write('    <species>\n')
        f.write('        {all with'+','.join(mix_elems)+'}\n')
        f.write('    </species>\n')
        f.write('</mixture>')
    mix = mpp.Mixture(mpp.MixtureOptions('mix_'+name))
    all_species = []
    for i_species in range(mix.nSpecies()):
        speciesName = mix.speciesName(i_species)
        if not '\"' in speciesName: all_species.append(speciesName) 
    species_combos = []
    print(len(all_species)+1)
    for i in range(max_reductions):
        print('Computing reduction {}'.format(i))
        [species_combos.append(list(combo)) for combo in combinations(all_species,len(all_species)-i)]
    max_species = 0
    q = multiprocessing.Queue()

    for i_mix, combo in enumerate(species_combos):
        p = multiprocessing.Process(target=test_mix_wrapper,args=(combo,i_mix,q))
        p.start()
        p.join(0.5)
        if not p.is_alive():
            new_combo = q.get()
            if len(new_combo)>max_species:
                max_species = len(new_combo)
                target_combo = new_combo
                print('Writing new mix...')
                with open(mpp_data_dir+'/mixtures/'+name+'.xml', 'w') as f:
                    f.write('<mixture thermo_db="NASA-9">\n')
                    f.write('    <species>\n')
                    f.write('        '+' '.join(target_combo)+'\n')
                    f.write('    </species>\n')
                    f.write('</mixture>')
                print('Cleaning up...')
                # for candidate in pathlib.Path(mpp_data_dir+'/mixtures/').glob('candidate_*.xml'): 
                #     print(candidate)
                #     candidate.unlink()
                print('Exiting...')
                exit()
        else:
            pathlib.Path(mpp_data_dir+'/mixtures/candidate_{}.xml'.format(i_mix)).unlink()
            p.terminate()
            p.kill()

def test_mix_wrapper(combo,i_mix, q):
    mpp_data_dir = mpp.GlobalOptions.dataDirectory()
    name = 'candidate_'+str(i_mix)
    
    with open(mpp_data_dir+'/mixtures/'+name+'.xml', 'w') as f:
        f.write('<mixture thermo_db="NASA-9">\n')
        f.write('    <species>\n')
        f.write('        '+' '.join(combo)+'\n')
        f.write('    </species>\n')
        f.write('</mixture>')
    mixO = mpp.MixtureOptions(name)
    print('Tried mix {}'.format(i_mix))
    mixO.setStateModel('Equil')
    mix = mpp.Mixture(mixO)
    pathlib.Path(mpp_data_dir+'/mixtures/'+name+'.xml').unlink()
    print('Succeeded with mix \n{}'.format(combo))
    q.put(combo)

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("-m", "--mix",
                        dest="mixname",
                        type=str,
                        help="input mixture name",
                        metavar="mix")
    
    parser.add_argument("-r", "--red",
                        dest="reductions",
                        type=int,
                        help="num reductions",
                        metavar="red")

    args=parser.parse_args()
    mixname = args.mixname.split(',')
    if args.reductions is not None: r=args.reductions
    else: r=3
    make_mixfile_if_needed(mixname,r)
    exit()
    print('Looking for mixture {}'.format(mixname))
    try:
        mixO = mpp.MixtureOptions(mixname)
        mix = mpp.Mixture(mixO)
        print('nSpecies : {}'.format(mix.nSpecies()))
        for i_species in range(mix.nSpecies()):
            print( '#{} : {}'.format(i_species,mix.speciesName(i_species)))
        print('Checking equil mixture')
        mixO = mpp.MixtureOptions(mixname)
        mixO.setStateModel('Equil')
        mix = mpp.Mixture(mixO)
        print('Success! Exiting')
        exit()
    except Exception as e:
        print('Error getting mix!')
        raise Exception(e)
