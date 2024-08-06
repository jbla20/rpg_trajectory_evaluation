#!/usr/bin/env python3

# Standard library imports
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import argparse
from tqdm import tqdm

# Type imports
from typing import List, Dict
from matplotlib.lines import Line2D

# Local imports
import add_path
from trajectory import Trajectory
from helpers import suppress_stdout, latexify, boldify

plt.rc('font', family='sans')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]

# Constants
ALG_COLOR = {'vins': 'b', 'svo': 'g', 'orb_slam3': 'r'}
TEST_MAP = {'1,1' : '1rect', 
            '1,2' : '1circ', 
            '1,3' : '1rand', 
            '2,1' : '2feat', 
            '2,2' : '2tilt'}
CONDITION_MAP = {'t' : 
                    {'0': '0ml', '1': '50ml', '2': '100ml'}, 
                'ms' : 
                    {'0': '0.0g', '1': '1.5g', '2': '3.0g', '3': '4.5g'}}

def color_box(bp : Dict[str, List[Line2D]], color : str):
    elements = ['medians', 'boxes', 'caps', 'whiskers']
    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color, linestyle='-', lw=1.0)
         for idx in range(len(bp[elem]))]
    return

def num_to_name(traj_num_identifier : str) -> str:    
    # Determine trajectory type
    if not traj_num_identifier[:3] in ['1,1', '1,2', '1,3', '2,1', '2,2']:
        raise ValueError("Invalid trajectory type: " + traj_num_identifier[:3])
    traj_type = TEST_MAP[traj_num_identifier[:3]]
    
    # Determine turbidity
    if not traj_num_identifier[4] in ['0', '1', '2']:
        raise ValueError("Invalid turbidity level: " + traj_num_identifier[4])
    turbidity = f"t={CONDITION_MAP['t'][traj_num_identifier[4]]}"
    
    # Determine marine snow
    if not traj_num_identifier[6] in ['0', '1', '2', '3']:
        raise ValueError("Invalid marine snow level: " + traj_num_identifier[6])
    marine_snow = f"ms={CONDITION_MAP['ms'][traj_num_identifier[6]]}"
    
    return f"{traj_type}_{turbidity}_{marine_snow}"

def get_individual_combination(traj_num_identifiers : List[str], condition_type : str, level : str) -> List[int]:
    if condition_type == 't': match_char_idx = 4
    elif condition_type == 'ms': match_char_idx = 6
    else:
        raise ValueError("Invalid condition type: " + condition_type)
    
    # Get indices of the identifiers that match the condition
    indices = [i for i, identifier in enumerate(traj_num_identifiers) if identifier[match_char_idx] == level]    
    return indices

def get_algs_from_dirs(dirs : List[str], alg_type : str) -> List[str]:
    if alg_type != 'mixed':
        return [alg_type]
    
    algs = []
    for dir in dirs:
        if dir.count('vins') > 0: algs.append('vins')
        elif dir.count('svo') > 0: algs.append('svo')
        elif dir.count('orb_slam3') > 0: algs.append('orb_slam3')
        else:
            raise ValueError("Invalid algorithm folder for alg_type='mixed': " + dir)
    
    return algs
    

def boxplot_comparison_individual(eval_dir : str, alg_type : str, plot_type : str = 'rel_trans_perc', save : bool = False):    
    """ Boxplot comparison of the relative error for different test IDs.
    :param eval_dir: Folder containing the evaluation results for the different tests.
    :param alg_type: Name of the algorithm to compare (valid options: 'vins', 'svo', 'orb_slam3', 'mixed')
    :param plot_type: Type of error to plot (valid options: 'rel_trans', 'rel_trans_perc', 'rel_yaw')
    :param save: If True, the plots are saved in the eval_dir folder. If False, the plots are shown.
    """
    # Load and initialise data
    test_type = TEST_MAP[Path(eval_dir).name[:3]] if Path(eval_dir).name[2] != 'x' else Path(eval_dir).name[0] + 'comb'
    alg_dirs = [eval_dir] if alg_type != 'mixed' else [dir for dir in sorted(glob.glob(eval_dir + '/*'), reverse=True) if Path(dir).is_dir()]
    alg_types = get_algs_from_dirs(alg_dirs, alg_type)
    boxplot_perc = [0.1]#, 0.2, 0.3, 0.4, 0.5]
    n_algs = len(alg_types)
    n_conds = 12
    n_perc = len(boxplot_perc)
    traj_num_identifiers = []
    
    data = [[[np.empty(0) for _ in range(n_perc)] for _ in range(n_conds)] for _ in range(n_algs)]
    for i, alg_dir in enumerate(alg_dirs):        
        test_dirs = [alg_dir] if test_type.count('comb') == 0 else [dir for dir in sorted(glob.glob(alg_dir + '/*'), reverse=True) if Path(dir).is_dir()]
        
        remove_idx = set()
        for test_dir in test_dirs:
            for j, sub_dir in enumerate(sorted(glob.glob(test_dir + '/*'))):
                if not Path(sub_dir).is_dir():
                    continue
                
                # Get trajectory number identifier
                traj_num_identifiers.append(Path(sub_dir).name[:7])
                
                # Skip if there is no data in the sub folder
                if Path(sub_dir).joinpath("saved_results/traj_est/cached/cached_rel_err.pickle").is_file():
                    # Load trajectory data (suppress stdout to avoid printing trajectory data to console)
                    with suppress_stdout():
                        traj = Trajectory(results_dir=sub_dir, preset_boxplot_percentages=boxplot_perc)
                    
                    # Get relative errors and save them in the data list
                    rel_errors, distances = traj.get_relative_errors_and_distances(error_types=[plot_type])
                    data[i][j] = [np.concatenate((a, b)) for a, b in zip(data[i][j], rel_errors[plot_type][0])]
                else:
                    # Add index to remove list since no data was found
                    remove_idx.add(j)
                    
        # Remove data where no results were found for at least one of the movement patterns
        for idx in sorted(remove_idx, reverse=True):
            data[i][idx] = [np.empty(0) for _ in range(n_perc)]
        
    print("Loaded data: ", traj_num_identifiers)
    
    
    conditions = [('t', '0'), ('t', '1'), ('t', '2'), ('ms', '0'), ('ms', '1'), ('ms', '2'), ('ms', '3')]
    for condition in conditions:
        # Get the combination of indices for the different levels of the condition
        indices = get_individual_combination(traj_num_identifiers[:n_conds], condition[0], condition[1])
        perc_alg_cond_data = [[alg_data[i] for i in indices] for alg_data in data]
        xlabels = [CONDITION_MAP['ms' if condition[0]=='t' else 't'][str(i)] for i in range(len(indices))]
        
        # Shift the dimension twice to the left to get the correct order of the data (boxplot_perc, alg_type, test_id)
        for i in range(2):
            perc_alg_cond_data = [[list(row) for row in zip(*col)] for col in zip(*perc_alg_cond_data)]
        
        n_xlabel = len(indices)
        w = 1/3 / n_algs
        widths = [w for _ in np.arange(n_xlabel)]
        step_dist = 0.1 + w
        outer_offset = step_dist * (n_algs - 1) / 2
        for boxplot_idx, alg_cond_data in tqdm(enumerate(perc_alg_cond_data), desc='Creating boxplots', leave=True, total=n_perc):
            # Create figure and axis
            plt.rc('axes', titlesize=24, titlepad=15, labelsize=14)
            fig = plt.figure(figsize=(10, 5))
            title = f'{alg_type}_{test_type}_{condition[0]}={CONDITION_MAP[condition[0]][condition[1]]}' \
                    f' [{str(boxplot_perc[boxplot_idx]*100)}%]'
            ax = fig.add_subplot(111,
                                xlabel=boldify('Marine Snow' if condition[0] == 't' else 'Turbidity'), 
                                ylabel=boldify('Translation error [%]'), 
                                title=latexify(title))
            
            leg_handles = []
            for alg_idx, cond_data in enumerate(alg_cond_data):
                # Convert from list to numpy array
                temp = np.empty((n_xlabel,), dtype=object)
                for i in range(n_xlabel):
                    temp[i] = cond_data[i]
                cond_data = temp

                # Calculate positions of the boxplots
                positions = [-outer_offset + alg_idx*step_dist + pos for pos in np.arange(n_xlabel)]
            
                # Create boxplot
                bp = ax.boxplot(cond_data, 0, '', positions=positions, widths=widths)
                color_box(bp, ALG_COLOR[alg_types[alg_idx]])
                
                leg_handle, = plt.plot([], [], ALG_COLOR[alg_types[alg_idx]])
                leg_handles.append(leg_handle)
            
            # Create legend
            ax.legend(leg_handles, alg_types, loc='upper left')
            map(lambda x: x.set_visible(False), leg_handles)
            
            # Set xticks and xticklabels
            ax.set_xticks(np.arange(n_xlabel))
            ax.set_xticklabels(latexify(xlabels))
            xlims = ax.get_xlim()
            ax.set_xlim([xlims[0]-0.1, xlims[1]-0.1])
            ax.set_ylim([0, 100])
            
            # Save or show plot
            fig.tight_layout()
            if save:
                fig.savefig(eval_dir + f'/{title}.pdf',
                            bbox_inches="tight")
            else:
                plt.show()
            plt.close(fig)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Compare boxplot of relative errors for each individual condition.''')
    parser.add_argument(
        'eval_dir', type=str,
        help="Folder containing the test runs with existing results.")
    parser.add_argument(
        '--alg_type', required=False, type=str, choices=['vins', 'svo', 'orb_slam3', 'mixed'],
        help="Name of the algorithm(s) to compare. If using 'mixed', ensure that the names of the algorithms are in the respective folder names.",
        default='svo')
    parser.add_argument(
        '--plot_type', required=False, type=str, choices=['rel_trans', 'rel_trans_perc', 'rel_yaw'],
        help="Type of error to plot",
        default='rel_trans_perc')
    parser.add_argument('--save', dest='save',
                        action='store_true')
    parser.set_defaults(save=True)
    args = parser.parse_args()
    
    
    # Call boxplot comparison function
    boxplot_comparison_individual(eval_dir=args.eval_dir, 
                                  alg_type=args.alg_type, 
                                  plot_type=args.plot_type, 
                                  save=args.save)