#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import argparse
from tqdm import tqdm

from contextlib import contextmanager
import add_path
import sys, os
from trajectory import Trajectory

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def color_box(bp, color):
    elements = ['medians', 'boxes', 'caps', 'whiskers']
    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color, linestyle='-', lw=1.0)
         for idx in range(len(bp[elem]))]
    return

def boxplot_comparison(eval_dir, plot_type = 'rel_trans_perc', save = False):    
    """ Boxplot comparison of the relative error for different test IDs.
    :param eval_dir: Folder containing the evaluation results for the different tests.
    :param plot_type: Type of error to plot (valid options: 'rel_trans', 'rel_trans_perc', 'rel_yaw')
    :param save: If True, the plots are saved in the eval_dir folder. If False, the plots are shown.
    """
    default_boxplot_perc = [0.1, 0.2, 0.3, 0.4, 0.5]
    data = []
    xlabels = []
    for subfolder in sorted(glob.glob(eval_dir + '/*')):
        if not Path(subfolder).is_dir():
            continue
        
        # Load trajectory data (suppress stdout to avoid printing trajectory data to console)
        with suppress_stdout():
            traj = Trajectory(results_dir=subfolder, preset_boxplot_percentages=default_boxplot_perc)
        
        # Get relative errors and save them in the data list along with the xlabels
        rel_errors, distances = traj.get_relative_errors_and_distances(error_types=[plot_type])
        data.append(rel_errors[plot_type][0])
        xlabels.append(Path(subfolder).name[:7])
    print("Loaded data: ", xlabels)
    
    # Invert list of lists to have the boxplot percent as the outer list and the test ID as the inner list
    data = [list(i) for i in zip(*data)]
    
    n_xlabel = len(xlabels)
    w = 1/3
    widths = [w for pos in np.arange(n_xlabel)]
    positions = [pos - 0.5 + 1.5 * w for pos in np.arange(n_xlabel)]
    for boxplot_idx, d in tqdm(enumerate(data), desc='Creating boxplots', leave=True, total=len(default_boxplot_perc)):
        # Create figure and axis
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111,
                            xlabel='Test ID', 
                            ylabel='Translation error [%]', 
                            title=f'Relative translation error comparison [{str(default_boxplot_perc[boxplot_idx]*100)}%]')
        
        # Convert from list to numpy array
        d = np.array(d, dtype=object)

        # Create boxplot
        bp = ax.boxplot(d, 0, '', positions=positions, widths=widths)
        color_box(bp, 'b')
        
        # Set xticks and xticklabels
        ax.set_xticks(np.arange(n_xlabel))
        ax.set_xticklabels(xlabels)
        xlims = ax.get_xlim()
        ax.set_xlim([xlims[0]-0.1, xlims[1]-0.1])
        
        # Add legend
        leg_handle, = plt.plot([], [], 'b')
        ax.legend([leg_handle], ['Estimate'], loc='upper left')
        map(lambda x: x.set_visible(False), [leg_handle])
        
        # Save or show plot
        fig.tight_layout()
        if save:
            fig.savefig(eval_dir + f'/rel_translation_error_perc_comparison_{str(default_boxplot_perc[boxplot_idx])}.pdf',
                        bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Compare boxplot of relative errors between different test runs.''')
    parser.add_argument(
        'eval_dir', type=str,
        help="Folder containing the test runs with existing results.")
    parser.add_argument(
        '--plot_type', required=False, type=str, choices=['rel_trans', 'rel_trans_perc', 'rel_yaw'],
        help="Type of error to plot",
        default='rel_trans_perc')
    parser.add_argument('--save', dest='save',
                        action='store_true')
    parser.set_defaults(save=True)
    args = parser.parse_args()
    
    
    # Call boxplot comparison function
    boxplot_comparison(eval_dir=args.eval_dir, plot_type=args.plot_type, save=args.save)