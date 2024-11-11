#!/usr/bin/env python3

# Standard library imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml

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

COLOR_MAP = 'magma_r'
CONDITION_MAP = {'t' :
                    {'0': '0ml', '1': '50ml', '2': '100ml'},
                'ms' :
                    {'0': '0.0g', '1': '1.5g', '2': '3.0g', '3': '4.5g'}}


def table_comparison(eval_dir : str, error_type : str = 'abs', metric_type : str = 'trans', save : bool = False):
    """ Table comparison of the root mean squared error (RMSE) for the different conditions.
    :param eval_dir: Folder containing the evaluation results for the different tests.
    :param error_type: Type of error to plot (valid options: 'abs', 'rel')
    :param metric_type: Type of metric to plot (valid options: 'trans', 'rot')
    :param save: If True, the plots are saved in the eval_dir folder. If False, the plots are shown.
    """
    # Load and initialise data
    tables_rel_perc = [10, 20, 30, 40, 50]
    num_tables = 1 if error_type == 'abs' else len(tables_rel_perc) if error_type == 'rel' else 0
    vals = np.full((num_tables, 3, 4), np.nan)
    # traj_lengths = np.full((3, 4), np.nan)
    for sub_dir in sorted(glob.glob(eval_dir + '/*')):
        if not Path(sub_dir).is_dir():
            continue

        # Save the identifier of the test run
        identifier = (int(Path(sub_dir).name[4]), int(Path(sub_dir).name[6]))
        # if Path(sub_dir).joinpath("saved_results/traj_est/cached/cached_rel_err.pickle").is_file():
        #     with suppress_stdout():
        #         traj = Trajectory(results_dir=sub_dir)
        #     traj_lengths[identifier] = traj.traj_length

        # Save the values for the table
        idx = 0
        for i, file in enumerate(sorted(glob.glob(sub_dir + "/saved_results/traj_est/" + error_type + "*.yaml"))):
            if error_type == 'rel' and (i+1)*10 not in tables_rel_perc:
                continue

            with open(file, 'r') as stream:
                data = yaml.safe_load(stream)

            vals[(idx,) + identifier] = np.around(data[metric_type]['rmse'], 2)
            idx += 1

    # Convert to percent of the full trajectory length
    # vals = vals / traj_lengths[None, :, :] * 100

    for idx in tqdm(range(num_tables), desc='Creating tables', leave=True, total=num_tables):
        # Create the table
        plt.rc('axes', titlesize=24, titlepad=15)
        fig, (ax_table, ax_colorbar) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [40, 1]}, frameon=False)
        title = f'rmse_{error_type}_{metric_type}_table' + ('' if error_type == 'abs' else f' [{tables_rel_perc[idx]}%]')
        ax_table.axis('off')
        ax_table.axis('tight')
        ax_table.set_title(latexify(title))

        # Define a colormap with a specific color for NaNs
        cmap = plt.get_cmap(COLOR_MAP)  # Base colormap
        colors = cmap(np.arange(cmap.N))  # Get colors from the base colormap
        colors = np.vstack((colors, [0, 0, 0, 1]))  # Add black color for NaNs
        cmap = mcolors.ListedColormap(colors)
        max_bound = 1 if metric_type == 'trans' else 100 if metric_type == 'rot' else 0
        bounds = np.linspace(0, max_bound, len(colors))  # Define bounds for each color
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Use a custom function to replace NaNs with the last index in the colormap
        colormap_vals = np.where(np.isnan(vals[idx,:,:]), len(colors), vals[idx,:,:])  # Replace NaNs with the index of black
        colors = cmap(norm(colormap_vals))

        # Plot the table
        table = ax_table.table(cellText=vals[idx,:,:],
                rowLabels=boldify(['T=' + CONDITION_MAP['t'][str(s)] for s in range(3)]),
                colLabels=boldify(['MS=' + CONDITION_MAP['ms'][str(s)] for s in range(4)]),
                colWidths=[0.16]*vals[idx,:,:].shape[1],
                loc='center',
                cellColours=colors)

        # Adjust the height of the rows
        for (i, j), cell in table.get_celld().items():
            if i == 0 or j == -1:
                cell.set_text_props(weight='bold')
            if i == 0: cell.set_height(0.10)  # Adjust this value as needed to increase row height
            else: cell.set_height(0.3)  # Adjust this value as needed to increase row height
            cell.set_text_props(fontsize=20)
            cell._loc = 'center'
        # table.scale(2, 2)  # Adjust this value as needed to increase row height

        # Plot colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=ax_colorbar, orientation='vertical')
        cbar.set_label(boldify('Value [' + ('m' if metric_type == 'trans' else 'deg' if metric_type == 'rot' else '') + ']'),
                       rotation=270, labelpad=10, fontsize=20)
        cbar.set_ticks([0, max_bound])
        cbar.ax.tick_params(labelsize=20)

        # Save or show plot
        plt.tight_layout()
        plt.subplots_adjust(wspace=-0.25)
        if save:
            plt.savefig(eval_dir + f'/{title}.pdf',
                        bbox_inches='tight',
                        transparent=True)
        else:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Create color-coded table of errors for each condition.''')
    parser.add_argument(
        'eval_dir', type=str,
        help="Folder containing the test runs with existing results.")
    parser.add_argument(
        '--error_type', required=False, type=str, choices=['abs', 'rel'],
        help="The type of error to plot.",
        default='abs')
    parser.add_argument(
        '--metric_type', required=False, type=str, choices=['trans', 'rot'],
        help="The type of metric to plot.",
        default='trans')
    parser.add_argument('--save', dest='save',
                        action='store_true')
    parser.set_defaults(save=True)
    args = parser.parse_args()


    # Call table comparison function
    table_comparison(eval_dir=args.eval_dir,
                    error_type=args.error_type,
                    metric_type=args.metric_type,
                    save=args.save)