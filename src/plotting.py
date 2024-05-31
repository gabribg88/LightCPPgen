import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from typing import Tuple

### Function for asymmetric color map taken from:
### https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

### Figure saving function taken from:
### https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/#saving-figures

def save_fig_original(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e))
            
            
def save_fig(
        fig, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [15, 15],  # Adjusted default size to match your clustermap size
        save: bool = True, 
        dpi: int = 300,
        transparent_png: bool = True,
    ):
    """Save the generated matplotlib figure or seaborn ClusterGrid to the specified 
    directory with the specified name and format.

    Parameters:
        fig (matplotlib.figure.Figure or seaborn.matrix.ClusterGrid): Figure or ClusterGrid to save
        fig_name (str): File name for saving the figure
        fig_dir (str): Directory path to save the figure
        fig_fmt (str): File format ('pdf', 'png', etc.)
        fig_size (tuple): Figure size in inches
        save (bool): If False, does not save the figure
        dpi (int): Resolution for saving raster formats
        transparent_png (bool): If True, saves PNGs with transparent background
    """
    if not save:
        return

    if isinstance(fig, matplotlib.figure.Figure):
        figure = fig
    elif hasattr(fig, 'fig'):  # Check if fig is a ClusterGrid
        figure = fig.fig
        fig_size = fig.fig.get_size_inches()  # Respect the size of the clustermap
    else:
        raise TypeError("Unsupported figure type")

    # Update figure size if needed
    figure.set_size_inches(fig_size, forward=True)
    pth = os.path.join(fig_dir, f"{fig_name}.{fig_fmt}")
    os.makedirs(fig_dir, exist_ok=True)
    
    if fig_fmt == 'pdf':
        metadata = {'Creator': '', 'Producer': '', 'CreationDate': None}
        figure.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        figure.patch.set_alpha(alpha)
        for ax in figure.axes:
            ax.patch.set_alpha(alpha)
        figure.savefig(pth, bbox_inches='tight', dpi=dpi, transparent=transparent_png)
    else:
        figure.savefig(pth, bbox_inches='tight')