import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from scipy.interpolate import griddata
from util import error_logging_decorator
'''
Plotting utilities for GBM problems
Assuming X has columns [t, x, y, z] for unstructured data
'''


@error_logging_decorator
def plot_grid_imshow_panels(
    pred2d,
    ref2d=None,
    *,
    fname='fig_grid_panels.png',
    savedir=None,
    title='',
    cmap='viridis',
):
    """Plot grid prediction using imshow.

    Args:
        pred2d: (nx, ny) array
        ref2d:  (nx, ny) array or None
    Notes:
        Uses ndgrid-like orientation: x down, y right (invert y-axis).
    """
    pred2d = np.asarray(pred2d)
    if ref2d is not None:
        ref2d = np.asarray(ref2d)
        vmin = float(np.min([pred2d.min(), ref2d.min()]))
        vmax = float(np.max([pred2d.max(), ref2d.max()]))
        err2d = np.abs(pred2d - ref2d)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ims = []
        ims.append(axes[0].imshow(pred2d, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='equal'))
        axes[0].set_title('pred')
        ims.append(axes[1].imshow(ref2d, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='equal'))
        axes[1].set_title('data')
        ims.append(axes[2].imshow(err2d, cmap=cmap, origin='lower', aspect='equal'))
        axes[2].set_title('abs error')

        for ax in axes:
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            ax.invert_yaxis()
        fig.suptitle(title)

        # colorbars
        fig.colorbar(ims[0], ax=axes[:2], fraction=0.046, pad=0.04, label='u (shared)')
        fig.colorbar(ims[2], ax=axes[2], fraction=0.046, pad=0.04, label='|err|')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(pred2d, cmap=cmap, origin='lower', aspect='equal')
        ax.set_title('pred')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.invert_yaxis()
        fig.suptitle(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='u')

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig


@error_logging_decorator
def plot_grid_contour_overlay(
    pred2d,
    ref2d=None,
    *,
    levels=None,
    fname='fig_grid_contour.png',
    savedir=None,
    title='',
):
    """Contour overlay for grid fields.

    Args:
        pred2d: (nx, ny) array
        ref2d:  (nx, ny) array or None
        levels: contour levels (shared)
    Notes:
        Uses ndgrid-like orientation: x down, y right (invert y-axis).
    """
    pred2d = np.asarray(pred2d)
    if levels is None:
        levels = np.array([0.01, 0.1, 0.3, 0.6])

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.contour(pred2d, levels=levels, linestyles='solid')

    if ref2d is not None:
        ref2d = np.asarray(ref2d)
        ax.contour(ref2d, levels=levels, linestyles='dashed')

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig, ax


@error_logging_decorator
def plot_grid_segmentation_overlay(
    pred2d,
    *,
    th1,
    th2,
    u1=None,
    u2=None,
    fname='fig_grid_seg_overlay.png',
    savedir=None,
    title='',
    gt_alpha=0.35,
):
    """Overlay predicted segmentation contours onto filled GT segmentations.

    Predicted segmentation is defined as contours of the predicted density `pred2d`
    at the inferred thresholds `th1` and `th2`.

    Args:
        pred2d: (nx, ny) array of predicted density (optionally already masked)
        th1: scalar threshold for seg1
        th2: scalar threshold for seg2
        u1: (nx, ny) array (ground-truth seg1 mask/field)
        u2: (nx, ny) array (ground-truth seg2 mask/field)
    Notes:
        Uses ndgrid-like orientation: x down, y right (invert y-axis).
    """
    pred2d = np.asarray(pred2d)
    th1 = float(np.asarray(th1).reshape(-1)[0])
    th2 = float(np.asarray(th2).reshape(-1)[0])

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    handles = []
    labels = []

    # Filled GT masks (u1 larger region than u2; plot u1 first then u2)
    if u1 is not None:
        u1 = np.asarray(u1)
        m1 = u1 > 0.5
        if np.any(m1):
            ax.contourf(m1.astype(float), levels=[0.5, 1.5], colors=['tab:blue'], alpha=gt_alpha)
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='tab:blue', alpha=gt_alpha))
            labels.append('GT seg1 (u1)')

    if u2 is not None:
        u2 = np.asarray(u2)
        m2 = u2 > 0.5
        if np.any(m2):
            ax.contourf(m2.astype(float), levels=[0.5, 1.5], colors=['tab:orange'], alpha=gt_alpha)
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='tab:orange', alpha=gt_alpha))
            labels.append('GT seg2 (u2)')

    # Predicted threshold contours
    ax.contour(pred2d, levels=[th1], colors=['tab:blue'], linestyles='solid', linewidths=2.0)
    ax.contour(pred2d, levels=[th2], colors=['tab:orange'], linestyles='solid', linewidths=2.0)
    handles.append(Line2D([0], [0], color='tab:blue', linewidth=2.0))
    labels.append(f'pred contour @ th1={th1:.3f}')
    handles.append(Line2D([0], [0], color='tab:orange', linewidth=2.0))
    labels.append(f'pred contour @ th2={th2:.3f}')

    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    if handles:
        ax.legend(handles, labels, loc='best', frameon=True)

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        fpath = os.path.join(savedir, fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'fig saved to {fpath}')

    return fig, ax

