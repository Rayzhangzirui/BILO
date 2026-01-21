#!/usr/bin/env python
import argparse
import os
import numpy as np

import torch
from torch.utils.data import IterableDataset

from MatDataset import MatDataset

import matplotlib
import matplotlib.pyplot as plt

class GBMDataset(MatDataset):
    """GBM-specific dataset adapter.

    Expects a grid-style matfile providing (2D example):
    - gx: 1 x nx, gy: 1 x ny (or nx / ny variants)
    - x0: 1 x xdim (center)
    - tgrid: 1 x nt (time grid)
    - spatial fields: phi, P, Pwm, Pgm, DxPphi, DyPphi, (optional DzPphi)
    - data fields: u1, u2 (and/or others)

    Produces point-cloud style tensors compatible with existing training code:
    - X_res: (n_space*m_time) x (1+xdim) with time in column 0
        plus aligned *_res tensors for spatial fields.
    - X_dat: (nx*ny) x (1+xdim) with fixed time (default 1.0)
        plus aligned *_dat tensors for u1/u2 (and any requested dat fields).

    Notes:
    - Spatial fields are repeated across sampled time points.
    - All reformatting is done in NumPy to preserve Matlab ndgrid conventions.
        Outputs are converted to torch tensors for downstream training code.
    """

    SPATIAL_FIELDS = ("phi", "P", "Pwm", "Pgm", "DxPphi", "DyPphi", "DzPphi", "u1", "u2")
    SPACE_TIME_FIELDS = ("uchar", "ugt")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _as_1d_axis(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1):
            return a.reshape(-1)
        if a.ndim == 1:
            return a
        raise ValueError(f"Expected axis as 1D or 1xN/Nx1, got shape {tuple(a.shape)}")

    @staticmethod
    def _flat_ndgrid(z: np.ndarray) -> np.ndarray:
        # Matlab uses column-major order; ndgrid flattening must use Fortran order.
        return np.asarray(z).ravel(order="F")


    def build_centered_meshgrid(self, issphere=True) -> None:
        """Build centered axes + meshgrid for xdim=1/2/3.

        Expected axis keys: gx (+gy, +gz). Each as 1D or 1xN/Nx1.
        Expected center key: x0 with shape (1,xdim) or (xdim,).

        Produces:
        - {g*}_axis centered
        - {g*}_mesh tensors (nd)
        - X_space_flat: (Nspace, xdim)
        - r_space_flat: (Nspace,)
        """
        axes_keys = [k for k in ("gx", "gy", "gz") if k in self]

        # if 1d, no reshape
        x0 = self['x0']
        # if just a float, make it ndarray
        if np.isscalar(x0):
            x0 = np.array([x0])
        x0 = np.asarray(x0).reshape(-1)
        
        xdim_raw = self.get("xdim", len(axes_keys))
        xdim = int(np.asarray(xdim_raw).reshape(-1)[0])

        # Use first xdim axes (must exist)
        need = ("gx", "gy", "gz")[:xdim]

        axes_1d = []
        for i, k in enumerate(need):
            axis = self._as_1d_axis(self[k])
            L = float(np.asarray(self["L"]).reshape(-1)[0])
            axis = (axis - float(x0[i])) / L
            self[f"{k}_scaled"] = axis
            axes_1d.append(axis)

        # Meshgrid
        meshes = np.meshgrid(*axes_1d, indexing="ij")
        for k, m in zip(need, meshes):
            self[f"{k}_mesh"] = m

        # Flattened spatial coordinates: Nspace x xdim
        # Flatten with Fortran order to match Matlab ndgrid linearization.
        X_space_full = np.stack([m.ravel(order="F") for m in meshes], axis=1)
        r_full = np.sqrt(np.sum(X_space_full ** 2, axis=1))

        # Scaled time grid
        tgrid = self._as_1d_axis(self["tgrid"])
        self["t_scaled"] = tgrid/ tgrid.max()
        nt = int(tgrid.size)

        # Apply spherical mask if requested
        if issphere:
            if "rmax" not in self:
                raise KeyError("issphere=True requires dataset key 'rmax'")
            rmax = self["rmax"]/self["L"]
            mask = r_full <= rmax
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                raise ValueError(f"issphere=True selected 0 points (rmax={rmax})")
            self["mask_space"] = mask
            self["idx_space"] = idx
            self["X_space_flat"] = X_space_full[mask]
            self["r_space_flat"] = r_full[mask]
        else:
            self["X_space_flat"] = X_space_full
            self["r_space_flat"] = r_full

    def prepare_all_space_field(self):
        # Prepare flattened spatial fields
        self.build_centered_meshgrid()

        xdim = int(self["X_space_flat"].shape[1])
        space_mask = self.get("mask_space", None)
        
        for f in self.SPATIAL_FIELDS:
            if f not in self:
                continue
            z = self[f]
            z_flat = self._flat_ndgrid(z).reshape(-1, 1)
            if space_mask is not None:
                z_flat = z_flat[space_mask]
            self[f"{f}_flat"] = z_flat
        
        for f in self.SPACE_TIME_FIELDS:
            # for uchar and ugt, take the last time slice
            if f not in self:
                continue
            z = self[f]
            # take last time slice
            z_last = self._flat_ndgrid(z[..., -1]).reshape(-1, 1)
            if space_mask is not None:
                z_last = z_last[space_mask]
            self[f"{f}_flat"] = z_last
        
        # Convenience: gradPphi_res (n x xdim)
        terms = ("DxPphi_flat", "DyPphi_flat", "DzPphi_flat")[:xdim]
        grad_terms = []
        
        for k in range(xdim):
            g = self[terms[k]]
            grad_terms.append(g.reshape(-1, 1))
        self["gradPphi_flat"] = np.concatenate(grad_terms, axis=1)


    def prepare_all_spacetime_field(self, m_time=100):
        # for uchar and ugt, prepare space-time flattened fields

        X_space = self["X_space_flat"]  # Nspace x xdim
        n_space = int(X_space.shape[0])
        space_mask = self.get("mask_space", None)

        nt = self["t_scaled"].size
        xdim = self['xdim']
        # if m_times < nt, evenly subsample time points
        if m_time < nt:
            idx_t = np.linspace(0, nt - 1, int(m_time)).astype(int)
        else:
            idx_t = np.arange(nt, dtype=int)
            m_time = nt

        t_s = self["t_scaled"][idx_t].reshape(-1)  # (m_time,)
        
        for f in self.SPACE_TIME_FIELDS:
            if f not in self:
                continue
            z = self[f]
            assert z.ndim == xdim + 1, f"Expected space-time field {f} to have {xdim+1} dims, got {z.ndim}"
            assert z.shape[-1] == nt, f"Expected last dim of {f} to be nt={nt}, got {z.shape[-1]}"

            slices = []
            for ti in idx_t:
                s = self._flat_ndgrid(z[..., ti])
                if space_mask is not None:
                    s = s[space_mask]
                slices.append(s)
            z_flat = np.concatenate(slices, axis=0).reshape(-1, 1)
            self[f"{f}_st"] = z_flat

        
        t_rep = np.repeat(t_s, n_space).reshape(-1, 1) # (m_time*n_space, 1), t0 ... t0, t1 ... t1 
        xs_rep = np.tile(X_space, (m_time, 1)) # (m_time*n_space, xdim), x1...xn, x1...xn
        X_st = np.concatenate([t_rep, xs_rep], axis=1)
        phi_st = np.tile(self["phi_flat"], (m_time, 1))

        self["X_st"] = X_st
        self["phi_st"] = phi_st
    

    def configure_dataloader(self, res_batch_size, dat_batch_size, include_st):
        self.iter = {}
        self.batch = {}
        # for data loss
        self.iter['dat'] = iter(GBMDataIterableDataset(self, batch_size=dat_batch_size, device=self.device))
        # for pde loss
        self.iter['res'] = iter(GBMResidualIterableDataset(self, batch_size=res_batch_size, device=self.device))
        if include_st:
            self.iter['st'] = iter(GBMSpaceTimeIterableDataset(self, batch_size=res_batch_size, device=self.device))
        
        self.next_batch()
    
    def next_batch(self):
        # update batch for each loader
        for loader_name in self.iter:
            self.batch[loader_name] = next(self.iter[loader_name])
    
    def visualize_sampling(self, savedir=None):
        """Visualize sampling distributions."""
        if savedir is None:
            return

        for i in range(1, 3):
            print(f'visualizaiing batch {i}')
            _save_residual_batch_plots(next(self.iter['res']), i, savedir)
            _save_dat_batch_plots(next(self.iter['dat']), i, savedir)
            if 'st' in self.iter:
                _save_spacetime_batch_plots(next(self.iter['st']), i, savedir)

    



class GBMResidualIterableDataset(IterableDataset):
    """Infinite iterator yielding residual batches on GPU."""

    def __init__(self, gbm_dataset: "GBMDataset", batch_size: int, device):
        super().__init__()
        self.ds = gbm_dataset
        self.batch_size = int(batch_size)
        self.device = device

    def __iter__(self):
        ds = self.ds
        X_space = ds["X_space_flat"]

        device = X_space.device
        n_space = int(X_space.shape[0])

        phi = ds["phi_flat"]
        P = ds["P_flat"]
        grad = ds["gradPphi_flat"]

        while True:
            idx = torch.randint(0, n_space, (self.batch_size,), device=device)

            xs = X_space[idx]  # (B, xdim)
            t = torch.rand(self.batch_size, 1, device=device)
            # sort by time
            t, _ = torch.sort(t, dim=0)
            X = torch.cat([t, xs], dim=1)  # (B, 1+xdim)

            out = {
                "X_res_train": X,
                "phi_res_train": phi[idx],
                "P_res_train": P[idx],
                "gradPphi_res_train": grad[idx],
            }
            yield out


class GBMSpaceTimeIterableDataset(IterableDataset):
    """Infinite iterator yielding data/supervision batches on GPU."""

    def __init__(self, gbm_dataset: "GBMDataset", batch_size: int, device):
        super().__init__()
        self.ds = gbm_dataset
        self.batch_size = int(batch_size)
        self.device = device

    def __iter__(self):
        X_st = self.ds["X_st"]

        n = int(X_st.shape[0])
        phi = self.ds["phi_st"]
        uchar = self.ds.get("uchar_st", None)
        ugt = self.ds.get("ugt_st", None)

        if self.batch_size >= n:
            out = {"X_st": X_st, "phi_st": phi}
            if uchar is not None:
                out["uchar_st"] = uchar
            if ugt is not None:
                out["ugt_st"] = ugt
            while True:
                yield out

        while True:
            device = X_st.device
            idx = torch.randint(0, n, (self.batch_size,), device=device)
            out = {
                "X_st": X_st[idx],
                "phi_st": phi[idx],
            }
            if uchar is not None:
                out["uchar_st"] = uchar[idx]
            if ugt is not None:
                out["ugt_st"] = ugt[idx]
            yield out

class GBMDataIterableDataset(IterableDataset):
    """Infinite iterator yielding data/supervision batches on GPU."""

    def __init__(self, gbm_dataset: "GBMDataset", batch_size: int, device):
        super().__init__()
        self.ds = gbm_dataset
        self.batch_size = int(batch_size)
        self.device = device

    def __iter__(self):
        X_space = self.ds["X_space_flat"]
        n_space = int(X_space.shape[0])

        phi = self.ds['phi_flat']
        u1 = self.ds.get("u1_flat", None)
        u2 = self.ds.get("u2_flat", None)

        uchar = self.ds.get("uchar_flat", None)
        ugt = self.ds.get("ugt_flat", None)

        if self.batch_size >= n_space:
            X = torch.cat([torch.ones(n_space, 1, device=self.device), X_space], dim=1)
            out = {
                "X_dat_train": X,
                "phi_dat_train": phi,
                "u1_dat_train": u1,
                "u2_dat_train": u2,
                "uchar_dat_train": uchar,
                "ugt_dat_train": ugt
            }
            while True:
                yield out

        while True:
            idx = torch.randint(0, n_space, (self.batch_size,), device=self.device)
            xs = X_space[idx]  # (B, xdim)
            t = torch.ones(self.batch_size, 1, device=self.device)
            X = torch.cat([t, xs], dim=1)

            out = {
                "X_dat_train": X,
                "phi_dat_train": phi[idx],
                "u1_dat_train":  u1[idx],
                "u2_dat_train":  u2[idx],
                "uchar_dat_train": uchar[idx],
                "ugt_dat_train": ugt[idx]
            }
            yield out


def _batch_scatter_1d_x_vs_y(x, y, title: str, path: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=6)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _batch_scatter_r_vs_y(r, y, title: str, path: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(r, y, s=6)
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _batch_scatter_axis_vs_y_colored_by_t(
    x_axis,
    y,
    t,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x_axis, y, c=t, s=6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="t")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _batch_hist_1d(x, title: str, path: str, xlabel: str, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=bins, density=True)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _batch_scatter_xy_colored_ndgrid(x, y, z, title: str, path: str, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(y, x, c=z, s=6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_dat_batch_plots(b, batch_id: int, savedir: str) -> None:
    X = b["X_dat_train"].detach().cpu()
    xs = X[:, 1:]
    xdim = xs.shape[1]

    if xdim == 1:
        x = xs[:, 0].numpy()
    else:
        r = torch.norm(xs, dim=1).numpy()

    for k in ("uchar_dat_train", "ugt_dat_train"):
        if k not in b:
            continue
        y = b[k].detach().cpu().reshape(-1).numpy()
        if xdim == 1:
            _batch_scatter_1d_x_vs_y(
                x,
                y,
                title=f"data batch {batch_id}: {k} vs x",
                path=os.path.join(savedir, f"batch{batch_id}_dat_{k}_vs_x.png"),
                ylabel=k,
            )
        else:
            _batch_scatter_r_vs_y(
                r,
                y,
                title=f"data batch {batch_id}: {k} vs r",
                path=os.path.join(savedir, f"batch{batch_id}_dat_{k}_vs_r.png"),
                ylabel=k,
            )

def _save_spacetime_batch_plots(b, batch_id: int, savedir: str) -> None:
    X = b["X_st"].detach().cpu()
    t = X[:, 0].numpy()
    xs = X[:, 1:]
    xdim = xs.shape[1]
    if xdim == 1:
        x_axis = xs[:, 0].numpy()
        xlabel = "x"
    else:
        x_axis = torch.norm(xs, dim=1).numpy()
        xlabel = "r"

    for k in ("uchar_st", "ugt_st"):
        if k not in b:
            continue
        y = b[k].detach().cpu().reshape(-1).numpy()
        _batch_scatter_axis_vs_y_colored_by_t(
            x_axis=x_axis,
            y=y,
            t=t,
            xlabel=xlabel,
            ylabel=k,
            title=f"space time batch {batch_id}: {k}",
            path=os.path.join(savedir, f"batch{batch_id}_st_{k}.png"),
        )

def _save_residual_batch_plots(b, batch_id: int, savedir: str) -> None:
    X = b["X_res_train"].detach().cpu()
    t = X[:, 0].numpy()
    xs = X[:, 1:]
    r = torch.norm(xs, dim=1).numpy()
    xdim = xs.shape[1]

    _batch_hist_1d(
        t,
        title=f"residual batch {batch_id}: t distribution",
        path=os.path.join(savedir, f"batch{batch_id}_res_hist_t.png"),
        xlabel="t",
        bins=30,
    )
    _batch_hist_1d(
        r,
        title=f"residual batch {batch_id}: r distribution",
        path=os.path.join(savedir, f"batch{batch_id}_res_hist_r.png"),
        xlabel="r",
        bins=30,
    )

    # 2D: scatter x/y colored by space fields (ndgrid convention as )
    if xdim == 2:
        x = xs[:, 0].numpy()
        y = xs[:, 1].numpy()
        for k in ("phi_res_train", "P_res_train", "Pwm_res_train", "Pgm_res_train"):
            if k not in b:
                continue
            z = b[k].detach().cpu().reshape(-1).numpy()
            _batch_scatter_xy_colored_ndgrid(
                x=x,
                y=y,
                z=z,
                title=f"residual batch {batch_id}: (x,y) colored by {k}",
                path=os.path.join(savedir, f"batch{batch_id}_res_scatter_xy_{k}.png"),
                cbar_label=k,
            )
        
        for k in ("DxPphi_res_train", "DyPphi_res_train"):    
            z = b['gradPphi_res_train'].detach().cpu().numpy()
            if k == "DxPphi_res_train":
                z = z[:, 0]
            else:
                z = z[:, 1]
            _batch_scatter_xy_colored_ndgrid(
                    x=x,
                    y=y,
                    z=z,
                    title=f"residual batch {batch_id}: (x,y) colored by {k}",
                    path=os.path.join(savedir, f"batch{batch_id}_res_scatter_xy_{k}.png"),
                    cbar_label=k,
                )



if __name__ == "__main__":
    
    p = argparse.ArgumentParser(description="GBM dataset reformatter + sampling visualizer")
    p.add_argument("matfile", type=str, help="Path to .mat file")
    p.add_argument("--o", dest="savedir", type=str, required=True, help="Output directory for plots")
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    p.add_argument("--m_time", type=int, default=100, help="Number of time points for residual collocation")
    
    args = p.parse_args()

    ds = GBMDataset(args.matfile)
    ds.prepare_all_space_field()
    ds.prepare_all_spacetime_field(m_time=args.m_time)

    os.makedirs(args.savedir, exist_ok=True)

    # Iterable-collocation (preferred)
    ds.to_device(args.device)

    ds.configure_dataloader(
        res_batch_size=10000,
        dat_batch_size=10000,
        include_st=True,
    )

    ds.visualize_sampling(savedir=args.savedir)