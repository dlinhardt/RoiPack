# roipack.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import h5py
import nibabel as nib
import numpy as np


class RoiPack:
    """Simplified reader for ROI mask packs with original_space and masked_space structure.

    Parameters
    ----------
    key : str
        Cache key identifying this pack's meta combination.
    h5_path : Path | str
        Path to the HDF5 file containing the masks.
    meta : dict
        Canonical metadata dictionary (includes 'analysis_space').
    grid_id : str | None, optional
        For volume mode, an explicitly chosen grid id if multiple are present.
    """

    def __init__(
        self, key: str, h5_path: Path | str, meta: dict, grid_id: str | None = None
    ):
        self.key = key
        self.h5_path = Path(h5_path)
        self.meta = dict(meta)
        self.grid_id = grid_id

    # --------------------------- grid selection ---------------------------

    @staticmethod
    def _grid_signature(shape, affine, round_dec: int = 6) -> str:
        """Generate a unique signature for a volume grid based on shape and affine.

        Parameters
        ----------
        shape : array-like
            The spatial shape of the volume (X, Y, Z).
        affine : array-like
            The affine transformation matrix (4x4).
        round_dec : int, optional
            Number of decimal places to round affine values, by default 6.

        Returns
        -------
        str
            A 12-character hexadecimal hash representing the grid signature.
        """
        A = np.asarray(affine, float).round(round_dec)
        s = np.asarray(shape, int)
        h = hashlib.sha1()
        h.update(s.tobytes())
        h.update(A.tobytes())
        return h.hexdigest()[:12]

    def _resolve_base_group(self, f) -> str:
        """Return base group for current mode.
        Surface (fsnative/fsaverage) -> "/"
        Volume -> f"grids/{grid_id}" (selected or inferred if single grid)
        """
        space = str(self.meta.get("analysis_space", "")).lower()
        if space in ("fsnative", "fsaverage"):
            return "/"
        # volume mode
        if self.grid_id:
            return f"grids/{self.grid_id}"
        # If only one grid exists, auto-select it
        if "grids" in f:
            grids = list(f["grids"].keys())
            if len(grids) == 1:
                return f"grids/{grids[0]}"
        # Default to root if no grids (legacy files)
        return "/"

    def select_grid_for_input(self, arg, round_dec: int = 6) -> str:
        """Select appropriate grid based on a single argument (string or image).

        Argument can be:
        - A space label string: one of {"volume", "fsnative", "fsaverage", "surface"}
        - A path string to an image (NIfTI → volume; GIFTI → surface)
        - A nibabel image object (NIfTI-like or GIFTI)

        Returns the selected volume grid id, or "surface" when selecting surface mode.
        """

        if arg is None:
            raise ValueError(
                "select_grid_for_input requires a string (label or path) or an image object"
            )

        # If a string, decide whether it's a space label or a file path
        if isinstance(arg, (str, Path)):
            s = str(arg)
            lbl = s.lower()
            # Case 1: label
            if lbl in {"volume", "fsnative", "fsaverage", "surface"}:
                if lbl in {"fsnative", "fsaverage", "surface"}:
                    # Surface mode
                    self.grid_id = None
                    # Also update meta hint so base group resolves to root
                    self.meta["analysis_space"] = (
                        lbl if lbl != "surface" else "fsnative"
                    )
                    return "surface"
                # Volume label without image: auto-select if single grid exists
                with h5py.File(str(self.h5_path), "r") as f:
                    if "grids" in f:
                        grids = list(f["grids"].keys())
                        if len(grids) == 1:
                            self.grid_id = grids[0]
                            self.meta["analysis_space"] = "volume"
                            return grids[0]
                        raise KeyError(
                            f"Multiple grids available {grids}; provide an image to disambiguate."
                        )
                    raise KeyError(
                        "No grids group found; cannot select volume without image"
                    )
            # Case 2: path to image → load it and proceed below
            img = nib.load(s)
        else:
            # Treat as nibabel image
            img = arg

        # Surface: GIFTI has 'agg_data'
        if hasattr(img, "agg_data"):
            self.grid_id = None
            self.meta["analysis_space"] = "fsnative"
            return "surface"

        # Volume: NIfTI-like
        shape = getattr(img, "shape", None)
        affine = getattr(img, "affine", None)
        if shape is None or affine is None:
            raise ValueError(
                "Unsupported image object; expected NIfTI-like with shape and affine"
            )
        gid = self._grid_signature(shape[:3], affine, round_dec=round_dec)
        with h5py.File(str(self.h5_path), "r") as f:
            if f.get(f"/grids/{gid}") is None:
                raise KeyError(
                    f"Grid {gid} not found in ROI pack (available: {list(f.get('grids', {}).keys()) if 'grids' in f else 'none'})"
                )
        self.grid_id = gid
        self.meta["analysis_space"] = "volume"
        return gid

    # --------------------------------- Core API ------------------------------------

    def get_original_space_indices(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """Get indices for ROI in original coordinate space (vertices/voxels).

        Parameters
        ----------
        atlas : str
            Atlas name (e.g., 'benson', 'wang', 'fullBrain')
        roi : str
            ROI name (e.g., 'V1', 'V2')
        hemi : str
            Hemisphere ('l' or 'r')

        Returns
        -------
        np.ndarray
            Flat indices into original coordinate space where ROI is active
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            path = f"original_space/{atlas}/{roi}/{hemi}"
            if path not in gbase:
                raise KeyError(f"ROI not found: {atlas}/{roi}/{hemi}")

            return gbase[path][()].astype(np.int32)

    def get_masked_space_indices(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """Get indices for ROI in masked coordinate space (compressed).

        Parameters
        ----------
        atlas : str
            Atlas name (e.g., 'benson', 'wang', 'fullBrain')
        roi : str
            ROI name (e.g., 'V1', 'V2')
        hemi : str
            Hemisphere ('l' or 'r')

        Returns
        -------
        np.ndarray
            Indices into masked space where ROI is active
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            path = f"masked_space/{atlas}/{roi}/{hemi}"
            if path not in gbase:
                raise KeyError(f"ROI not found in masked space: {atlas}/{roi}/{hemi}")

            return gbase[path][()].astype(np.int32)

    def get_masked_space_flat_index(self, hemi: str) -> np.ndarray:
        """Get mapping from masked space to original space for a hemisphere.

        Parameters
        ----------
        hemi : str
            Hemisphere ('l' or 'r')

        Returns
        -------
        np.ndarray
            Flat indices mapping masked space positions to original space
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            flat_index_key = f"flat_index_{hemi}"
            if f"masked_space/{flat_index_key}" not in gbase:
                raise KeyError(
                    f"Masked space flat index not found for hemisphere {hemi}"
                )

            return gbase[f"masked_space/{flat_index_key}"][()].astype(np.int32)

    def get_grid_shape(self) -> tuple[int, int, int]:
        """Get the 3D shape of the current volume grid.

        Returns
        -------
        tuple[int, int, int]
            Shape (nx, ny, nz) of the volume grid

        Raises
        ------
        ValueError
            If not in volume mode or no grid selected
        KeyError
            If shape information not found in HDF5 file
        """
        space = str(self.meta.get("analysis_space", "")).lower()
        if space in ("fsnative", "fsaverage"):
            raise ValueError("get_grid_shape() only available in volume mode")

        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            # Try to find any dataset with full_shape attribute
            for space_type in ["original_space", "masked_space"]:
                if space_type in gbase:
                    space_group = gbase[space_type]
                    for atlas_name in space_group:
                        atlas_group = space_group[atlas_name]
                        for roi_name in atlas_group:
                            roi_group = atlas_group[roi_name]
                            for hemi_name in roi_group:
                                dataset = roi_group[hemi_name]
                                if "full_shape" in dataset.attrs:
                                    shape = tuple(dataset.attrs["full_shape"])
                                    return shape

            raise KeyError("No grid shape information found in HDF5 file")

    def get_original_space_coordinates(
        self, atlas: str, roi: str, hemi: str
    ) -> np.ndarray:
        """Get 3D coordinates for ROI in volume space.

        Parameters
        ----------
        atlas : str
            Atlas name (e.g., 'benson', 'wang', 'fullBrain')
        roi : str
            ROI name (e.g., 'V1', 'V2')
        hemi : str
            Hemisphere ('l' or 'r')

        Returns
        -------
        np.ndarray
            Array of shape (n_voxels, 3) with (i, j, k) coordinates

        Raises
        ------
        ValueError
            If not in volume mode
        """
        space = str(self.meta.get("analysis_space", "")).lower()
        if space in ("fsnative", "fsaverage"):
            raise ValueError(
                "get_original_space_coordinates() only available in volume mode"
            )

        flat_indices = self.get_original_space_indices(atlas, roi, hemi)
        if flat_indices.size == 0:
            return np.array([], dtype=int).reshape(0, 3)

        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            try:
                grid_shape = np.array(gbase.attrs["grid_shape"]).astype(float)
            except KeyError:
                raise KeyError("Grid shape attribute not found in HDF5 file")

        coordinates = np.vstack(np.unravel_index(flat_indices, grid_shape)).T
        return coordinates.astype(np.int32)

    # --------------------------------- Utility methods ------------------------------------

    def has_roi(self, atlas: str, roi: str, hemi: str) -> bool:
        """Check if ROI exists in original space."""
        try:
            self.get_original_space_indices(atlas, roi, hemi)
            return True
        except KeyError:
            return False

    def list_atlases(self) -> list[str]:
        """Get sorted list of available atlases."""
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            if "original_space" not in gbase:
                return []

            return sorted(list(gbase["original_space"].keys()))

    def list_rois(self, atlas: str) -> list[str]:
        """Get sorted list of ROIs for an atlas."""
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            atlas_path = f"original_space/{atlas}"
            if atlas_path not in gbase:
                return []

            return sorted(list(gbase[atlas_path].keys()))

    def list_hemispheres(self, atlas: str, roi: str) -> list[str]:
        """Get available hemispheres for an atlas/ROI combination."""
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            roi_path = f"original_space/{atlas}/{roi}"
            if roi_path not in gbase:
                return []

            return sorted(list(gbase[roi_path].keys()))

    def get_dense_mask(
        self, atlas: str, roi: str, hemi: str, shape: tuple
    ) -> np.ndarray:
        """Convert sparse indices to dense boolean mask.

        Parameters
        ----------
        atlas : str
            Atlas name
        roi : str
            ROI name
        hemi : str
            Hemisphere
        shape : tuple
            Shape of the output mask

        Returns
        -------
        np.ndarray
            Dense boolean mask with True where ROI is active
        """
        indices = self.get_original_space_indices(atlas, roi, hemi)
        mask = np.zeros(np.prod(shape), dtype=bool)
        if indices.size > 0:
            mask[indices] = True
        return mask.reshape(shape)

    def convert_masked_to_original(
        self, masked_indices: np.ndarray, hemi: str
    ) -> np.ndarray:
        """Convert masked space indices to original space indices.

        Parameters
        ----------
        masked_indices : np.ndarray
            Indices in masked space
        hemi : str Hemisphere

        Returns
        -------
        np.ndarray
            Corresponding indices in original space
        """
        flat_index = self.get_masked_space_flat_index(hemi)
        return flat_index[masked_indices]

    def summary(self) -> dict:
        """Get summary of available data."""
        atlases = self.list_atlases()
        summary = {
            "key": self.key,
            "file": str(self.h5_path),
            "meta": self.meta,
            "atlases": {},
        }

        for atlas in atlases:
            rois = self.list_rois(atlas)
            summary["atlases"][atlas] = {}
            for roi in rois:
                hemispheres = self.list_hemispheres(atlas, roi)
                summary["atlases"][atlas][roi] = hemispheres

        return summary
