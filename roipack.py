# roipack.py

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import h5py
import nibabel as nib
import numpy as np

#: Dataset creation kwargs, matching exactly what prfprepare writes for the
#: roi_pack_tree_v2 schema. Keeping these identical means an added ROI is
#: indistinguishable from an atlas ROI to any reader.
_CREATE_KW = dict(compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)


@contextmanager
def open_pack_for_write(h5_path, atomic: bool = True):
    """Open an ROI pack read-write.

    With ``atomic=True`` the edits go to a sibling temp copy that only replaces
    the original once it closed cleanly, so an interrupted write can never leave
    a half-written pack behind. The temp file has to be a sibling because
    ``os.replace`` is only atomic within one filesystem.

    Parameters
    ----------
    h5_path : Path | str
        The pack to edit.
    atomic : bool, optional
        Copy-then-replace when True, edit in place when False.

    Yields
    ------
    h5py.File
        The writable file object.
    """
    h5_path = Path(h5_path)

    def _open(p):
        try:
            return h5py.File(str(p), "r+")
        except OSError as e:
            raise OSError(
                f"could not open {p} for writing: {e}\n"
                "On network mounts (ceph/SMB) this is usually HDF5 file locking; "
                "try running with HDF5_USE_FILE_LOCKING=FALSE"
            ) from e

    if not atomic:
        f = _open(h5_path)
        try:
            yield f
        finally:
            f.close()
        return

    tmp = h5_path.with_name(h5_path.name + f".tmp{os.getpid()}")
    shutil.copyfile(str(h5_path), str(tmp))
    try:
        # keep the permission bits, but not BSD flags or timestamps: copystat
        # raises on SMB/ceph mounts and on tmpfs, and we replace the file anyway
        shutil.copymode(str(h5_path), str(tmp))
    except OSError:
        pass
    try:
        f = _open(tmp)
        try:
            yield f
        finally:
            f.close()
        os.replace(str(tmp), str(h5_path))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _put_int32(grp, name, data, attrs):
    """(Re)create an int32 index dataset with the pack's standard filters.

    HDF5 datasets here are fixed-size, so an existing one is deleted rather
    than resized.
    """
    if name in grp:
        del grp[name]
    ds = grp.create_dataset(
        name, data=np.ascontiguousarray(data, dtype=np.int32), **_CREATE_KW
    )
    for k, v in attrs.items():
        ds.attrs[k] = v
    return ds


def rebuild_index_group(f):
    """Rewrite ``/index`` from the current contents of ``/original_space``.

    The four index datasets are not resizable, so the whole group is dropped
    and recreated. Walking the groups in sorted order reproduces the ordering
    prfprepare writes (atlas, then roi, then hemi, all alphabetical).

    Returns
    -------
    int
        Number of (atlas, roi, hemi) entries written.
    """
    atl, roi, hem, pth = [], [], [], []
    if "original_space" in f:
        for a in sorted(f["original_space"].keys()):
            for r in sorted(f["original_space"][a].keys()):
                for h in sorted(f["original_space"][a][r].keys()):
                    atl.append(a)
                    roi.append(r)
                    hem.append(h)
                    pth.append(f"/original_space/{a}/{r}/{h}")

    if "index" in f:
        del f["index"]
    g = f.create_group("index")
    dt = h5py.string_dtype(encoding="utf-8")
    for name, vals in (("atlas", atl), ("hemi", hem), ("path", pth), ("roi", roi)):
        g.create_dataset(name, data=np.array(vals, dtype=object), dtype=dt)
    return len(pth)


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

    # --------------------------------- Write API ------------------------------------

    def list_roi_attrs(self, atlas: str, roi: str, hemi: str) -> dict:
        """Get all attributes of an ROI's original_space dataset.

        Used to read back provenance written by :meth:`write_surface_rois`.

        Parameters
        ----------
        atlas, roi, hemi : str
            The ROI to look up.

        Returns
        -------
        dict
            The dataset's HDF5 attributes.
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            p = f"original_space/{atlas}/{roi}/{hemi}"
            if p not in gbase:
                raise KeyError(f"ROI not found: {atlas}/{roi}/{hemi}")
            return dict(gbase[p].attrs)

    def hemi_vertex_count(self, hemi: str) -> Optional[int]:
        """Vertex count of a full hemisphere mesh, or None if not recorded.

        Prefers ``vertex_counts`` from the pack metadata (written by newer
        prfprepare versions), else the ``full_shape`` attribute of any
        original_space dataset for that hemisphere.

        Parameters
        ----------
        hemi : str
            Hemisphere ('l' or 'r').

        Returns
        -------
        int | None
            Number of vertices, or None when the pack records neither.
        """
        vc = self.meta.get("vertex_counts")
        if isinstance(vc, dict):
            for k in (hemi, hemi.lower(), hemi.upper()):
                if k in vc:
                    return int(vc[k])

        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]
            if "original_space" not in gbase:
                return None
            for a in gbase["original_space"]:
                for r in gbase["original_space"][a]:
                    grp = gbase["original_space"][a][r]
                    if hemi in grp and "full_shape" in grp[hemi].attrs:
                        return int(np.asarray(grp[hemi].attrs["full_shape"]).ravel()[0])
        return None

    def drop_atlas(self, atlas: str, rebuild_index: bool = True, atomic: bool = True):
        """Delete an atlas from both index spaces and from the extras group.

        Note that HDF5 does not reclaim the freed space, so this is meant for
        the small added atlases (like hand-drawn ROIs), not for the big ones.

        Parameters
        ----------
        atlas : str
            Atlas to remove.
        rebuild_index : bool, optional
            Rewrite ``/index`` afterwards.
        atomic : bool, optional
            Use the copy-then-replace path.

        Returns
        -------
        int
            Number of ROI groups removed.
        """
        n = 0
        with open_pack_for_write(self.h5_path, atomic=atomic) as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]
            for space in ("original_space", "masked_space", "full_extent"):
                p = f"{space}/{atlas}"
                if space in gbase and atlas in gbase[space]:
                    n += len(gbase[p].keys())
                    del gbase[p]
            if rebuild_index:
                rebuild_index_group(f)
        return n

    def write_surface_rois(self, specs, rebuild_index: bool = True, atomic: bool = True):
        """Write surface ROIs into the pack, as one atomic batch.

        Every spec must satisfy the pack's core invariant,
        ``original_indices == flat_index_<hemi>[masked_indices]`` with equal
        length and both strictly increasing. It is re-checked here and raises,
        because PRFclass' ``_maskROI_from_roipack`` indexes the two arrays
        positionally and would silently mismatch parameters and vertices.

        Writing the whole batch in one call matters: the atomic path copies the
        file, so one call per ROI would copy it once per ROI.

        Parameters
        ----------
        specs : list of dict
            Each with keys ``atlas``, ``roi``, ``hemi``, ``original_indices``,
            ``masked_indices``, ``n_vertices``, and optionally ``provenance``
            (extra attrs for both datasets) and ``full_extent`` (the unclipped
            vertex list, stored under ``/full_extent/<atlas>/<roi>/<hemi>``).
        rebuild_index : bool, optional
            Rewrite ``/index`` after the batch.
        atomic : bool, optional
            Use the copy-then-replace path.

        Returns
        -------
        int
            Number of ROIs written.
        """
        specs = list(specs)
        if not specs:
            return 0

        with open_pack_for_write(self.h5_path, atomic=atomic) as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]

            flats = {}
            for s in specs:
                h = s["hemi"]
                if h not in flats:
                    key = f"masked_space/flat_index_{h}"
                    if key not in gbase:
                        raise KeyError(f"masked space flat index missing for hemi {h}")
                    flats[h] = gbase[key][()].astype(np.int64)

            for s in specs:
                atlas, roi, hemi = s["atlas"], s["roi"], s["hemi"]
                o = np.asarray(s["original_indices"], dtype=np.int64)
                m = np.asarray(s["masked_indices"], dtype=np.int64)
                tag = f"{atlas}/{roi}/{hemi}"

                if o.size == 0:
                    raise ValueError(
                        f"{tag}: refusing to write an empty ROI; an empty dataset "
                        "would show up in list_rois() while masking nothing"
                    )
                if o.size != m.size:
                    raise ValueError(
                        f"{tag}: original ({o.size}) and masked ({m.size}) index "
                        "counts differ; they have to be positionally aligned"
                    )
                if np.any(np.diff(o) <= 0) or np.any(np.diff(m) <= 0):
                    raise ValueError(f"{tag}: indices have to be strictly increasing")
                flat = flats[hemi]
                if m.min() < 0 or m.max() >= flat.size:
                    raise ValueError(
                        f"{tag}: masked indices out of range for a masked space of "
                        f"{flat.size} vertices"
                    )
                if not np.array_equal(flat[m], o):
                    raise ValueError(
                        f"{tag}: flat_index[masked] != original, the two index "
                        "spaces are not aligned"
                    )

                prov = dict(s.get("provenance") or {})
                shared = dict(atlas=atlas, roi=roi, hemi=hemi, dtype="int32")
                _put_int32(
                    gbase.require_group(f"original_space/{atlas}/{roi}"),
                    hemi,
                    o,
                    {
                        **shared,
                        "kind": "index",
                        "order": "C",
                        "space_hint": "surface",
                        "full_shape": np.array([int(s["n_vertices"])], np.int32),
                        **prov,
                    },
                )
                _put_int32(
                    gbase.require_group(f"masked_space/{atlas}/{roi}"),
                    hemi,
                    m,
                    {
                        **shared,
                        "kind": "masked_space_indices",
                        "grid_order": "vertex",
                        **prov,
                    },
                )

                # The unclipped delineation, kept outside original_space on
                # purpose so it never turns up in list_atlases()/list_rois()
                # and cannot break the alignment invariant. Used for borders.
                full = s.get("full_extent")
                if full is not None:
                    full = np.unique(np.asarray(full, dtype=np.int64))
                    _put_int32(
                        gbase.require_group(f"full_extent/{atlas}/{roi}"),
                        hemi,
                        full,
                        {
                            **shared,
                            "kind": "full_extent_index",
                            "order": "C",
                            "space_hint": "surface",
                            "full_shape": np.array([int(s["n_vertices"])], np.int32),
                            **prov,
                        },
                    )

            if rebuild_index:
                rebuild_index_group(f)

        return len(specs)

    def get_full_extent_indices(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """Get the unclipped vertex indices of an ROI, if they were stored.

        Added ROIs (hand-drawn ones) keep their full delineation here, since
        ``original_space`` can only hold the part that survives the analysed
        mask. Falls back to ``original_space`` when no full extent was stored.

        Parameters
        ----------
        atlas, roi, hemi : str
            The ROI to look up.

        Returns
        -------
        np.ndarray
            Vertex indices in the original surface space.
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]
            p = f"full_extent/{atlas}/{roi}/{hemi}"
            if p in gbase:
                return gbase[p][()].astype(np.int32)
        return self.get_original_space_indices(atlas, roi, hemi)

    def has_full_extent(self, atlas: str, roi: str, hemi: str) -> bool:
        """Check whether an unclipped extent is stored for this ROI."""
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            gbase = f if base in ("/", None) else f[base]
            return f"full_extent/{atlas}/{roi}/{hemi}" in gbase
