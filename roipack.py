# roipack.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np


class RoiPack:
    """Reader/utility for ROI mask packs.

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
        self._by_key = None  # lazy: (atlas, roi, hemi) -> h5 dataset path
        self._by_roi = None  # lazy: (atlas, roi) -> list[(hemi, path)]

    # --------------------------- index building ---------------------------

    @staticmethod
    def _load_index(
        h5: h5py.File, base_group: str = "/"
    ) -> Tuple[
        Dict[Tuple[str, str, str], str], Dict[Tuple[str, str], list[Tuple[str, str]]]
    ]:
        """Build indices from an opened HDF5 file, relative to base_group.

        Returns
        -------
        by_key : dict
            Mapping (atlas, roi, hemi) -> dataset path string.
        by_roi : dict
            Mapping (atlas, roi) -> list of (hemi, dataset path) tuples.
        """
        gbase = h5 if base_group in ("/", None) else h5[base_group]

        idx = []
        if "index" in gbase:
            atlas = list(gbase["index/atlas"][...])
            roi = list(gbase["index/roi"][...])
            hemi = list(gbase["index/hemi"][...])
            path = list(gbase["index/path"][...])
            for a, r, h, p in zip(atlas, roi, hemi, path):
                if isinstance(a, bytes):
                    a = a.decode()
                if isinstance(r, bytes):
                    r = r.decode()
                if isinstance(h, bytes):
                    h = h.decode()
                if isinstance(p, bytes):
                    p = p.decode()
                idx.append((a, r, h, p))
        else:
            g_masks = gbase.get("masks")
            if g_masks is not None:
                for atlas in g_masks:
                    g_atlas = g_masks[atlas]
                    for roi in g_atlas:
                        g_roi = g_atlas[roi]
                        for hemi in g_roi:
                            idx.append(
                                (
                                    atlas,
                                    roi,
                                    hemi,
                                    f"{g_masks.name}/{atlas}/{roi}/{hemi}",
                                )
                            )

        by_key = {(a, r, h): p for (a, r, h, p) in idx}
        by_roi: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
        for a, r, h, p in idx:
            by_roi.setdefault((a, r), []).append((h, p))
        return by_key, by_roi

    def _ensure_index(self) -> None:
        """Ensure that the internal indices are loaded from the HDF5 file.

        This method lazily loads the by_key and by_roi indices from the HDF5 file
        if they haven't been loaded yet. It uses the current base group based on
        the selected grid or surface mode.
        """
        if self._by_key is not None:
            return
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            self._by_key, self._by_roi = self._load_index(f, base_group=base)

    # --------------------------- dataset helpers ---------------------------

    @staticmethod
    def _dense_from_dataset(d: h5py.Dataset) -> np.ndarray:
        """Reconstruct a dense boolean mask from a stored dataset.

        Supports only index-based storage (current schema). Older bool modes
        are intentionally not supported to keep the code path minimal.
        """
        kind = d.attrs.get("kind")
        if kind != "index":
            # Treat any non-index as legacy dense array: convert nonzero to True
            arr = d[()]
            return np.asarray(arr) != 0
        full_shape = tuple(map(int, d.attrs.get("full_shape", ())))
        if not full_shape:
            raise ValueError("Missing full_shape attribute for index mask")
        flat = np.zeros(np.prod(full_shape, dtype=int), dtype=bool)
        idx = d[()].astype(np.int64)
        if idx.size:
            flat[idx] = True
        return flat.reshape(full_shape)

    # ---------------------------- grid selection -----------------------------

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

    # --------------------------------- API ------------------------------------

    def has(self, atlas: str, roi: str, hemi: str) -> bool:
        """Check if a specific ROI mask exists in the pack.

        Parameters
        ----------
        atlas : str
            The atlas name (e.g., 'benson', 'wang').
        roi : str
            The ROI name (e.g., 'V1', 'V2').
        hemi : str
            The hemisphere ('l', 'r', or 'both').

        Returns
        -------
        bool
            True if the ROI mask exists, False otherwise.
        """
        self._ensure_index()
        return (atlas, roi, hemi) in self._by_key  # type: ignore[operator]

    def list_atlases(self) -> list[str]:
        """Get a sorted list of all available atlas names in the pack.

        Returns
        -------
        list[str]
            Sorted list of atlas names.
        """
        self._ensure_index()
        return sorted({a for (a, r, h) in self._by_key.keys()})

    def list_rois(self, atlas: str) -> list[str]:
        """Get a sorted list of all ROI names for a specific atlas.

        Parameters
        ----------
        atlas : str
            The atlas name to list ROIs for.

        Returns
        -------
        list[str]
            Sorted list of ROI names for the specified atlas.
        """
        self._ensure_index()
        return sorted({r for (a, r, h) in self._by_key.keys() if a == atlas})

    def find(
        self,
        atlas: Optional[Iterable[str] | str] = None,
        roi: Optional[Iterable[str] | str] = None,
        hemi: Optional[Iterable[str] | str] = None,
    ) -> Iterator[Tuple[Tuple[str, str, str], str]]:
        """
        Iterate ((atlas, roi, hemi), hdf5_path) matching filters. Any filter can be None or a collection.
        """
        self._ensure_index()

        def _ok(val, flt):
            if flt is None:
                return True
            if isinstance(flt, (list, tuple, set)):
                return val in flt
            return val == flt

        for k, p in self._by_key.items():
            a, r, h = k
            if _ok(a, atlas) and _ok(r, roi) and _ok(h, hemi):
                yield k, p

    def get(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """
        Load a single mask (numpy array) for (atlas, roi, hemi).
        """
        self._ensure_index()
        path = self._by_key.get((atlas, roi, hemi))
        if path is None:
            raise KeyError(f"Mask not found: (atlas={atlas}, roi={roi}, hemi={hemi})")
        with h5py.File(str(self.h5_path), "r") as f:
            return self._dense_from_dataset(f[path])

    def get_both(self, atlas: str, roi: str) -> dict:
        """
        Returns {'l': arr, 'r': arr} if available; else {'both': arr} for volume fullBrain.
        """
        self._ensure_index()
        out = {}
        with h5py.File(str(self.h5_path), "r") as f:
            for hemi in ("l", "r"):
                path = self._by_key.get((atlas, roi, hemi))
                if path is not None:
                    out[hemi] = self._dense_from_dataset(f[path])
            if not out:
                path = self._by_key.get((atlas, roi, "both"))
                if path is not None:
                    out["both"] = self._dense_from_dataset(f[path])
        if not out:
            raise KeyError(f"No hemispheres available for (atlas={atlas}, roi={roi})")
        return out

    def _normalize_filter(self, val):
        """Normalize filter values for consistent processing.

        Parameters
        ----------
        val : str, iterable, or None
            The filter value to normalize. Can be None, 'all', a string, or an iterable.

        Returns
        -------
        None, set, or str
            - None if val is None or 'all' (match everything)
            - A set if val is an iterable
            - A set containing the string if val is a string
        """
        if val is None or (isinstance(val, str) and val.lower() == "all"):
            return None
        if isinstance(val, str):
            return {val}
        return set(val)

    def get_union(
        self,
        hemi: str,
        atlas: str | None = "all",
        roi: str | None = "all",
    ) -> np.ndarray:
        """
        Logical-OR across all masks matching the filters (atlas/roi/hemi).
        Returns a single boolean array with the same shape as the masks.

        Examples
        --------
        - All ROIs from all atlases in LH:   get_union(atlas="all", roi="all", hemi="l")
        - All Benson ROIs in RH:             get_union(atlas="benson", roi="all", hemi="r")
        - Specific list across hemis:        get_union(atlas=["benson","wang"], roi="V1", hemi=["l","r"])
        """
        self._ensure_index()
        A = self._normalize_filter(atlas)
        R = self._normalize_filter(roi)
        H = self._normalize_filter(hemi)

        out = None
        shape0 = None
        with h5py.File(str(self.h5_path), "r") as f:
            for (a, r, h), path in self._by_key.items():
                if A is not None and a not in A:
                    continue
                if R is not None and r not in R:
                    continue
                if H is not None and h not in H:
                    continue
                arr = self._dense_from_dataset(f[path])
                # initialize / validate shape
                if out is None:
                    out = arr.copy()
                    shape0 = out.shape
                    # ensure we're not accidentally mixing surface & volume
                    if out.ndim not in (1, 3):
                        raise ValueError(f"Unexpected mask rank {out.ndim} at {path}")
                else:
                    if arr.shape != shape0:
                        raise ValueError(
                            f"Shape mismatch in union: {arr.shape} vs {shape0} at {path}"
                        )
                    out |= arr
        if out is None:
            raise KeyError(
                f"No masks matched filters atlas={atlas}, roi={roi}, hemi={hemi}"
            )
        return out

    def get_all(
        self,
        hemi: str,
        atlas: str | None = "all",
        roi: str | None = "all",
    ) -> np.ndarray:
        """Alias of get_union(...)."""
        return self.get_union(atlas=atlas, roi=roi, hemi=hemi)

    def summary(self) -> dict:
        """
        Small metadata snapshot: counts per atlas/hemi and example shapes.
        """
        self._ensure_index()
        per_atlas = {}
        with h5py.File(str(self.h5_path), "r") as f:
            for (atlas, roi, hemi), path in self._by_key.items():
                d = f[path]
                shp = tuple(d.shape)
                per_atlas.setdefault(atlas, {}).setdefault(
                    hemi, {"count": 0, "example_shape": shp}
                )
                per_atlas[atlas][hemi]["count"] += 1
                if "example_shape" not in per_atlas[atlas][hemi]:
                    per_atlas[atlas][hemi]["example_shape"] = shp
        return {
            "key": self.key,
            "file": str(self.h5_path),
            "meta": self.meta,
            "atlases": per_atlas,
        }

    def get_union_index(self, hemi: Optional[str] = None) -> np.ndarray:
        """Return union flat indices (global indices into original space).

        Layouts handled:
        - Unified (volume or new surface): /union with datasets including flat_index.

                If a unified surface concatenation (grid_order == 'vertex_concat') is present:
                    - Attributes 'hemi_order' (array of hemisphere labels) and 'hemi_offsets'
                        (array of starting offsets) define the concatenation layout.
                    - When a hemisphere is specified, the returned indices are localized
                        to that hemisphere (i.e. returned indices are in that hemi's local
                        vertex index space, not the concatenated global space).
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            ug = f[base + "/union"] if base != "/" else f["union"]
            if "flat_index" in ug:  # unified union (volume or new surface)
                flat = ug["flat_index"][()].astype(np.int32)
                grid_order = ug.attrs.get("grid_order", "")
                # Volume hemisphere filtering via stored hemi_l_idx / hemi_r_idx
                if grid_order == "C" and hemi is not None:
                    hemi_key = f"hemi_{hemi}_idx"
                    if hemi_key in ug:
                        # Return only union indices that belong to this hemi by intersecting
                        hemi_idx = ug[hemi_key][()]
                        # hemi_idx are original-space voxel indices
                        # We need the subset of union 'flat' that are in hemi_idx, preserving order
                        mask = np.isin(flat, hemi_idx)
                        return flat[mask]
                if grid_order == "vertex_concat" and hemi is not None:
                    # Preferred attributes
                    hemi_ranges = None
                    if "hemi_order" in ug.attrs and "hemi_offsets" in ug.attrs:
                        hemi_order = [
                            h.decode() if isinstance(h, bytes) else h
                            for h in ug.attrs["hemi_order"]
                        ]
                        hemi_offsets = list(map(int, ug.attrs["hemi_offsets"]))
                        pairs = list(zip(hemi_order, hemi_offsets))
                        pairs.sort(key=lambda kv: kv[1])
                        hemi_ranges = {}
                        for i, (hh, st) in enumerate(pairs):
                            en = (
                                pairs[i + 1][1]
                                if i + 1 < len(pairs)
                                else (int(flat.max()) + 1 if flat.size else st)
                            )
                            hemi_ranges[hh] = (st, en)
                    if hemi_ranges:
                        if hemi not in hemi_ranges:
                            raise KeyError(
                                f"Requested hemi '{hemi}' not present in union offsets {list(hemi_ranges.keys())}"
                            )
                        st, en = hemi_ranges[hemi]
                        mask = (flat >= st) & (flat < en)
                        return (flat[mask] - st).astype(np.int32)
                return flat

    def get_roi_positions(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """Return union-order positions for an ROI.

        Works with unified (volume or concatenated surface).
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            ug = f[base + "/union"] if base != "/" else f["union"]
            atlas_arr = ug["atlas"][()].astype(str)
            roi_arr = ug["roi"][()].astype(str)
            hemi_arr = ug["hemi"][()].astype(str)
            indptr = ug["indptr"][()]
            indices = ug["indices"][()]
            # find row
            i = None
            for k, (a, r, h) in enumerate(zip(atlas_arr, roi_arr, hemi_arr)):
                if a == atlas and r == roi and h == hemi:
                    i = k
                    break
            if i is None:
                raise KeyError(f"ROI not found in union: {(atlas, roi, hemi)}")
            return indices[indptr[i] : indptr[i + 1]].astype(np.int32)

    def get_roi_flat_indices(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """Convenience: return original-space flat indices for an ROI via union mapping.

        This composes get_union_index(hemi=hemi) and get_roi_positions(atlas, roi, hemi)
        so callers can directly obtain the flat indices into the original vertex/voxel grid.
        """
        # Determine if surface per-hemi union is present by attempting hemi lookup
        try:
            union_flat = self.get_union_index(hemi=hemi)
        except (ValueError, KeyError):
            # Fall back to single union (volume or legacy layout)
            union_flat = self.get_union_index()
        pos = self.get_roi_positions(atlas, roi, hemi)
        return union_flat[pos]
