from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import hashlib


class RoiPack:
    def __init__(self, key: str, h5_path: Path | str, meta: dict):
        self.key = str(key)
        self.h5_path = Path(h5_path)
        self.meta = dict(meta)
        self._by_key: Optional[Dict[Tuple[str, str, str], str]] = (
            None  # (atlas, roi, hemi) -> h5 path
        )
        self._by_roi: Optional[Dict[Tuple[str, str], list[Tuple[str, str]]]] = (
            None  # (atlas, roi) -> [(hemi, path)]
        )
        # For volume mode, selected grid id; None means surface or not selected
        self.grid_id: Optional[str] = None

    # ------------------------------- helpers -------------------------------------
    def _load_index(
        self: "RoiPack", h5, base_group: str = "/"
    ) -> Tuple[
        Dict[Tuple[str, str, str], str], Dict[Tuple[str, str], list[Tuple[str, str]]]
    ]:
    ]:
        """
        Build indices from an opened HDF5 file, relative to base_group.
        Returns:
        by_key: (atlas, roi, hemi) -> h5_path
        by_roi: (atlas, roi) -> [(hemi, h5_path), ...]
        """
        # resolve base
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

    def select_grid_for_input(self, path_or_img, round_dec: int = 6) -> str:
        """Select appropriate grid based on an input image (surface vs volume).
        - For GIFTI (surface), clears grid_id (uses root surface masks).
        - For NIfTI (volume), computes grid id and validates presence in file.
        Returns the selected grid id ("surface" as a label for surface mode).
        """
        # Load image
        if isinstance(path_or_img, (str, Path)):
            img = nib.load(str(path_or_img))
        else:
            img = path_or_img

        # Surface: GIFTI
        if hasattr(img, "agg_data"):
            # Clear selection for surface mode
            self.grid_id = None
            # Reset caches to force reindex under root
            self._by_key = None
            self._by_roi = None
            return "surface"

        # Volume: NIfTI-like
        shape = getattr(img, "shape", None)
        affine = getattr(img, "affine", None)
        if shape is None or affine is None:
            raise ValueError("Unsupported image object; expected nibabel image or path")
        gid = self._grid_signature(shape[:3], affine, round_dec=round_dec)
        with h5py.File(str(self.h5_path), "r") as f:
            if f.get(f"/grids/{gid}") is None:
                raise KeyError(
                    f"Grid {gid} not found in ROI pack (available: {list(f.get('grids', {}).keys()) if 'grids' in f else 'none'})"
                )
        self.grid_id = gid
        # Reset caches so index is reloaded under the selected grid
        self._by_key = None
        self._by_roi = None
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
            d = f[path]
            arr = d[()]
            if d.attrs.get("kind") == "bool":
                arr = arr.astype(bool)
            return arr

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
                    d = f[path]
                    arr = d[()]
                    if d.attrs.get("kind") == "bool":
                        arr = arr.astype(bool)
                    out[hemi] = arr
            if not out:
                path = self._by_key.get((atlas, roi, "both"))
                if path is not None:
                    d = f[path]
                    arr = d[()]
                    if d.attrs.get("kind") == "bool":
                        arr = arr.astype(bool)
                    out["both"] = arr
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
                d = f[path]
                arr = d[()]
                # normalize to bool
                if d.attrs.get("kind") == "bool":
                    arr = arr.astype(bool)
                else:
                    # 0/1 or labelmaps → treat nonzero as True
                    arr = np.asarray(arr) != 0
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

    def get_union_index(self):
        """Return union flat indices (global voxel indices) into the selected grid.
        For surface, returns indices into the flattened 1D vertex array.
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            ug = f[base + "/union"] if base != "/" else f["union"]
            return ug["flat_index"][()].astype(np.int32)

    def get_roi_positions(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """
        Return union-row positions (0..n_union-1) for this ROI, for the selected base group.
        """
        with h5py.File(str(self.h5_path), "r") as f:
            base = self._resolve_base_group(f)
            ug = f[base + "/union"] if base != "/" else f["union"]
            atlas_arr = ug["atlas"][()].astype(str)
            roi_arr = ug["roi"][()].astype(str)
            hemi_arr = ug["hemi"][()].astype(str)
            # locate row index i
            # (for speed, cache a dict[{(a,r,h)->i}] on first call if you like)
            i = None
            for k, (a, r, h) in enumerate(zip(atlas_arr, roi_arr, hemi_arr)):
                if a == atlas and r == roi and h == hemi:
                    i = k
                    break
            if i is None:
                raise KeyError(f"ROI not found: {(atlas, roi, hemi)}")
            indptr = ug["indptr"][()]
            indices = ug["indices"][()]
            return indices[indptr[i] : indptr[i + 1]].astype(np.int32)

        """
        get ROI positions in the union index
            # Load BOLD (X,Y,Z,T)
            img  = nib.load(str(in_path))
            data = img.get_fdata(dtype=np.float32)
            X,Y,Z,T = data.shape

            # 1) extract only union voxels once:
            u = roi_pack.get_union_index()              # shape (n_union,)
            vox_ts = data.reshape(-1, T)[u]             # shape (n_union, T)

            # 2) any ROI (overlaps allowed) → slice rows by positions:
            pos_v1l_wang   = roi_pack.get_roi_positions("wang",   "V1", "l")
            pos_v1l_benson = roi_pack.get_roi_positions("benson", "V1", "l")

            ts_v1l_wang   = vox_ts[pos_v1l_wang]        # (n_wang, T)
            ts_v1l_benson = vox_ts[pos_v1l_benson]      # (n_benson, T)

            # combine (union) or intersect as needed:
            pos_union = np.unique(np.concatenate([pos_v1l_wang, pos_v1l_benson]))
            pos_inter = np.intersect1d(pos_v1l_wang, pos_v1l_benson, assume_unique=False)

            ts_union = vox_ts[pos_union]
            ts_inter = vox_ts[pos_inter]
        """
