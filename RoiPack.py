import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

class RoiPack:
    """
    Read-only handle to an on-disk HDF5 ROI cache.

    Attributes
    ----------
    key : str
        Stable identifier (hash of canonical meta).
    h5_path : Path
        Path to the HDF5 file backing this ROI pack.
    meta : dict
        Provenance (sub, analysis_space, atlases, rois, fs_dir, key).
    """

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

    # ------------------------------- helpers -------------------------------------

    def _load_index(h5) -> Tuple[
        Dict[Tuple[str, str, str], str], Dict[Tuple[str, str], list[Tuple[str, str]]]
    ]:
        """
        Build indices from an opened HDF5 file.
        Returns:
        by_key: (atlas, roi, hemi) -> h5_path
        by_roi: (atlas, roi) -> [(hemi, h5_path), ...]
        """
        idx = []
        if "index" in h5:
            atlas = list(h5["index/atlas"][...])
            roi = list(h5["index/roi"][...])
            hemi = list(h5["index/hemi"][...])
            path = list(h5["index/path"][...])
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
            g_masks = h5.get("/masks")
            if g_masks is not None:
                for atlas in g_masks:
                    g_atlas = g_masks[atlas]
                    for roi in g_atlas:
                        g_roi = g_atlas[roi]
                        for hemi in g_roi:
                            idx.append((atlas, roi, hemi, f"/masks/{atlas}/{roi}/{hemi}"))

        by_key = {(a, r, h): p for (a, r, h, p) in idx}
        by_roi: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
        for a, r, h, p in idx:
            by_roi.setdefault((a, r), []).append((h, p))
        return by_key, by_roi

    def _ensure_index(self) -> None:
        if self._by_key is not None:
            return
        with h5py.File(str(self.h5_path), "r") as f:
            self._by_key, self._by_roi = self._load_index(f)

    # --------------------------------- API ------------------------------------

    def has(self, atlas: str, roi: str, hemi: str) -> bool:
        self._ensure_index()
        return (atlas, roi, hemi) in self._by_key  # type: ignore[operator]

    def list_atlases(self) -> list[str]:
        self._ensure_index()
        return sorted({a for (a, r, h) in self._by_key.keys()})

    def list_rois(self, atlas: str) -> list[str]:
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
        """None/'all' -> None (match everything); str -> {str}; iterable -> set."""
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
        """Return union flat indices (global voxel indices) into the 3D grid."""
        with h5py.File(str(self.h5_path), "r") as f:
            return f["union/flat_index"][()].astype(np.int32)

    def get_roi_positions(self, atlas: str, roi: str, hemi: str) -> np.ndarray:
        """
        Return union-row positions (0..n_union-1) for this ROI.
        """
        with h5py.File(str(self.h5_path), "r") as f:
            ug = f["union"]
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
