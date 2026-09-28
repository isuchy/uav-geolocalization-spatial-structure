"""
DescriptorExtractor — single source of truth for tile descriptors.

Three strategies are supported, all producing L2-normalised descriptors:

  - 'cls': use the CLS token directly. Fast, no spatial info, not
    rotation-tolerant.
  - 'mean_patch': global arithmetic mean over all L2-normalised patch
    tokens. Smooth but loses spatial structure.
  - 'radial': radial token pooling. Each patch token is assigned to one
    of R concentric annular bins by its Euclidean distance to the patch
    grid centre. Within each bin tokens are averaged after per-token L2
    normalisation. The R per-bin vectors are concatenated and L2-normalised.
    This is the descriptor used in the article.

DESIGN NOTES
------------
- The pooling is MEAN, not GeM. DINOv3 patch tokens contain negative
  components, so a naïve GeM with clamp(min=eps) would discard half the
  signal. A signed GeM is a possible future extension but is NOT what the
  current pipeline ships, and the article text must say 'mean'.
- Bin boundaries are equal-radius (not quantile-balanced). Token counts
  per bin therefore vary across the descriptor; for a 32x32 patch grid
  and R=8 the inner bin holds ~2.3% of tokens and the busiest bin ~24%.
  This is documented honestly in the article.
- Approximate rotation tolerance: invariant for 90/180/270 deg discrete
  rotations of the patch grid; for intermediate angles the descriptor
  drifts because tokens cross bin boundaries when their radius changes
  on the discrete grid.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _make_radial_bin_ids(grid_h: int, grid_w: int, bins: int, device: torch.device) -> torch.Tensor:
    """Equal-radius binning of a grid_h x grid_w patch grid into R bins."""
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (grid_h - 1) / 2.0
    cx = (grid_w - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = rr.max().clamp_min(1e-6)
    bin_ids = torch.floor(rr / max_r * bins).long().clamp(0, bins - 1)
    return bin_ids.reshape(-1)


def _radial_pool(tokens: torch.Tensor, grid_h: int, grid_w: int, bins: int) -> torch.Tensor:
    """tokens: [B, grid_h*grid_w, D] -> [B, bins*D] L2-normalized."""
    bsz, n_tokens, dim = tokens.shape
    expected = grid_h * grid_w
    if n_tokens < expected:
        raise ValueError(f"Not enough patch tokens: got {n_tokens}, expected {expected}")
    tokens = tokens[:, :expected, :]
    tokens = F.normalize(tokens, p=2, dim=-1)

    bin_ids = _make_radial_bin_ids(grid_h, grid_w, bins, tokens.device)
    pooled = []
    for b in range(bins):
        mask = bin_ids == b
        if not torch.any(mask):
            pooled.append(torch.zeros((bsz, dim), device=tokens.device, dtype=tokens.dtype))
        else:
            pooled.append(tokens[:, mask, :].mean(dim=1))
    desc = torch.cat(pooled, dim=-1)
    desc = F.normalize(desc, p=2, dim=-1)
    return desc


class DescriptorExtractor:
    """
    Wraps a frozen DINOv3 backbone and exposes a single .extract(pixel_values)
    method. Selecting the strategy here makes the ablation script (09) trivial:
    just swap the string.
    """

    def __init__(
        self,
        model,
        strategy: str = "radial",
        radial_bins: int = 8,
    ):
        if strategy not in ("cls", "mean_patch", "radial"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.model = model
        self.strategy = strategy
        self.radial_bins = radial_bins

    def _split_tokens(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_token [B,D], patch_tokens [B,N,D]).
        DINOv3 layout: [CLS] + [register tokens] + [patch tokens].
        """
        num_register = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        cls = hidden[:, 0, :]
        patch_start = 1 + num_register
        patches = hidden[:, patch_start:, :]
        return cls, patches

    def _infer_grid(self, n_tokens: int, pixel_values: torch.Tensor) -> tuple[int, int]:
        patch_size = int(getattr(self.model.config, "patch_size", 16) or 16)
        grid_h = int(pixel_values.shape[-2] // patch_size)
        grid_w = int(pixel_values.shape[-1] // patch_size)
        if grid_h * grid_w == n_tokens:
            return grid_h, grid_w
        sq = int(math.sqrt(n_tokens))
        if sq * sq == n_tokens:
            return sq, sq
        raise ValueError(
            f"Cannot infer patch grid: n_tokens={n_tokens}, "
            f"pixel_values_shape={tuple(pixel_values.shape)}, patch_size={patch_size}"
        )

    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W] on the same device/dtype as the model.
        Returns: [B, D_out] L2-normalised float tensor.
        """
        outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        cls, patches = self._split_tokens(hidden)

        if self.strategy == "cls":
            desc = F.normalize(cls, p=2, dim=-1)
            return desc

        # mean_patch and radial use patch tokens
        grid_h, grid_w = self._infer_grid(patches.shape[1], pixel_values)

        if self.strategy == "mean_patch":
            patches_n = F.normalize(patches[:, : grid_h * grid_w, :], p=2, dim=-1)
            desc = patches_n.mean(dim=1)
            desc = F.normalize(desc, p=2, dim=-1)
            return desc

        # radial
        return _radial_pool(patches, grid_h, grid_w, self.radial_bins)


def multirotation_query(
    pixel_values: torch.Tensor,
    extractor: DescriptorExtractor,
    angles_deg: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    Query-time multi-rotation strategy.

    For each angle in `angles_deg`, rotate the input by k*90 degrees (only
    multiples of 90 are exact under tensor flip/rot90; intermediate angles
    would require interpolation and are intentionally not supported here),
    extract a descriptor, and return all R descriptors stacked.

    Returns: [B, R, D] tensor. The caller is expected to search each rotation
    against the index and keep the best score per DB tile.

    The DATABASE IS NOT ROTATED. multi-rotation is a query-time strategy only;
    it does not grow the index.
    """
    if angles_deg is None:
        angles_deg = [0, 90, 180, 270]

    descs = []
    for ang in angles_deg:
        if ang % 90 != 0:
            raise ValueError(f"multirotation_query supports multiples of 90 only, got {ang}")
        k = (ang // 90) % 4
        if k == 0:
            x = pixel_values
        else:
            x = torch.rot90(pixel_values, k=k, dims=(-2, -1)).contiguous()
        d = extractor.extract(x)
        descs.append(d)
    return torch.stack(descs, dim=1)  # [B, R, D]
