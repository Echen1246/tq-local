from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import gamma, sqrt

import numpy as np
import torch


@dataclass(frozen=True)
class ScalarCodebook:
    dimension: int
    bits: int
    centers: np.ndarray
    boundaries: np.ndarray
    grid_size: int


@dataclass(frozen=True)
class BatchMetrics:
    num_vectors: int
    dimension: int
    bits: int
    mse: float
    rmse: float
    mean_cosine: float
    min_cosine: float
    max_cosine: float
    mean_vector_norm: float
    std_vector_norm: float
    mean_residual_norm: float
    std_residual_norm: float
    inner_product_mse: float


def _sphere_coordinate_density(dimension: int, x: np.ndarray) -> np.ndarray:
    density = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) <= 1.0
    coefficient = gamma(dimension / 2.0) / (sqrt(np.pi) * gamma((dimension - 1.0) / 2.0))
    density[mask] = coefficient * np.power(1.0 - np.square(x[mask]), (dimension - 3.0) / 2.0)
    return density


def _cdf_from_density(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    dx = grid[1] - grid[0]
    cumulative = np.cumsum(density) * dx
    cumulative /= cumulative[-1]
    cumulative[0] = 0.0
    cumulative[-1] = 1.0
    return cumulative


def _conditional_mean(grid: np.ndarray, density: np.ndarray, left: float, right: float) -> float:
    if right <= left:
        return (left + right) / 2.0
    mask = (grid >= left) & (grid <= right)
    local_grid = grid[mask]
    local_density = density[mask]
    if local_grid.size == 0:
        return (left + right) / 2.0
    weights = local_density
    denom = np.trapezoid(weights, local_grid)
    if denom <= 0:
        return (left + right) / 2.0
    numer = np.trapezoid(local_grid * weights, local_grid)
    return float(numer / denom)


@lru_cache(maxsize=32)
def build_scalar_codebook(
    dimension: int,
    bits: int,
    grid_size: int = 32769,
    max_iters: int = 200,
    tol: float = 1e-10,
) -> ScalarCodebook:
    if bits <= 0:
        raise ValueError("bits must be positive for TurboQuant_mse.")
    levels = 2**bits
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    density = _sphere_coordinate_density(dimension, grid)
    cdf = _cdf_from_density(grid, density)

    quantiles = np.linspace(0.0, 1.0, levels + 2, dtype=np.float64)[1:-1]
    centers = np.interp(quantiles, cdf, grid)

    for _ in range(max_iters):
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        interval_edges = np.concatenate(([-1.0], boundaries, [1.0]))
        updated = np.array(
            [
                _conditional_mean(grid, density, interval_edges[i], interval_edges[i + 1])
                for i in range(levels)
            ],
            dtype=np.float64,
        )
        if np.max(np.abs(updated - centers)) < tol:
            centers = updated
            break
        centers = updated

    boundaries = (centers[:-1] + centers[1:]) / 2.0
    return ScalarCodebook(
        dimension=dimension,
        bits=bits,
        centers=centers,
        boundaries=boundaries,
        grid_size=grid_size,
    )


@lru_cache(maxsize=16)
def random_rotation_matrix(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    return q.astype(np.float32)


def quantize_vectors_mse(
    vectors: np.ndarray,
    bits: int,
    seed: int = 0,
    grid_size: int = 32769,
) -> tuple[np.ndarray, np.ndarray]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected shape [num_vectors, dimension], got {vectors.shape}.")

    vectors32 = np.asarray(vectors, dtype=np.float32)
    num_vectors, dimension = vectors32.shape
    codebook = build_scalar_codebook(dimension=dimension, bits=bits, grid_size=grid_size)
    rotation = random_rotation_matrix(dimension=dimension, seed=seed)

    norms = np.linalg.norm(vectors32, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    normalized = vectors32 / safe_norms
    rotated = normalized @ rotation.T

    indices = np.digitize(rotated, codebook.boundaries, right=False)
    quantized_rotated = codebook.centers[indices]
    reconstructed = (quantized_rotated @ rotation) * safe_norms
    reconstructed = reconstructed.astype(np.float32)
    return reconstructed, indices.astype(np.int16)


def reconstruct_tensor_mse(
    tensor: torch.Tensor,
    bits: int,
    seed: int = 0,
    grid_size: int = 32769,
) -> torch.Tensor:
    tensor32 = tensor.detach().to(dtype=torch.float32, device="cpu")
    array = tensor32.numpy()
    original_shape = array.shape
    reconstructed, _ = quantize_vectors_mse(
        vectors=array.reshape(-1, original_shape[-1]),
        bits=bits,
        seed=seed,
        grid_size=grid_size,
    )
    reconstructed = reconstructed.reshape(original_shape)
    return torch.from_numpy(reconstructed).to(device=tensor.device, dtype=tensor.dtype)


def quantize_past_key_values_mse(
    past_key_values,
    bits: int,
    seed: int = 0,
    grid_size: int = 32769,
    token_slice: slice | None = None,
):
    def _reconstruct_selected(tensor: torch.Tensor) -> torch.Tensor:
        if token_slice is None:
            return reconstruct_tensor_mse(
                tensor=tensor,
                bits=bits,
                seed=seed,
                grid_size=grid_size,
            )
        updated = tensor.clone()
        updated[:, :, token_slice, :] = reconstruct_tensor_mse(
            tensor[:, :, token_slice, :],
            bits=bits,
            seed=seed,
            grid_size=grid_size,
        )
        return updated

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if getattr(layer, "keys", None) is None or getattr(layer, "values", None) is None:
                continue
            layer.keys = _reconstruct_selected(layer.keys)
            layer.values = _reconstruct_selected(layer.values)
        return past_key_values

    if isinstance(past_key_values, (tuple, list)):
        updated_layers = []
        for layer_cache in past_key_values:
            if not isinstance(layer_cache, (tuple, list)):
                raise TypeError(f"Unsupported layer cache type: {type(layer_cache)!r}")
            items = list(layer_cache)
            tensor_indexes = [index for index, item in enumerate(items) if torch.is_tensor(item)]
            if len(tensor_indexes) < 2:
                raise ValueError(
                    f"Expected at least two tensor entries in a layer cache, got {len(tensor_indexes)}."
                )
            items[tensor_indexes[0]] = _reconstruct_selected(items[tensor_indexes[0]])
            items[tensor_indexes[1]] = _reconstruct_selected(items[tensor_indexes[1]])
            updated_layers.append(tuple(items) if isinstance(layer_cache, tuple) else items)
        return tuple(updated_layers) if isinstance(past_key_values, tuple) else updated_layers

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def evaluate_quantization(
    vectors: np.ndarray,
    reconstructed: np.ndarray,
    query_seed: int = 0,
    num_query_samples: int = 128,
) -> BatchMetrics:
    vectors32 = np.asarray(vectors, dtype=np.float32)
    reconstructed32 = np.asarray(reconstructed, dtype=np.float32)
    residual = vectors32 - reconstructed32

    mse = float(np.mean(np.square(residual)))
    vector_norms = np.linalg.norm(vectors32, axis=1)
    residual_norms = np.linalg.norm(residual, axis=1)

    denom = np.linalg.norm(vectors32, axis=1) * np.linalg.norm(reconstructed32, axis=1)
    safe_denom = np.where(denom > 0, denom, 1.0)
    cosines = np.sum(vectors32 * reconstructed32, axis=1) / safe_denom

    rng = np.random.default_rng(query_seed)
    queries = rng.standard_normal((num_query_samples, vectors32.shape[1]), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    original_dots = vectors32 @ queries.T
    reconstructed_dots = reconstructed32 @ queries.T
    inner_product_mse = float(np.mean(np.square(original_dots - reconstructed_dots)))

    return BatchMetrics(
        num_vectors=int(vectors32.shape[0]),
        dimension=int(vectors32.shape[1]),
        bits=0,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mean_cosine=float(np.mean(cosines)),
        min_cosine=float(np.min(cosines)),
        max_cosine=float(np.max(cosines)),
        mean_vector_norm=float(np.mean(vector_norms)),
        std_vector_norm=float(np.std(vector_norms)),
        mean_residual_norm=float(np.mean(residual_norms)),
        std_residual_norm=float(np.std(residual_norms)),
        inner_product_mse=inner_product_mse,
    )


def turboquant_mse_analyze(
    vectors: np.ndarray,
    bits: int,
    seed: int = 0,
    query_seed: int = 0,
    num_query_samples: int = 128,
    grid_size: int = 32769,
) -> dict[str, float | int]:
    reconstructed, _ = quantize_vectors_mse(
        vectors=vectors,
        bits=bits,
        seed=seed,
        grid_size=grid_size,
    )
    metrics = evaluate_quantization(
        vectors=vectors,
        reconstructed=reconstructed,
        query_seed=query_seed,
        num_query_samples=num_query_samples,
    )
    payload = metrics.__dict__.copy()
    payload["bits"] = bits
    payload["grid_size"] = grid_size
    payload["rotation_seed"] = seed
    payload["query_seed"] = query_seed
    payload["num_query_samples"] = num_query_samples
    payload["inner_product_metric_type"] = "random_unit_query_proxy"
    return payload
