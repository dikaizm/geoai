import numpy as np
import rasterio

def _build_inverse_mapping(mapping: dict, nodata: int = 255) -> dict:
    """Build inverse mapping (compact → original codes)."""
    inv_map = {v: k for k, v in mapping.items()}
    inv_map[nodata] = nodata
    return inv_map


def remap_raster(in_path: str, out_path: str, mapping: dict, nodata: int = 255):
    """Remap a raster file (CDL → compact labels)."""
    with rasterio.open(in_path) as src:
        mask = src.read(1)
        profile = src.profile

        dtype = np.uint8 if max(mapping.values()) < 256 else np.uint16
        remapped = np.full(mask.shape, nodata, dtype=dtype)  # start with nodata

        for old_val, new_val in mapping.items():
            remapped[mask == old_val] = new_val

        profile.update(dtype=dtype, nodata=nodata, compress="lzw")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(remapped, 1)

    print(f"Saved remapped raster to {out_path}")
    print("Unique values:", np.unique(remapped))


def remap_back_raster(in_path: str, out_path: str, mapping: dict, nodata: int = 255):
    """Remap a compact raster file back to original CDL codes."""
    inverse_mapping = _build_inverse_mapping(mapping, nodata=nodata)

    with rasterio.open(in_path) as src:
        mask = src.read(1).astype(np.int32)  # ensure integers
        profile = src.profile

        dtype = np.uint16  # CDL codes can exceed 255
        restored = np.full(mask.shape, nodata, dtype=dtype)  # start with nodata

        print("Unique compact values in prediction:", np.unique(mask))

        for new_val, old_val in inverse_mapping.items():
            restored[mask == new_val] = old_val

        profile.update(dtype=dtype, nodata=nodata, compress="lzw")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(restored, 1)

    print(f"Saved restored raster to {out_path}")
    print("Unique values:", np.unique(restored))


def get_unique_classes(raster_path: str) -> np.ndarray:
    """Get unique class values from a raster file, excluding nodata."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        unique_classes = np.unique(data)
    return unique_classes


def plot_class_distribution(
    raster_path: str,
    class_mapping: dict = None,
    ignore_index: int = 255,
    skip_background: bool = False,
    background_index: int = 0,
):
    """Compute and plot class frequency distribution from a raster file.

    Args:
        raster_path (str): Path to the label raster file.
        class_mapping (dict, optional): Mapping of ``{class_id: class_name}``.
            If None, uses class indices directly.
        ignore_index (int, optional): Nodata / ignore value. Defaults to 255.
        skip_background (bool, optional): If True, exclude the background class
            (defined by ``background_index`` or a mapped name like "background")
            from counts and percentage calculations. Defaults to False.
        background_index (int, optional): Numeric class id to treat as background
            when ``skip_background`` is True. Defaults to 0.

    Returns:
        dict: ``{class_id|class_name: (count, percentage)}`` (background removed
            if ``skip_background=True``). Percentages sum to ~100 after any
            exclusions.
    """

    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt

    with rasterio.open(raster_path) as src:
        data = src.read(1).flatten()

        # Remove ignore_index
        if ignore_index in data:
            data = data[data != ignore_index]

        # Optionally remove background class before counting so it doesn't
        # appear at all in frequencies/percentages.
        if skip_background and background_index in data:
            data = data[data != background_index]

        if data.size == 0:
            return {}

        frequencies = np.bincount(data, minlength=int(data.max()) + 1)

    if class_mapping is not None:
        labels = [class_mapping.get(i, str(i)) for i in range(len(frequencies))]
    else:
        labels = list(range(len(frequencies)))

    total = frequencies.sum()
    if total == 0:
        return {}
    freqs_dict = {
        labels[i]: (frequencies[i], (frequencies[i] / total) * 100)
        for i in range(len(frequencies)) if frequencies[i] > 0
    }

    # Sort by count descending
    sorted_items = sorted(freqs_dict.items(), key=lambda x: x[1][0], reverse=True)
    sorted_labels = [item[0] for item in sorted_items]
    counts = [item[1][0] for item in sorted_items]
    percentages = [item[1][1] for item in sorted_items]

    # Plot horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_labels, counts, color="skyblue")
    plt.xlabel("Pixel Count")
    plt.ylabel("Class")
    plt.title("Class Frequency Distribution")

    # Annotate bars with percentage
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                 f"{pct:.2f}%", va="center", ha="left")

    plt.gca().invert_yaxis()  # largest class at top
    plt.show()

    return freqs_dict