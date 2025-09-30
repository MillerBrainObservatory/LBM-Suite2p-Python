import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def load_plane(folder: Path):
    ops = np.load(folder / "ops.npy", allow_pickle=True).item()
    stat = np.load(folder / "stat.npy", allow_pickle=True)
    F = np.load(folder / "F.npy")  # shape (n_cells, T)
    return ops, stat, F

def roi_mask(stat_entry, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[stat_entry["ypix"], stat_entry["xpix"]] = True
    return mask

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def find_overlaps(stat1, stat2, shape, thresh=0.7):
    overlaps = []
    for i, s1 in enumerate(stat1):
        m1 = roi_mask(s1, shape)
        for j, s2 in enumerate(stat2):
            m2 = roi_mask(s2, shape)
            score = iou(m1, m2)
            if score >= thresh:
                overlaps.append((i, j, score))
    return overlaps


def trace_corr(F1, F2, pairs, corr_thresh=0.6):
    matches = []
    for i, j, iou_score in pairs:
        r, _ = pearsonr(F1[i], F2[j])
        if r >= corr_thresh:
            matches.append((i, j, iou_score, r))
    return matches

def match_cells_across_planes(base_dir, iou_thresh=0.7, corr_thresh=0.6):
    base_dir = Path(base_dir)
    subdirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    planes = [load_plane(d) for d in subdirs]
    results = []

    for p1, (ops1, stat1, F1) in enumerate(planes):
        for p2, (ops2, stat2, F2) in enumerate(planes):
            if p2 <= p1:
                continue
            shape = ops1["Ly"], ops1["Lx"]
            overlaps = find_overlaps(stat1, stat2, shape, iou_thresh)
            matches = trace_corr(F1, F2, overlaps, corr_thresh)
            results.extend([(p1, p2, *m) for m in matches])
    return results

def plot_match(ops1, stat1, i, ops2, stat2, j):
    mean1 = ops1.get("meanImgE", ops1["meanImg"])
    mean2 = ops2.get("meanImgE", ops2["meanImg"])

    mask1 = roi_mask(stat1[i], mean1.shape)
    mask2 = roi_mask(stat2[j], mean2.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(mean1, cmap="gray")
    axes[0].contour(mask1, colors="red", linewidths=1)
    axes[0].set_title(f"Plane1 ROI {i}")

    axes[1].imshow(mean2, cmap="gray")
    axes[1].contour(mask2, colors="blue", linewidths=1)
    axes[1].set_title(f"Plane2 ROI {j}")

    plt.show()


def plot_combined(ops1, stat1, i, ops2, stat2, j):
    mean1 = ops1.get("meanImgE", ops1["meanImg"])
    mean2 = ops2.get("meanImgE", ops2["meanImg"])
    combined = (mean1 + mean2) / 2.0

    mask1 = roi_mask(stat1[i], combined.shape)
    mask2 = roi_mask(stat2[j], combined.shape)

    plt.figure(figsize=(6, 6))
    plt.imshow(combined, cmap="gray")
    plt.contour(mask1, colors="red", linewidths=1, label="plane1")
    plt.contour(mask2, colors="blue", linewidths=1, label="plane2")
    plt.title(f"ROI match: p1={i}, p2={j}")
    plt.show()


def plot_traces(F1, i, F2, j):
    plt.figure(figsize=(8, 3))
    plt.plot(F1[i], label="Plane1 ROI")
    plt.plot(F2[j], label="Plane2 ROI")
    plt.legend()
    plt.title(f"Trace correlation")
    plt.show()

if __name__ == "__main__":
    base_directory = r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite2p\z_registered\anatomical_1"
    matches = match_cells_across_planes(
        base_directory,
        iou_thresh=0.7,
        corr_thresh=0.7,
    )
    for match in matches:
        p1, p2, i, j, iou_score, corr = match
        print(f"Plane {p1} Cell {i} <-> Plane {p2} Cell {j} | IoU: {iou_score:.2f}, Corr: {corr:.2f}")

    planes = [load_plane(d) for d in sorted(Path(base_directory).iterdir()) if d.is_dir()]
    for match in matches[:5]:  # just the first few
        p1, p2, i, j, iou_score, corr = match
        ops1, stat1, _ = planes[p1]
        ops2, stat2, _ = planes[p2]
        plot_match(ops1, stat1, i, ops2, stat2, j)
