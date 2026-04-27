import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

bias_files = [
    "ALJa300382.fits",
    "ALJa300383.fits",
    "ALJa300384.fits",
    "ALJa300385.fits",
    "ALJa300386.fits",
    "ALJa300387.fits",
    "ALJa300388.fits",
    "ALJa300389.fits",
    "ALJa300390.fits",
    "ALJa300391.fits",
    "ALJa300392.fits",
]

flat_files = [
    "ALJa300054.fits",
    "ALJa300055.fits",
    "ALJa300056.fits",
]

science_file = "ALJa300217.fits"

print("bias files:", bias_files)
print("flat files:", flat_files)


def read_ext_data(fname, ext=1):
    with fits.open(fname, memmap=False) as hdul:
        return np.array(hdul[ext].data, dtype=float)


# build master bias (median)
bias_stack = [read_ext_data(f) for f in bias_files]
master_bias = np.median(np.stack(bias_stack, axis=0), axis=0)


# write master bias to a FITS file
master_bias_safe = np.where(np.isfinite(master_bias), master_bias, 0.0)
out_bias = "master_bias.fits"
hdu_bias = fits.PrimaryHDU(master_bias_safe.astype(np.float32))
hdu_bias.header["COMMENT"] = (
    "Master bias (median of bias frames), written by flat_plain_test.py"
)
hdu_bias.writeto(out_bias, overwrite=True)
print("Wrote master bias to", out_bias)

# visualize master bias
plt.figure()
plt.imshow(master_bias_safe, origin="lower", cmap="gray")
plt.colorbar()
plt.title("Master Bias")
plt.show()

# build normalized flats
normalized_flats = []
for f in flat_files:
    flat = read_ext_data(f)
    flat_bs = flat - master_bias  # bias-subtract
    # central region median for scaling (adjust indices if different detector)

    central = flat_bs
    scale = np.median(central)
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    flat_norm = flat_bs.astype(float) / float(scale)
    normalized_flats.append(flat_norm)

# combine normalized flats with median
master_flat = np.median(np.stack(normalized_flats, axis=0), axis=0)

# avoid divide-by-zero in flat correction
master_flat_safe = np.where(
    np.isfinite(master_flat) & (master_flat != 0), master_flat, 1.0
)
master_flat_safe[master_flat_safe < 0.8] = 1.0
master_flat_safe[master_flat_safe > 1.2] = 1.0

# write master flat to a FITS file
out_fname = "flat_test.fits"
hdu = fits.PrimaryHDU(master_flat_safe.astype(np.float32))
hdu.header["COMMENT"] = "Normalized master flat (safe), written by flat_plain_test.py"
hdu.writeto(out_fname, overwrite=True)
print("Wrote master flat to", out_fname)

# visualize master flat
plt.figure()
plt.imshow(master_flat_safe, vmin=0.8, vmax=1.2, origin="lower", cmap="gray")
plt.colorbar()
plt.title("Normalized Master Flat")
plt.show()

# apply to science
science = read_ext_data(science_file)
science_bs = science - master_bias  # bias-subtracted science
science_red = science_bs.astype(float) / master_flat_safe


# display raw and reduced images side-by-side with independent stretches
def pct_range(arr, pmin=30, pmax=90):
    a = np.asarray(arr).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = np.nanpercentile(finite, pmin)
    vmax = np.nanpercentile(finite, pmax)
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


vmin0, vmax0 = pct_range(science, 30, 95)
vmin1, vmax1 = pct_range(science_red, 30, 95)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im0 = axes[0].imshow(science, origin="lower", cmap="gray", vmin=vmin0, vmax=vmax0)
axes[0].set_title("Raw science")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(science_red, origin="lower", cmap="gray", vmin=vmin1, vmax=vmax1)
axes[1].set_title("Reduced science")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
