import astropy.io.fits as fits

import os
import glob
import astropy.io.fits as fits

def replace_tmp_into_target(dirpath='.'):
    os.chdir(dirpath)
    tmp_files = sorted(glob.glob('tmp_sky_subtracted_*.fits'))
    if not tmp_files:
        print("No tmp_sky_subtracted_*.fits files found.")
        return

    for tmp_path in tmp_files:
        target_path = tmp_path.replace('tmp_', '', 1)
        if not os.path.exists(target_path):
            print(f"Skipping {tmp_path}: target {target_path} does not exist.")
            continue

        try:
            with fits.open(tmp_path, memmap=False) as h_tmp:
                tmp_data = h_tmp[0].data
            if tmp_data is None:
                print(f"Skipping {tmp_path}: no data in hdul[0].")
                continue

            with fits.open(target_path, mode='update', memmap=False) as h_tgt:
                if len(h_tgt) <= 1:
                    print(f"Skipping {target_path}: no extension 1 to write into.")
                    continue

                ext1 = h_tgt[1]
                # If shapes and dtypes match, copy in-place; otherwise replace the HDU
                if ext1.data is not None and ext1.data.shape == tmp_data.shape and ext1.data.dtype == tmp_data.dtype:
                    ext1.data[...] = tmp_data
                else:
                    new_hdu = fits.ImageHDU(data=tmp_data, header=ext1.header)
                    h_tgt[1] = new_hdu

                h_tgt.flush()
                print(f"Updated {target_path} from {tmp_path}.")

        except Exception as e:
            print(f"Error processing {tmp_path} -> {target_path}: {e}")

if __name__ == '__main__':
    # run in the directory with the FITS files; change path if needed
    replace_tmp_into_target('.')