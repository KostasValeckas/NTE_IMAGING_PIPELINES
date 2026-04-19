import numpy as np 
from astropy.io import fits
import os
import shlex
import subprocess
import matplotlib.pyplot as plt
import json
from IO import open_fits_file
from matplotlib.patches import Ellipse


class Photometric_parser():

    def __init__(self, reduced_dir, logger, object_setup, show_plots=False):
        
        self.reduced_dir = reduced_dir
        self.logger = logger
        self.object_setup = object_setup
        self.show_plots = show_plots


    def run(self):
        # dummy for inheritance
        pass

    def determine_configurations(self):
        # TODO might not need this - keep for API
        pass

    # these are identical, maybe refactor or just leave out at all
    def run_sex(self, command):
        print('Executing:', ' '.join(shlex.quote(a) for a in command))
        subprocess.run(command, cwd=self.reduced_dir, check=True)

    def run_scamp(self, command):
        print('Executing:', ' '.join(shlex.quote(a) for a in command))
        subprocess.run(command, cwd=self.reduced_dir, check=True)

    def run_swarp(self, command):
        # Do this when WCS is solid
        pass

    def plot_apertures(self, masked_frame, good_objects, filename):
        """
        Helper method to plot SExtractor apertures to avoid repeated code
        """

        fig, ax = plt.subplots()
        vmin = np.percentile(masked_frame.compressed(), 5)
        vmax = np.percentile(masked_frame.compressed(), 95)
        im = ax.imshow(masked_frame, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)

        # Determine which radius field is available and build masks accordingly
        names = good_objects.names
        has_kron = 'KRON_RADIUS' in names
        has_fluxrad = 'FLUX_RADIUS' in names

        # ensure valid numeric entries for positions and at least one radius field
        mask_valid = np.isfinite(good_objects['XWIN_IMAGE']) & np.isfinite(good_objects['YWIN_IMAGE'])
        if has_kron:
            mask_valid &= np.isfinite(good_objects['KRON_RADIUS'])
        elif has_fluxrad:
            mask_valid &= np.isfinite(good_objects['FLUX_RADIUS'])

        xs = good_objects['XWIN_IMAGE'][mask_valid]
        ys = good_objects['YWIN_IMAGE'][mask_valid]

        # parse Kron factor from the SExtractor config (PHOT_AUTOPARAMS)
        kron_factor = 2.5  # default fallback
        try:
            with open(self.sex_config, 'r') as f:
                for line in f:
                    if 'PHOT_AUTOPARAMS' in line and not line.strip().startswith('#'):
                        rhs = line.split('PHOT_AUTOPARAMS', 1)[1]
                        rhs = rhs.split('#')[0]
                        rhs = rhs.replace(',', ' ')
                        parts = rhs.split()
                        for p in parts:
                            try:
                                kron_factor = float(p)
                                break
                            except ValueError:
                                continue
                        break
        except Exception:
            pass


        if has_kron:
            kron_vals = good_objects['KRON_RADIUS'][mask_valid].astype(float)
            # If we have A_IMAGE/B_IMAGE/THETA_IMAGE, draw ellipses scaled by kron
            if 'A_IMAGE' in names and 'B_IMAGE' in names:
                a_vals = good_objects['A_IMAGE'][mask_valid].astype(float)
                b_vals = good_objects['B_IMAGE'][mask_valid].astype(float)
                theta_vals = good_objects['THETA_IMAGE'][mask_valid].astype(float) if 'THETA_IMAGE' in names else np.zeros_like(a_vals)
                for x, y, k, a, b, theta in zip(xs, ys, kron_vals, a_vals, b_vals, theta_vals):
                    # A_IMAGE and B_IMAGE are semi-axis (pixels). Width/height for Ellipse are total extents.
                    width = 2.0 * k * kron_factor * a
                    height = 2.0 * k * kron_factor * b
                    ell = Ellipse((x, y), width=width, height=height, angle=theta, edgecolor='red', facecolor='none', linewidth=1)
                    ax.add_patch(ell)
            else:
                # fallback: draw circular apertures with radius = kron * kron_factor
                for x, y, k in zip(xs, ys, kron_vals):
                    circ = plt.Circle((x, y), k * kron_factor, edgecolor='red', facecolor='none', linewidth=1)
                    ax.add_patch(circ)
        elif has_fluxrad:
            rs = good_objects['FLUX_RADIUS'][mask_valid].astype(float)
            for x, y, r in zip(xs, ys, rs):
                circ = plt.Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=1)
                ax.add_patch(circ)
        else:
            self.logger.warning('No radius parameter available to plot apertures (KRON_RADIUS or FLUX_RADIUS).')
        ax.set_title('Detected Objects with FLAGS=0')
        ax.set_xlabel('XWIN_IMAGE')
        ax.set_ylabel('YWIN_IMAGE')

        save_path = os.path.join(self.reduced_dir, "apertures_" + filename.replace('.fits', '.png'))
        plt.savefig(save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close()




class ALFOSC_parser(Photometric_parser):

    def __init__(self, reduced_dir, logger, object_setup, show_plots=False):

        #TODO make a global directory for these
        self.sex_config = os.path.normpath(os.path.join(reduced_dir, 'default.sex'))
        self.sex_param = os.path.normpath(os.path.join(reduced_dir, 'run1.param'))

        #TODO expland with more filters as needed
        self.filter_query_mapping = {
            "u'": "tbd",
            "g'": "tbd",
            "r'": "tbd",
            "i'": "tbd",
            "z'": "tbd",
        }

        self.mask_x_fraction = 0.1
        self.mask_y_fraction = 0.1

        super().__init__(reduced_dir, logger, object_setup, show_plots=show_plots)


    def run(self):

        for obj_key, object_info in self.object_setup.items():
            
            for object_name in object_info.keys():
                
                info = object_info[object_name]

                filters = info.get('filter', [])

                # remove all entries that are not "Open"
                filters = [f for f in filters if not f.startswith("Open")]

                # ensure only one filter remains 
                if len(filters) == 0:
                    self.logger.error(f"No valid filters found for {object_name} in {obj_key}.")
                    continue

                if len(filters) > 1:
                    self.logger.error(f"Multiple filters found for {object_name} in {obj_key}, only single filters are supported. Skipping...")
                    continue

                self.logger.info(f"Processing {object_name} in {obj_key} with filter {filters[0]}")

                stripped_filter_name = filters[0].split('_', 1)[0].strip()

                if stripped_filter_name not in self.filter_query_mapping.keys():
                    self.logger.error(f"Filter {filters[0]} not implemented. Skipping...")
                    continue

                files = info.get('files', [])

                if files is None or len(files) == 0:
                    self.logger.error(f"No files found for {object_name} in {obj_key}. Skipping...")
                    continue

                stack = False

                if len(files) == 1:
                    self.logger.info("Only single exposure found, no stacking is needed")

                else:
                    self.logger.info(f"Multiple exposures found for {object_name} in {obj_key}, stacking will be performed")
                    stack = True

                # first step - SExtractor on all frames 

                for file in files:

                    file_path = os.path.join(self.reduced_dir, "reduced_science_" + file)

                    hdul = open_fits_file(file_path, self.logger)

                    data = hdul[1].data
                    bpm = hdul[2].data

                    y_size, x_size = bpm.shape
                    x_start = int(x_size * self.mask_x_fraction)
                    y_start = int(y_size * self.mask_y_fraction)
                    x_end = int(x_size * (1 - self.mask_x_fraction))
                    y_end = int(y_size * (1 - self.mask_y_fraction))

                    # we use this as a weight map for source extraction
                    inverted_bpm = np.where(bpm == 0, 1, 0)
                    inverted_bpm[0:y_start, :] = 0
                    inverted_bpm[y_end:, :] = 0
                    inverted_bpm[:, 0:x_start] = 0
                    inverted_bpm[x_end, :] = 0

                    # write to disc to comply so it can be used by SExtractor    

                    weights_filename = os.path.join(self.reduced_dir, file.replace('.fits', '_weights.fits'))

                    fits.writeto(weights_filename, inverted_bpm.astype(np.uint8), overwrite=True)

                    cmd = [
                         'sex',
                         '-c', self.sex_config,
                         file_path + '[1]',
                         '-PARAMETERS_NAME', self.sex_param,
                         '-WEIGHT_TYPE', 'MAP_WEIGHT',
                         '-WEIGHT_IMAGE', weights_filename,
                         '-CATALOG_NAME', file.replace('.fits', '.cat'),
                         '-VERBOSE_TYPE', 'FULL'
                     ]

                    self.run_sex(cmd)

                    table_path = os.path.join(self.reduced_dir, file.replace('.fits', '.cat'))

                    with fits.open(table_path) as hdul_table:
                        data_table = hdul_table[2].data


                    good_objects = data_table[data_table['FLAGS'] == 0]
    
                    masked_frame = np.ma.masked_where(bpm == 1, data)

                    self.plot_apertures(masked_frame, good_objects, file)


                    





class NOTCAM_parser(Photometric_parser):

    def __init__(self, reduced_dir, logger, object_setup, show_plots=False):
        super().__init__(reduced_dir, logger, object_setup, show_plots=show_plots)

    def run(self):
        
        print(self.object_setup)