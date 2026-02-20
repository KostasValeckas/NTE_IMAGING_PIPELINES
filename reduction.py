from  instruments import Instrument
import os

class ReductionPipeline:
    def __init__(self, instrument: Instrument, raw_data_path):
        self.instrument = instrument
        self.raw_data_path = raw_data_path

    def sort_data(self):
        
        all_files = os.listdir(self.raw_data_path)

        bias_files = []
        dark_files = []
        flat_files = []
        science_files = []

        for filename in all_files:
            type = self.instrument.match_image_type(filename)
            if type == "BIAS":
                bias_files.append(os.path.join(self.raw_data_path, filename))
            elif type == "DARK":
                dark_files.append(os.path.join(self.raw_data_path, filename))
            elif type == "FLAT":
                flat_files.append(os.path.join(self.raw_data_path, filename))
            elif type == "SCIENCE":
                science_files.append(os.path.join(self.raw_data_path, filename))

        return bias_files, dark_files, flat_files, science_files
    

    def run_pipeline(self):
        bias_files, dark_files, flat_files, science_files = self.sort_data()
        # Further processing can be done here

        