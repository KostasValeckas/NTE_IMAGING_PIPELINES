import numpy as np

def sort_data(input_path, sort_func):
    """
    Sorts the data into a dictionary with keys 'bias', 'dark', and 'flat'.
    Each key contains a list of file paths corresponding to that type of data.
    
    Parameters:
    input_path (str): The path to the directory containing the data files.
    
    Returns:
    dict: A dictionary with sorted file paths.
    """
    data_dict = {'bias': [], 'dark': [], 'flat': []}
    
    for filename in os.listdir(input_path):
        if filename.endswith('.fits'):
            if 'bias' in filename.lower():
                data_dict['bias'].append(os.path.join(input_path, filename))
            elif 'dark' in filename.lower():
                data_dict['dark'].append(os.path.join(input_path, filename))
            elif 'flat' in filename.lower():
                data_dict['flat'].append(os.path.join(input_path, filename))
    
    return data_dict