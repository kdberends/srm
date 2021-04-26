import os
import shutil 
import re
from tqdm import tqdm
from pathlib import Path 
from netCDF4 import Dataset
from typing import List 
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def load_dhydro_ensemble(ensemble_path:str) -> np.ndarray:
    ensemble_path = Path(ensemble_path)
    # load all data

    waterlevel = []
    for subdir in tqdm(ensemble_path.iterdir()):
        ncfile = subdir.joinpath('dflow1d/output/gridpoints.nc')
        with Dataset(ncfile) as df:
            waterlevel.append(df.variables.get('water_level')[-1,0].compressed()[0])
    
    return np.array(waterlevel)

def create_dhydro_ensemble(design_matrix:dict, case_path:str, overwrite:bool=False):
    """
    This function creates an ensemble for D-Hydro models

    Arguments:
        design_matrix: dictionary containing values for the various parameters
        case_path: path to the location of the case. Should have a folder named 'template'

    Returns:

    """
    # Set paths
    case_path = Path(case_path)
    output_path = case_path.joinpath('ensemble')
    template_path = case_path.joinpath('template')
    file_roughness = template_path.joinpath('roughness-Main.ini')
    file_boundcond = template_path.joinpath('BoundaryConditions.bc')

    # Template files have placeholders like @01
    pattern = re.compile(r'@\d+')
    skipped_iterations = 0
    total_iterations = len(design_matrix['manning'])
    for i in tqdm(range(total_iterations)):
        iter_path = output_path.joinpath(f"{i:03d}")
        iter_file_roughness = iter_path.joinpath('dflow1d').joinpath('roughness-Main.ini')
        iter_file_boundcond = iter_path.joinpath('dflow1d').joinpath('BoundaryConditions.bc')

        # make new 
        if overwrite:
            try:
                shutil.rmtree(iter_path)
            except FileNotFoundError:
                pass

        # Copy template model to output
        try:
            shutil.copytree(template_path.joinpath('model'), iter_path)
        except FileExistsError:
            skipped_iterations += 1
            continue
        
        # Modify roughness files
        with open(file_roughness, 'r') as fread, \
            open(iter_file_roughness, 'w') as fwrite:
            for line in fread:
                for match in re.findall(pattern, line):
                    parameterid = int(match.split('@')[1]) 
                    line = re.sub('@{:02d}'.format(parameterid), 
                                '{:.5f}'.format(design_matrix['manning'][i]),
                                line)
                fwrite.write(line)
        
        # Modify boundary condition file
        with open(file_boundcond, 'r') as fread, \
            open(iter_file_boundcond, 'w') as fwrite:
            for line in fread:
                for match in re.findall(pattern, line):
                    parameterid = int(match.split('@')[1]) 
                    if parameterid == 1:
                        # discharge
                        line = re.sub('@{:02d}'.format(parameterid), 
                                    '{:.5f}'.format(design_matrix['discharge'][i]),
                                line)
                    elif parameterid == 2:
                        # downstream water level
                        line = re.sub('@{:02d}'.format(parameterid), 
                                    '{:.5f}'.format(design_matrix['h_downstream'][i]),
                                line)
                        
                fwrite.write(line)

    print (f"Skipped {skipped_iterations}/{total_iterations} iterations because they already existed")
    return output_path

def get_test_sets(design_matrix, model_output, split:float=0.8):
    """
    outputs a validation / calibration set based on a random split
    """

    # Read arguments
    manning = np.array(design_matrix['manning'])
    discharge = np.array(design_matrix['discharge'])
    h_downstream = np.array(design_matrix['h_downstream'])
    samplesize = len(design_matrix['manning'])

    # The mask divides between the calibration and cross-validation set
    # First we populathe the mask with the approate number of True/Falses
    mask = [True] * int(samplesize * split)
    validation_set = [False] * (samplesize - len(mask))

    # Next randomly shuffle
    mask.extend(validation_set)
    mask = np.array(mask)
    np.random.shuffle(mask)  # this replaces mask..

    # create the sets
    xcal = np.array([manning[mask], discharge[mask], h_downstream[mask]]).T
    ycal = model_output[mask]
    
    xval = np.array([manning[~mask], discharge[~mask], h_downstream[~mask]]).T
    yval = model_output[~mask]
    
    return (xcal, ycal), (xval, yval)


def fit_response_surface(degree:int=2, cal:list=[]):
    """
    Fits a polynomial response surfce based on the calibration set
    Returns:
        fitted response surface model
    """
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])

    model = model.fit(cal[0], cal[1])
    model.named_steps['linear'].coef_
    return model
 
def surrogate_gof(model, data):
    """ 
    Arguments:
        model: surrogaate model
        data: set (validation/test/calibratio)

    Returns:
        bias, std, rmse

    """
    bias= np.mean(data[1] - model.predict(data[0]))
    std = np.std(data[1] - model.predict(data[0]))
    rmse = np.sqrt(np.sum((data[1] - model.predict(data[0]))**2/len(data[1])))
    return bias, std, rmse

