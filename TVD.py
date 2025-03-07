import numpy as np
import os
from skimage.restoration import denoise_tv_chambolle
# import matplotlib.pyplot as plt
import warnings

# Function to read files
def readFile(filepath):

    #Check for file
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    #Read the file
    with open(filepath, 'r') as file:
        #Read lines and insert into a 2d array of string (list of lists)
        data = [line.split() for line in file]

    #Convert to a 2d array of float64
    array2D = np.array(data, dtype=np.float64)
    return array2D

# Function to write files
def writeFile(filepath, data):
    with open(filepath, 'w') as file:
        for row in data:
            line = '\t'.join(map(str, row))
            file.write(line + '\n')

# Main function for denoising the dataset 1 column at a time 
def denoisedFiles(input_dir, output_dir, weightUsr, Y_column):
    #Check for an output dir and create it if it doesn't exist
    os.makedirs(output_dir, exist_ok = True)

    #List all files in the input directory
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):

            try:

                #print(f"Processing file: {filename}")

                # Read data from file
                data = readFile(filepath)

                #Import the data, extract the XY values and replace any -inf/ NaN artefacts w/ min value
                dataY = data[:, Y_ColumnNo]
                min_value = np.min(dataY[np.isfinite(dataY)])
                dataY[np.isneginf(dataY)] = min_value
                dataY[np.isnan(dataY)] = min_value
                dataY_2D = dataY[np.newaxis, :]

                #Perform denoising (TVD)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=RuntimeWarning)  #Convert warnings to errors for catch
                    try:
                        dataY_denoised_2D = denoise_tv_chambolle(dataY_2D, weight=weightUsr, eps=0.0000002)
                        dataY_denoised = dataY_denoised_2D.flatten()
                        data[:, Y_column] = dataY_denoised
                    except RuntimeWarning as rw:
                        print(f"Skipping file: {filename} due to problems with denoising")
                        continue
            
            
                #Replace dataY with dataY_denoised
                data[:, Y_column] = dataY_denoised
            
                #Define the output file path
                output_filepath = os.path.join(output_dir, filename)
            
                #Write the updated data to a new file
                writeFile(output_filepath, data)

            except Exception as e:
                print(f"Skipping file: {filename} due to error: {e}")
                continue  #Skip to the next file if there's a broader exception


#Take user inputs for directories and TVD weight
pathIN = input("Please enter the input file directory: ")
weightUsr = float(input("Please enter the regularisation weight: "))
Y_ColumnNo = int(input("Enter column number for the signal that is to be denoised (please use 0-indexing): "))
folderName = 'TVD_wt-' + str(weightUsr)
pathOUT = os.path.join(pathIN, folderName)
denoisedFiles(pathIN, pathOUT, weightUsr, Y_ColumnNo)

