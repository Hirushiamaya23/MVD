
"""Data preprocessing and partitioning for federated learning simulations"""

# Importing required libraries
import pandas as pd
import numpy as np
import shutil
import os

from typing import Optional, Union, List
from pathlib import Path, PurePath

from numpy.random import normal, dirichlet

from sklearn.model_selection import train_test_split
from flwr.common.logger import log
from logging import INFO


def emptyDirectoryContents(folder_path):
    """
    Empties the specified folder by deleting all its contents.

    Parameters:
    folder_path (str): The path to the folder to be emptied.
    """
    
    if not os.path.exists(folder_path):
        log(INFO,f'Folder {folder_path} does not exist')
        return
    
    if os.listdir(folder_path) == []:
        log(INFO,f'Folder {folder_path} is already empty')
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            log(INFO,f'File {filename} has been removed from {folder_path}')
            
        except Exception as e:
            log(INFO,f'Failed to delete {file_path}. Reason: {e}')
            
def loadDatasetWithRatio(
    csv_path:Union[PurePath, Path],
    dataPathPreFix:Union[PurePath,Path],
    sequenceHeaders:List[str],
    dataRatio:float,
    rngState:Optional[int],
    shuffle:bool=True,
    debug:bool=False,
    ):
    """
    Load and filter the dataset.
    Parameters:
        csv_path (PurePath): Path to the CSV file containing the dataset.
        dataPathPreFix (Path): Prefix for the dataset directory.
        sequenceHeaders (List[str]): List of headers for the sequences.
        dataRatio (float): Ratio of data to sample from the dataset. 0.01 to 1.0 (0% to 100%.
        rngState (Optional[int]): Random state for reproducibility.
        numDigits (int): Number of digits for rounding.
        shuffle (bool): Whether to shuffle the dataset before sampling.
        debug (bool): Whether to print debug information.
    """
    #Datapath Loading
    originalData = pd.read_csv(csv_path)

    # Apply actual sequence labels to the columns replacing "channel_"
    # Create mapping dictionary from channel columns to dataHeaders values
    column_mapping = dict(zip([col for col in originalData.columns], sequenceHeaders))
    originalData = originalData.rename(columns=column_mapping)
    
    #Aply prefix for the dataset directory apart from the SubjectID
    sequenceHeaders.remove('SubjectID')  
    originalData[sequenceHeaders] = originalData[sequenceHeaders].astype(str).map(lambda x: os.path.join(dataPathPreFix, x))

    # calculate the number of rows to sample based on the data ratio
    if dataRatio > 0:
        dataRatioFromDF = round(len(originalData) * dataRatio)
    else: 
        raise ValueError('dataRatio must be greater than 0.0, given value: {dataRatio}')
    
    # shuffle data if set True and take a sample size of n from the dataset
    # else take a sample of n from the dataset
    if shuffle:
        shuffledDF = originalData.sample(frac=1, random_state=rngState)
        sampledDF = shuffledDF.sample(n=dataRatioFromDF, random_state=rngState)
    else:
        sampledDF = originalData.sample(n=dataRatioFromDF, random_state=rngState)
    
    #sanity check
    if debug:
        sanityOutput = ( 
            f'original data size: {len(originalData)}\n'
            f'sampled data size: {len(sampledDF)} with {dataRatio * 100}% of total dataset\n'
            f'Sanity check % dataRatio x 100 / dataset size: {round(dataRatioFromDF * 100 / len(sampledDF) / 100, 2)}'
            )
        log(INFO,sanityOutput)
    
    return originalData, sampledDF
 

def gaussianDistGenerator(mean: float, stdDev: float, numCols: int, rngState: int| None = None):
    """
    Generate a Gaussian distribution.

    Parameters:
        mean (float): The mean of the Gaussian distribution.
        stdDev (float): The standard deviation of the Gaussian distribution.
        numCols (int): The number of values to generate.
        rngState (int | None, optional): A random state for reproducibility. Defaults to None.

    Returns:
        np.ndarray: An array of generated values following the Gaussian distribution.
    """
    return normal(mean, stdDev, numCols)


def dirichletDistGenerator(numCols, alpha=1.0, rngState: int | None = None):
    """
    Generate a Dirichlet distribution.

    Parameters:
        numCols (int): The number of values to generate.
        alpha (float, optional): The concentration parameter of the Dirichlet distribution. Defaults to 1.0.
        rngState (int | None, optional): A random state for reproducibility. Defaults to None.

    Returns:
        np.ndarray: An array of generated values following the Dirichlet distribution.
    """
    alpha = np.full(numCols, alpha)
    return dirichlet(alpha)
    
    

def normalizeDistribution(distribution: np.ndarray) -> np.ndarray:
    """
    Normalize a distribution to sum to 1.

    Parameters:
        distribution (numpy.ndarray): The input distribution array.
    Returns:
        numpy.ndarray: The normalized distribution where the sum of all elements is 1.
    """
    normalizedDist = distribution / distribution.sum()
    return normalizedDist

  

def distributeDataforClients(
    datasetSize: int,
    minSamples: int,
    numClients: int,
    distType: tuple[str, dict],
    rngSeed: Optional[int],
    debug: bool = True
    ) -> dict[int, int]:
    """
    Distribute data samples to collaborators based on the distribution type.
    Distribution types (distType):   
        gaussian:       Distribute data based on a normal distribution sample
        stratified:     Distribute data based on the stratified distribution with random overlapping
    """
    # Set random state for reproducibility or None for random
    rngState = rngSeed
    if debug: log(INFO,f'Random state: {rngState}')

    # Remove the minimum number of samples per client to satisfy trainValSplit from the dataset size.
    dataSize = datasetSize - (minSamples * numClients)

    if distType[0] == 'gaussian':
        
        if debug: log(INFO,f'Using Gaussian distribution with {numClients} clients')
        
        mean = distType[1]['mean'] / numClients
        stdDev = distType[1]['stdDev']
        if debug: log(INFO,f'Gaussian Mean: {mean}, StdDev: {stdDev}')

        colDistributions = gaussianDistGenerator(mean=mean,
                                                 stdDev=stdDev,
                                                 numCols=numClients,
                                                 rngState=rngState)
        colDistributions = np.abs(colDistributions)
        
        # Ensure no negative values in distribution
        colDistributions[colDistributions < 0] = 0

        # Normalize the distribution
        colDistributions = normalizeDistribution(colDistributions)

        if debug: log(INFO,f'Collab distributions:\n{colDistributions}')
        if debug: log(INFO,f'Collab distribution sum: {round(np.sum(colDistributions))}')
        

    elif distType[0] == 'dirichlet':
        
        alpha = distType[1]['alpha']
        if debug: log(INFO,f'Dirichlet Alpha: {alpha}')

        colDistributions = dirichletDistGenerator(numCols=numClients,
                                                  alpha=alpha,
                                                  rngState=rngState)
        
        # Normalize the distribution
        colDistributions = normalizeDistribution(colDistributions)

        if debug: log(INFO,f'Collab distributions:\n{colDistributions}')
        if debug: log(INFO,f'Collab distribution sum: {round(np.sum(colDistributions))}')

    # Distribute data rows to collaborators based on the distribution generated
    rowsPerCollab: np.ndarray = (dataSize * colDistributions).astype(np.int64) # type: ignore

    if debug: log(INFO,f'Collab distributions before max val addition:{rowsPerCollab}\n')

    # Adjust the last collaborator's rows to match total data size
    rowsPerCollab[-1] += (dataSize - np.sum(rowsPerCollab))

    # Ensure no collaborator has negative rows assigned
    rowsPerCollab[rowsPerCollab < 0] = 0

    if debug: log(INFO,f'Collab distributions before minSamples:{rowsPerCollab}\n')

    # Add the minimum number of samples to each collaborator
    rowsPerCollab += minSamples
    
    if debug: log(INFO,f'Collab distributions after minSamples:{rowsPerCollab}\n')
    
    colDistributions = {(client_id+1):int(rowsPerCollab[client_id]) for client_id in range(numClients)}
    

    
    return colDistributions



def calculateTrainValSplit(dataSize: int, splitRatio: float):
    """
    Calculate datasplit based on dataset size and train/val split ratio. Checks 
    """
    trainSize = round(dataSize * splitRatio)
    
    if trainSize < 1:
        raise ValueError(f'Datasize is 0, cant split into train/val')
    
    valSize = dataSize - trainSize
    return trainSize, valSize

def trainValSplitPerCollaborators(
    df: pd.DataFrame,
    dataDist: dict[int,int],
    dataset_config: dict,
    debug: bool = False
    ) -> dict[int, dict[str, pd.DataFrame]]:
    
    """
    TODO: Add docstring
    """

    dataQueue: pd.DataFrame = df.copy()
    dataDistributions: dict[int,int] = dataDist
    splitsPerCollaborator: dict = {col: {'train': None, 'val': None} for col in dataDistributions}
    
    for col, dataSize in dataDistributions.items():
        
        trainSize, valSize = calculateTrainValSplit(
            dataSize=dataSize,
            splitRatio=dataset_config['train_val_split']
        )
        
        if debug: 
            log(INFO,f'col: {col:<15} datasize: {dataSize:<6} train: {trainSize:<6} val: {valSize:<6} sum check: {trainSize + valSize == dataSize}')

        
        #get the first n rows from the data
        tempData = dataQueue.iloc[:dataSize]
        #drop used rows from the data
        dataQueue = dataQueue.iloc[dataSize:]
        #split the data into train and validation sets
        trainData, valData = train_test_split(
            tempData, 
            train_size=trainSize,
            test_size=valSize,
            shuffle=dataset_config['dataset_shuffle']
        )
        #Store the split data info into the collaborator dictionary
        splitsPerCollaborator[col]['train'] = trainData
        splitsPerCollaborator[col]['val'] = valData

    return splitsPerCollaborator