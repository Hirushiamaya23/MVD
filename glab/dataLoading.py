
#os / files / types
import os
import sys
from typing import Optional, Union
from pathlib import Path

# logger
from flwr.common import log
from logging import INFO, WARNING
from tqdm import tqdm

#data
import pickle as pkl
import pandas as pd
import random

# datasampling
import math
from .dataSampling import emptyDirectoryContents
from ..utils.systemUtils import (
    get_cpu_count,
    get_gpu_info,
    get_gpu_vram_info,
    get_volume_multiplier,
    get_system_memory_info,
)
from ..utils.metricUtils import RemapLabelsd
# pytorch
import torch
from torch.utils.data import DataLoader
# monai
from monai.data.iterable_dataset import ShuffleBuffer
from monai.data.grid_dataset import PatchDataset
from monai.data.dataset import PersistentDataset
from monai.data.utils import list_data_collate
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ConcatItemsd, ToTensord, EnsureTyped
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld
from monai.data.dataset import Dataset
from monai.utils.misc import set_determinism


def clientPartitioingFromIndexes(
    type: str,
    runID: int,
    nodeID: int,
    dataset_csv_path: Path,
    dataset_metadata: dict,
    partition_path: Path,
    data_indexes: tuple[int,int]|list[int,int],
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Load train and val dataframes based on row indexes for given client ID.
    If both train / val files are already found, they are loaded from disk,
    if either one is missing partitions are created and saved to disk for later rounds.
    """
    
    trainPath   = Path(partition_path, f'R{runID}_N{nodeID}_trainDF.csv')
    valPath     = Path(partition_path, f'R{runID}_N{nodeID}_valDF.csv')
    
    isTrain = checkIfPartitioningExists(partitionfile=trainPath)
    isVal   = checkIfPartitioningExists(partitionfile=valPath)
    
    if isTrain and isVal:
        
        if type == 'fit':
            
            trainDF = loadIndexedPartitions(type='train', partitionPath=trainPath)
            valDF   = loadIndexedPartitions(type='val', partitionPath=valPath)
            log(INFO, "[%s] Train/Val partition found, loading", nodeID)
            return trainDF, valDF
        
        elif type == 'val':
            
            valDF = loadIndexedPartitions(type='val', partitionPath=valPath)
            log(INFO, "[%s] Val partition found, loading", nodeID)
            return valDF
        
    else:
        
        try:
            
            train_idx = data_indexes[0]
            val_idx = data_indexes[1]
            
            log(INFO, "[%s] Loading train idx %s and val idx %s", nodeID, train_idx, val_idx)
    
            datasetDF       : pd.DataFrame = pd.read_csv(dataset_csv_path)
            prefix          : str = dataset_metadata['dataset_path']
            sequenceHeaders : list = [k.lower() for k in dataset_metadata['key_order']]

            # Apply actual sequence labels to the columns replacing "channel_"
            # Create mapping dictionary from channel columns to dataHeaders values
            column_mapping = dict(zip([col for col in datasetDF.columns], sequenceHeaders))
            datasetDF = datasetDF.rename(columns=column_mapping)

            
            #Aply prefix for the dataset directory apart from the SubjectID
            #sequenceHeaders.remove('SubjectID')  
            sequenceHeaders.remove('subjectid')  
            datasetDF[sequenceHeaders] = datasetDF[sequenceHeaders].astype(str).map(lambda x: os.path.join(prefix, x))
            
            trainDF = datasetDF.iloc[train_idx[0]:train_idx[1]].copy()
            valDF = datasetDF.iloc[val_idx[0]:val_idx[1]].copy()
            
            trainDF.to_csv(trainPath, index=False)
            valDF.to_csv(valPath, index=False)
            
            if type == 'fit':
                log(INFO, f'Loaded indexes for train: {len(trainDF)} val: {len(valDF)}')
                return trainDF, valDF
            
            elif type == 'val':
                log(INFO, f'Loaded indexes for val: {len(valDF)}')
                return valDF
            
        except Exception as e:
            
            raise e
                
def checkIfPartitioningExists(partitionfile: list[Path|str]) -> bool:
        
        return True if partitionfile.is_file() and partitionfile.exists() else False
    
def loadIndexedPartitions(type: str, partitionPath: Path) -> pd.DataFrame:
    
    match type:
        case 'train':
            traindf = pd.read_csv(partitionPath)
            return traindf
        
        case 'val':
            valdf   = pd.read_csv(partitionPath)
            return valdf

def getTrainingLoader(
    train_df: dict,
    image_sequences: list[str],
    label: str,
    patch_count: int,
    patch_size: tuple[int,int,int],
    volume_multiplier: int,
    numWorkers: int,
    cache_directory: str|Path,
    ) -> tuple[DataLoader, int]:
    
    log(INFO, "Constructing training dataloader..")
    
    image_sequences = [k.lower() for k in image_sequences]
    trainDict = train_df[image_sequences + [label]].to_dict(orient="records")

    cached_transforms = Compose([
        LoadImaged(keys=image_sequences + [label]),
        EnsureChannelFirstd(keys=image_sequences + [label]),
        RemapLabelsd(keys=[label]),
        ConcatItemsd(keys=image_sequences, name="image", dim=0),
    ],
    lazy=True)
    
    patch_sampler_fn = RandCropByPosNegLabeld(
            keys            = ["image", "label"],
            label_key       = label,
            spatial_size    = patch_size,
            pos             = 1.0,
            neg             = 0.0,
            num_samples     = patch_count,
            image_threshold = 0,
            lazy            = True,
        )
    
    per_patch_transform = Compose(
        [ToTensord(
            keys=["image", "label"])
        ], lazy=True)
    
    # base dataset that caches pre-processed volumes to local disk
    persistent_volume_ds = PersistentDataset(
        data        = trainDict,
        transform   = cached_transforms,
        cache_dir   = cache_directory
    )
    
    patch_ds = PatchDataset(
        data=persistent_volume_ds,
        patch_func=patch_sampler_fn,
        samples_per_image=patch_count,
        transform=per_patch_transform,
    )
    
    shuffled_patch_ds = ShuffleBuffer(patch_ds, buffer_size = int(patch_count * volume_multiplier))
    iterPerEpoch = len(patch_ds)

    # Standard DataLoader with patch dataset
    training_loader = DataLoader(
        shuffled_patch_ds,
        batch_size=1,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    )
    
    return training_loader, iterPerEpoch
    
    
def getValidationLoader(
    val_df: pd.DataFrame,
    image_sequences: list[str],
    label:str,
    numWorkers: int,
    cache_directory: str|Path
    ) -> DataLoader:
    
    log(INFO, "Constructing validation dataloader..")
    
    image_sequences = [k.lower() for k in image_sequences]
    valDict         = val_df[image_sequences + [label]].to_dict(orient="records")
    val_transforms = Compose([
        LoadImaged(keys=image_sequences + [label]),
        EnsureChannelFirstd(keys=image_sequences + [label]),
        RemapLabelsd(keys=[label]),
        ConcatItemsd(keys=image_sequences, name="image", dim=0),
        ToTensord(keys=["image", label]),
    ])
    val_ds = PersistentDataset(
        data=valDict,
        transform=val_transforms,
        cache_dir=cache_directory
    )
    validation_loader = DataLoader(
        val_ds, 
        batch_size=1,
        num_workers=numWorkers, 
        pin_memory=torch.cuda.is_available(),
    )

    return validation_loader

def initializeDataLoaders(
    clientID        : str | int,
    cache_path      : str | Path,
    state           : str,
    val_df          : pd.DataFrame,
    image_sequences : list[str],
    label           : str,
    patch_count     : int | None                    = None,
    patch_size      : tuple[int, int, int] | None   = None,
    train_df        : Optional[pd.DataFrame]        = None,
    seed            : Optional[int]                 = None,
    
) -> Union[tuple[DataLoader, DataLoader, int, int], tuple[DataLoader, int]]:
    
    # Determinism
    if seed is not None:
        set_determinism(seed=seed)

    # per client persistent dataset caches
    train_cache = cache_path / str(clientID) / 'train'
    val_cache = cache_path / str(clientID) / 'val'
    
    train_cache.mkdir(parents=True, exist_ok=True)
    val_cache.mkdir(parents=True, exist_ok=True)
    
    numWorkers = 0 # TODO: maybe set worker dynamically, currently tricky because simulation vs deployment differences.
    
    #volume_multiplier = get_volume_multiplier(
    #    patch_count     = patch_count,
    #    patch_size      = patch_size,
    #    volumes         = len(train_df),
    #    sequences       = len(image_sequences))
        
    
    # if training -> return (trainloader, valloader), if validation -> return valloader
    match state:

        case 'training':
            numSamples = len(train_df)
            
            training_loader, iterPerEpoch = getTrainingLoader(
                train_df            = train_df,
                image_sequences     = image_sequences,
                label               = label,
                patch_count         = patch_count,
                patch_size          = patch_size,
                numWorkers          = numWorkers,
                volume_multiplier   = 0.1,
                cache_directory     = train_cache
            )    
            
            validation_loader = getValidationLoader(
                val_df          = val_df,
                label           = label,
                image_sequences = image_sequences,
                numWorkers      = numWorkers,
                cache_directory = val_cache
            )
            
            log(
                INFO, 'Training DATALOADERS: train: (%s subjects, %s patches) val: (%s subjects)',
                numSamples, iterPerEpoch, len(validation_loader)
            )
            
            return training_loader, validation_loader, numSamples, len(validation_loader), iterPerEpoch
        
        case 'validation':
            
            numSamples = len(val_df)
            
            validation_loader = getValidationLoader(
                val_df          = val_df,
                label           = label,
                image_sequences = image_sequences,
                numWorkers      = numWorkers,
                cache_directory = val_cache
            )
            
            log(INFO, 'Validation DATALOADER: val: %s', numSamples)
            
            return validation_loader, numSamples
    
