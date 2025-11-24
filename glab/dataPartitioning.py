
import os
from typing import Any, Dict, Union, Optional
from pathlib import Path
import pickle as pkl

#logging
from logging import WARNING, INFO, DEBUG
from flwr.common.logger import log

#data
import numpy as np
import pandas as pd
from dataclasses import field
from fractions import Fraction

# Monai / SaFlaas / torch
from.dataSampling import calculateTrainValSplit, loadDatasetWithRatio, distributeDataforClients, trainValSplitPerCollaborators, emptyDirectoryContents
from .dataConfiguration import FlaasConfigs

class DataPartitioning:
    
    __datasetConfig         : dict|Any  = field(init=True, repr=True, default={})
    __datasetMetadata       : dict|Any  = field(init=True, repr=True, default={})
    __num_partitions        : int       = field(init=True, repr=True, default=1)
    __min_client_samples    : int       = field(init=False, repr=True, default=1)   
    __random_seed           : int|None  = field(init=True, repr=True, default=None) 
    __debug                 : bool|Any  = field(init=True, repr=True, default=False)
    

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize the DataPartitioning class with dataset settings, number of partitions and dataset metadata.
        :param dataset_settings: Dictionary containing dataset settings such as ratio, dataset name, train val split, etc.
        :param num_partitions: Number of clients to partition the dataset for.
        :param dataset_metadata: Metadata of the dataset to be used for partitioning.
        """
        cls.__datasetConfig     = FlaasConfigs.get_dataset_cfg(asDict=True)
        cls.__num_partitions    = FlaasConfigs.get_flwr_app_cfg(key='num_partitions') 
        cls.__datasetMetadata   = FlaasConfigs.get_dataset_metadata(asDict=True)
        cls.__random_seed       = cls.__datasetConfig.get('random_seed', None)
        cls.__debug             = cls.__datasetConfig.get('debug', False)
        cls._checkRatioSettings()
    
    @classmethod
    def _checkRatioSettings(cls) -> None:
        """
        Check and calculate that given partitioning settings are possible.
        Takes ratio, num_partitions and checks that dataset can be partitioned correctly for the given number of clients.
        """
        dataRatio: float = float(cls.__datasetConfig['dataset_ratio'])
        trainValSplit: float = float(cls.__datasetConfig['train_val_split'])

        if not 0.0 < dataRatio <= 1.0:
            raise ValueError(
                f'Invalid dataset ratio: {dataRatio}. Ratio must in range 0.0 and 1.0, exclusive of 0.0 and inclusive of 1.0')

        if not 0.0 < trainValSplit < 1.0:
            raise ValueError(
                f'Invalid trainValSplit: {trainValSplit}. Split must be in range 0.0 and 1.0, exclusive of 0.0 and 1.0')
        
        # Calculate the smallest possible dataset size per client based on the trainValSplit and dataRatio
        # The denominator of fraction is the smallest integer N such that N * ratio and N * (1 - ratio) are both whole numbers.
        # This value is used in partitioning to ensure minimum samples per client for given trainValSplit.
        closestFraction = Fraction(trainValSplit).limit_denominator()
        smallestDataSizePerClient = closestFraction.denominator
        
        if (cls.__datasetMetadata['train']['size'] * dataRatio) < cls.__num_partitions * smallestDataSizePerClient:
            raise ValueError(
                f'Invalid partitioning settings: dataset size incompatible with num clients and trainValSplit.\n'
                f'{cls.__num_partitions * smallestDataSizePerClient} > {np.floor((cls.__datasetMetadata["train"]["size"] * dataRatio))}.\n'
                f'Ensure that the dataset size is sufficient for the number of partitions and minimum samples per client.\n'
            )
        
        cls.__min_client_samples = smallestDataSizePerClient
        log(INFO, f'Partitioning settings are valid. Minimum samples per client: {cls.__min_client_samples}')
        

    @classmethod
    def createClientDataPartitions(cls,
        num_partitions: int|None = None
        ) -> dict[int, dict[str, pd.DataFrame]]:
        """
        Create client data partitions based on the dataset metadata and partitioning settings.
        This method loads the selected (open-access) dataset, samples it according to the specified ratio
        and partitions it into the training and validation datasets for specified number of clients.
        :param num_partitions: The number of clients to partition the dataset for.
        :return: dictionary[int, dict[str, pd.DataFrame | Any]] where keys are client IDs and
        values are dictionaries containing 'train' and 'val' DataFrames and metadata for that client.
        """
        dataset_path = cls.__datasetMetadata.get('dataset_path', "")
        dataset_csv_file_path = Path(dataset_path, cls.__datasetMetadata['train']['csv'])
        num_partitions = num_partitions if num_partitions is not None else cls.__num_partitions
        
        if cls.__debug:
            debug_output = (
                f'\nPartitions by ID    : {[ele for ele in range(1, num_partitions+1)]}'
                f'\nDataset dir         : {dataset_path}, dir exists: {os.path.exists(dataset_path)}'
                f'\ncsv_path            : {dataset_csv_file_path} path exists: {os.path.exists(dataset_csv_file_path)}'
                f'\nRandom state        : {cls.__random_seed}'
            )
            log(INFO, debug_output)
        
        
        # Load the open access dataset with given arguments
        # Returns sampled and original datasets with columns renamed for correct poth prefixes for selected dataset
        
        originalDataset, sampledDataset = loadDatasetWithRatio(
            csv_path            = dataset_csv_file_path,
            dataPathPreFix      = dataset_path,
            sequenceHeaders     = cls.__datasetMetadata['key_order'],
            dataRatio           = cls.__datasetConfig['dataset_ratio'],
            shuffle             = cls.__datasetConfig['dataset_shuffle'],
            rngState            = cls.__random_seed,
            debug               = cls.__debug,
        )
        
        if cls.__debug:
            debug_output = (
                f'Original Dataset  : {originalDataset.columns} / {originalDataset.size}\n'
                f'Sampled Dataset   : {sampledDataset.columns} / {sampledDataset.size}\n'
            )        
            log(INFO, debug_output)
        
        # distribute the sampled dataset to the clients
        # The sampled dataset is partitioned into num_partitions
        sampledDatasetSize = len(sampledDataset)
        
        DISTRIBUTIONPARAMS = {
        'gaussian': {'mean': sampledDatasetSize / 3 , 'stdDev': 30},
        'dirichlet': {'alpha': 5.0}
        }
        
        dataDistributionPerClient = distributeDataforClients(
        datasetSize         = sampledDatasetSize,
        minSamples          = cls.__min_client_samples,
        numClients          = num_partitions,
        distType            = (cls.__datasetConfig['partition_type'], DISTRIBUTIONPARAMS[cls.__datasetConfig['partition_type']]),
        rngSeed             = cls.__datasetConfig['random_seed'],
        debug               = cls.__debug
        )
        
        
        trainValsplits = trainValSplitPerCollaborators(
            df                  = sampledDataset,
            dataDist            = dataDistributionPerClient,
            dataset_config      = cls.__datasetConfig,
            debug               = cls.__debug
        )
        
        return trainValsplits

    
    @classmethod
    def createClientPartitionIndexes(cls, num_partitions: int|None = None) -> dict[int, tuple[int,int]]:
        
        dataset_path = cls.__datasetMetadata.get('dataset_path', "")
        dataset_csv_file_path = Path(dataset_path, cls.__datasetMetadata['train']['csv'])
        num_partitions = num_partitions if num_partitions is not None else cls.__num_partitions
        
        if cls.__debug:
            debug_output = (
                f'\nPartitions          : {num_partitions}'
                f'\nDataset dir         : {dataset_path}, dir exists: {os.path.exists(dataset_path)}'
                f'\ncsv_path            : {dataset_csv_file_path} path exists: {os.path.exists(dataset_csv_file_path)}'
                f'\nRandom state        : {cls.__random_seed}'
            )
            log(INFO, debug_output)

        originalDataset, sampledDataset = loadDatasetWithRatio(
            csv_path            = dataset_csv_file_path,
            dataPathPreFix      = dataset_path,
            sequenceHeaders     = cls.__datasetMetadata['key_order'],
            dataRatio           = cls.__datasetConfig['dataset_ratio'],
            shuffle             = cls.__datasetConfig['dataset_shuffle'],
            rngState            = cls.__random_seed,
            debug               = cls.__debug,
        )
        
        if cls.__debug:
            debug_output = (
                f'Original Dataset  : {originalDataset.info}\n'
                f'Sampled Dataset   : {sampledDataset.info}\n'
            )        
            log(INFO, debug_output)
        
        sampledDatasetSize = len(sampledDataset)
        
        DISTRIBUTIONPARAMS = {
        'gaussian': {'mean': sampledDatasetSize / 3 , 'stdDev': 30},
        }
        
        dataDistributionPerClient = distributeDataforClients(
        datasetSize         = sampledDatasetSize,
        minSamples          = cls.__min_client_samples,
        numClients          = num_partitions,
        distType            = (cls.__datasetConfig['partition_type'], DISTRIBUTIONPARAMS[cls.__datasetConfig['partition_type']]),
        rngSeed             = cls.__datasetConfig['random_seed'],
        debug               = cls.__debug
        )
        
        splitsPerCollaborator: dict = {col: {'train': None, 'val': None} for col in dataDistributionPerClient}

        start_idx = 0

        for col, dataSize in dataDistributionPerClient.items():
            
            current_datasize = dataSize
            
            trainSize, valSize = calculateTrainValSplit(
                dataSize=current_datasize,
                splitRatio=cls.__datasetConfig['train_val_split']
                )
            
            # set training start idx, end idx
            train_end_idx = start_idx + trainSize
            splitsPerCollaborator[col]['train'] = (start_idx, train_end_idx)
            
            # set validation start idx, end idx
            val_start_index = train_end_idx
            val_end_index = val_start_index + valSize
            splitsPerCollaborator[col]['val'] = (val_start_index, val_end_index)
            
            # set current last index into global start
            start_idx = val_end_index
            
            if cls.__debug:
                log(INFO, f'Processed {col}:')
                log(INFO, f'Train: ({trainSize} rows) assigned range {splitsPerCollaborator[col]["train"]}')
                log(INFO, f'Val:   ({valSize} rows) assigned range {splitsPerCollaborator[col]["val"]}')
                log(INFO, f'Next global_start_index will be: {start_idx}')

        return splitsPerCollaborator

        

    @classmethod
    def savePartitioningToDisk(cls, partitioning: dict[int, dict[str, pd.DataFrame]], partitionPath: Path):
        """
        Save the partitioning data to disk in the specified directory to be loaded later by the clients.
        :param partitioning: Partitioning data from _createClientDataPartitions method.
        :param partitionPath: Path to the directory where the partitioning data will be saved.
        :return: None
        """
        # empty partitioning directory before saving new data
        emptyDirectoryContents(partitionPath)
        
        # Save the dictionary of train and validation dataframes into a pickle file to be loaded later by client
        for idx, splits in partitioning.items():
    
            client_csv_path = Path(partitionPath, f'client_{idx}_partition.pkl')
            
            try:
                with open(client_csv_path, 'wb') as f:
                    pkl.dump(splits, f)
                    if cls.__debug: log(INFO, f'Client {idx} data saved to {client_csv_path}')
            
            except Exception as e:
                log(WARNING, f'Error saving client partitioning to disk: {e}')
                raise e
