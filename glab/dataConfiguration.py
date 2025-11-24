# Parsing / file handling
import os
import tomllib
import yaml
import json

# Typing / Pathing / Printing
from pprint import pp
from pathlib import Path, Path
from typing import Optional, Any, Union


#logging
from logging import WARNING, INFO, DEBUG
from flwr.common.logger import log

#data
from dataclasses import dataclass, field, asdict
from fractions import Fraction

@dataclass(frozen=True)
class FlwrAppConfig:
    """
    Configuration class for the Flower application.

    This dataclass holds the configuration parameters for the Initial Server App config.
    These settings are frozen to ensure immutability after initialization.

    Attributes:
        num_server_rounds (int): Total number of rounds for federated learning.
        num_partitions (int): Number of total clients.
        client_selection_method (str): Method used for selecting clients in each round.
        aggregation_method (str): Method used for aggregating client updates.
        random_seed (Optional[int]): Seed for random number generation, if any.
        debug (bool): Flag to enable or disable extra log messages.
    """
    num_server_rounds       : int   = field(init=True, repr=True, default=3)
    num_partitions          : int   = field(init=True, repr=True, default=1)
    cpu_per_client          : int   = field(init=True, repr=True, default=2)
    client_selection_method : str   = field(init=True, repr=True, default='default')
    aggregation_method      : str   = field(init=True, repr=True, default='default')
    random_seed             : Optional[int] = field(init=True, repr=True, default=None)
    debug                   : bool  = field(init=True, repr=True, default=False)

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for dataset partitioning.
    This dataclass holds all the necessary parameters to configure how a dataset
    is loaded, partitioned, and prepared for training and validation for each client.
    Attributes:
        model_name: Name of the model for which the dataset is being prepared.
        dataset_root: Root directory where the dataset is located.
        dataset_name: Specific name of the dataset to be used.
        dataset_shuffle: Boolean indicating whether to shuffle the dataset before partitioning.
        dataset_ratio: Float representing the ratio of the dataset to be used (e.g., 0.8 for 80%).
        train_val_split: Float representing the ratio for splitting the dataset into training and validation sets
                        (e.g., 0.2 means 20% for validation, 80% for training from the `dataset_ratio` portion).
        partition_type: String indicating the type of partitioning to be performed (e.g., 'random', 'stratified').
        random_seed: Optional integer seed for random operations to ensure reproducibility.
    """
    model_name      : str           = field(init=True, repr=True, default_factory=str)
    dataset_root    : str           = field(init=True, repr=True, default_factory=str)
    dataset_name    : str           = field(init=True, repr=True, default_factory=str)
    dataset_shuffle : bool          = field(init=True, repr=True, default_factory=bool)         
    dataset_ratio   : float         = field(init=True, repr=True, default_factory=float) 
    train_val_split : float         = field(init=True, repr=True, default_factory=float) 
    partition_type  : str           = field(init=True, repr=True, default_factory=str)         
    random_seed     : Optional[int] = field(init=True, repr=True, default=None)
    debug           : bool          = field(init=True, repr=True, default=False)                

@dataclass(frozen=True)
class DatasetMetadata:
    """
    Metadata for a selected open-source dataset.
    This dataclass holds the metadata information for a dataset, including its name,
    training and validation paths, dataset sizes, sequences and ground truth labels.
    """
    dataset_name    : str           = field(init=True, repr=True, default_factory=str)
    dataset_path    : str           = field(init=True, repr=True, default_factory=str)
    train           : dict          = field(init=True, repr=True, default_factory=dict)
    test            : dict          = field(init=True, repr=True, default_factory=dict)
    sequence        : list          = field(init=True, repr=True, default_factory=list)
    key_order       : list          = field(init=True, repr=True, default_factory=list)
    label           : str           = field(init=True, repr=True, default_factory=str)
    classes         : dict          = field(init=True, repr=True, default_factory=dict[str, int])


@dataclass(frozen=False)
class ModelConfig:
    """
    Configuration class for defining model and its init / input arguments.

    This dataclass stores settings related to a machine learning model,
    including its function/filename, name, computation device, and
    any specific arguments required for its initialization or operation.

    Attributes:
        model_fn (str): The function name or path to the model script/file.
                        Defaults to an empty string.
        model_name (str): The designated name for the model.
                        Defaults to an empty string.
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').
                    Defaults to 'cuda'.
        model_args (dict): A dictionary of arguments to be passed to the model.
                        Defaults to an empty dictionary.
    """
    model_fn        : str           = field(init=True, repr=True, default_factory=str)
    model_name      : str           = field(init=True, repr=True, default_factory=str)
    device          : str           = field(init=True, repr=True, default='cuda')
    model_args      : dict          = field(init=True, repr=True, default_factory=dict)

@dataclass(frozen=False)
class TrainingConfig:
    """
    Configuration class for training a model.
    This class holds various hyperparameters and settings used during the model training process.
    It utilizes the `field` function, presumably from a library like `dataclasses` or a similar
    configuration management tool, to define its attributes with initial values, string
    representations, and default values or factories.
    Attributes:
        batch_size (int): The number of samples processed before the model is updated.
                          Defaults to 1.
        epochs (int): The total number of passes through the entire training dataset.
                      Defaults to 1.
        lr (float): The learning rate for the optimizer.
                    Defaults to 0.001.
        loss_fn (str): The identifier or name of the loss function to be used for training.
                       Defaults to an empty string, implying it needs to be specified.
        optimizer (str): The identifier or name of the optimization algorithm to be used.
                         Defaults to an empty string, implying it needs to be specified.
        metrics (Union[list, str]): A list of metric names (strings) or a single metric name
                                    (string) to be evaluated during training and validation.
                                    Defaults to 'dice'.
    """
    amp                 : bool          = field(init=True, repr=True, default=True)
    batch_size          : int           = field(init=True, repr=True, default=1)
    epochs              : int           = field(init=True, repr=True, default=1)
    lr                  : float         = field(init=True, repr=True, default=0.001)
    loss_fn             : str           = field(init=True, repr=True, default_factory=str)
    optimizer           : str           = field(init=True, repr=True, default_factory=str)
    device              : str           = field(init=True, repr=True, default='cuda')
    overrides           : dict          = field(init=True, repr=True, default_factory=dict)
    
@dataclass(frozen=False)
class PreprocessConfig:
    """
    Configuration class for preprocessing parameters.
    This class holds settings related to how data, typically images or volumetric data,
    is divided into smaller patches for processing.
    Attributes:
        patch_size  (tuple) :   Defines the dimensions of each patch (e.g., (height, width, depth) 
                                for 3D patches. Defaults to (64,64,64)
        patch_count (int)   :   Specifies the total number of patches to be extracted or 
                                processed. Defaults to 40.
        ram_limit   (float) :   Percentage of total available RAM can be allocated during data preprocessing.
        reserve_cpu (int)   :   How many cpu's are reserved from total available that are not allocated to preprocess workers.
    """
    patch_size      : tuple         = field(init=True, repr=True, default=(64, 64, 64))
    patch_count     : int           = field(init=True, repr=True, default=40)
    ram_limit       : float         = field(init=True, repr=True, default=0.25)
    reserve_cpu     : float         = field(init=True, repr=True, default=0.75)


class FlaasConfigs:
        
    workspaceRoot         : Path
    pyprojectPath         : Path
    federationSettingPath : Path
    partitionsPath        : Path
    monai_cache_dir       : Path
    
    __flwr_app_cfg          : FlwrAppConfig
    __dataset_cfg           : DatasetConfig
    __dataset_metadata      : DatasetMetadata
    __model_cfg             : ModelConfig       
    __training_cfg          : TrainingConfig 
    __preprocess_cfg        : PreprocessConfig
    
    
    @classmethod
    def __repr__(cls) -> str:
        """
        Return a string representation of the FlaasConfigs class.
        This method provides a summary of the configuration settings.
        :return: A string representation of the FlaasConfigs class.
        """
        return (
            f'FlaasConfigs\n'
            f'workspaceRoot         = {cls.workspaceRoot}\n'
            f'pyprojectPath         = {cls.pyprojectPath}\n'
            f'federationSettingPath = {cls.federationSettingPath}\n'
            f'partitionsPath        = {cls.partitionsPath})\n'
            f'FlwrAppConfig         = {bool(cls.__flwr_app_cfg)}\n'
            f'DatasetConfig         = {bool(cls.__dataset_cfg)}\n'
            f'DatasetMetadata       = {bool(cls.__dataset_metadata)}\n'
            f'ModelConfig           = {bool(cls.__model_cfg)}\n'
            f'TrainingConfig        = {bool(cls.__training_cfg)}\n'
            f'PreprocessConfig      = {bool(cls.__preprocess_cfg)}\n'
        )
    
    @classmethod
    def initialize(cls, workspaceRoot: Path) -> None:
        """Initialize the FlaasConfigs class variables.
        This method sets the necessary path variables based on the provided
        workspace root, loads the Flower application configuration from
        the pyproject.toml file, and loads the federation configuration
        from the federationConfig.yml file.

        Args:
            workspaceRoot: The root directory of the workspace.
        Returns:
            None
        """
        
        cls.__setPathvars(workspaceRoot)
        cls.__loadFlowerConfig()
        cls.__loadFederationConfig()
        cls.__loadDatasetMetadata()
        print(cls.__repr__())
    
    
    @classmethod
    def __setPathvars(cls, workspaceRoot: Path) -> None:
        """
        Set the path variables for the configuration files.
        This method sets the paths for the workspace Root, pyproject.toml and federation settings YAML file.
        :param workspaceRoot: The root directory of the workspace.
        """
        cls.workspaceRoot          = workspaceRoot
        cls.pyprojectPath          = Path(workspaceRoot, 'pyproject.toml')
        cls.federationSettingPath  = Path(workspaceRoot, 'MVD', 'configs', 'federationConfig.yml')
        cls.partitionsPath         = Path(workspaceRoot, 'MVD', 'data', 'partitions')
        cls.monai_cache_dir        = Path(workspaceRoot, 'monai_cache')
        
        

        if not cls.partitionsPath.exists():
            cls.partitionsPath.mkdir(parents=True, exist_ok=True)
            
        if not cls.monai_cache_dir.exists(): 
            cls.monai_cache_dir.mkdir(parents=True, exist_ok=True)

        if not all([
            cls.workspaceRoot.exists(),
            cls.pyprojectPath.exists(),
            cls.federationSettingPath.exists(),
            cls.partitionsPath.exists(),
            cls.monai_cache_dir.exists()
        ]):
            raise FileExistsError(
                f'Federation settings file or partitions directory not found in {workspaceRoot}'
            )
        
        
    @classmethod
    def __loadFlowerConfig(cls) -> None:
        
        try:
            with open(cls.pyprojectPath, 'rb') as f:
                tomlData = tomllib.load(f) 
            
            flwrConfig: dict = tomlData.get('tool', {}).get('flwr', {}).get('app', {}).get('config', {})
            default_sim: str = tomlData.get('tool', {}).get('flwr', {}).get('federations', {}).get('default', {})
            flwrConfig.update(
                {
                    'num_partitions': tomlData.get(
                        'tool', {}).get('flwr', {}).get('federations', {}).get(default_sim, {}).get('options', {}).get('num-supernodes', {}),
                    'cpu_per_client': tomlData.get(
                        'tool', {}).get('flwr', {}).get('federations', {}).get(default_sim, {}).get('options', {}).get('num-cpus', {})
                }
            )
            cls.__flwr_app_cfg = FlwrAppConfig(
                **flwrConfig
            )
            
            
        except Exception as e:
            log(WARNING, f"Error loading pyproject config from {cls.pyprojectPath}:\n{e}")
            raise e
        
        
    @classmethod
    def __loadFederationConfig(cls):
        
        ymlPath: Path = cls.federationSettingPath
        if not ymlPath.exists():
            raise FileExistsError(ymlPath) 
        try:
            with open(ymlPath, 'rb') as f:
                ymlData = yaml.safe_load(f)
            
            cls.__dataset_cfg = DatasetConfig(
                **ymlData.get('DatasetConfig', {})    
            )
            
            model_name  = ymlData.get('DatasetConfig', {}).get('model_name', {})
            model_args  = ymlData.get('ModelConfig', {}).get(model_name, {}).get('args', {})
            model_fn    = model_args.get('model_fn', {})
            
            cls.__model_cfg = ModelConfig(
                model_name  = model_name,
                model_fn    = model_fn,
                model_args  = model_args
            )
            
            cls.__training_cfg = TrainingConfig(
                **ymlData.get('TrainingConfig', {})
            )
            
            cls.__preprocess_cfg = PreprocessConfig(
                **ymlData.get('PreprocessConfig', {})
            )
            
        except Exception as e:
            log(WARNING, f"Error loading federation config from {ymlPath}:\n{e}")
            raise e
        
        if all(
            [cls.__flwr_app_cfg, cls.__dataset_cfg, cls.__model_cfg, cls.__training_cfg, cls.__preprocess_cfg]
            ):
            log(INFO, f'All configs loaded successfully.')
    
    @classmethod
    def __loadDatasetMetadata(cls) -> None:
        """
        Load the dataset-specific metadata from a '<dataset name>_metadata.yml' file.
        The path is constructed using 'cls._dataset_cfg.dataset_root' and 'cls._dataset_cfg.dataset_name'.
        :return: None
        """
        datasetName = cls.__dataset_cfg.dataset_name
        dataset_root_path = os.path.join(
            cls.__dataset_cfg.dataset_root, datasetName
        )
        metadata_file_path = os.path.join(dataset_root_path, f'{datasetName}_metadata.yml')

        try:
            with open(metadata_file_path, 'rb') as f:
                
                metadataYml = yaml.safe_load(f)[datasetName]
                cls.__dataset_metadata = DatasetMetadata(
                    dataset_name=datasetName,
                    dataset_path=dataset_root_path,
                    **{k.lower():v for k,v in metadataYml.items()}
                )
                
        except Exception as e:
            log(WARNING, f"Error loading dataset metadata from {metadata_file_path}:\n{e}")
            raise e
    
    
    """ Getters for the configuration classes as either objects or dicts."""        
            
    @classmethod
    def get_flwr_app_cfg(cls, key: str|None = None, asDict: bool = True) -> Union[FlwrAppConfig, dict[str,Any], Any]:
        
        """Get the Flower application configuration or a specific key from it.
        keys:
        
        `num_server_rounds`: Total number of rounds for federated learning.
        `num_partitions`: Number of total clients.
        `client_selection_method`: Method used for selecting clients in each round.
        `aggregation_method`: Method used for aggregating client updates.
        `random_seed`: Seed for random number generation, if any.
        `debug`: Flag to enable or disable extra log messages.
            
        :param key: Optional key to retrieve a specific configuration value.
        :param asDict: If True, return the configuration as a dictionary; otherwise, return the object.
        :return: The configuration value for the specified key or the entire configuration as a dictionary or object.
        """
        
        if key:
            if hasattr(cls.__flwr_app_cfg, key):
                return getattr(cls.__flwr_app_cfg, key)
            else:
                raise KeyError(f"Key '{key}' not found in FlwrAppConfig.")
        else:
            return asdict(cls.__flwr_app_cfg) if asDict else cls.__flwr_app_cfg

    @classmethod
    def get_dataset_cfg(cls, key: str|None = None, asDict: bool = True) -> DatasetConfig |dict[str,Any] | Any:
        
        if key:
            if hasattr(cls.__dataset_cfg, key):
                return getattr(cls.__dataset_cfg, key)
            else:
                raise KeyError(f"Key '{key}' not found in DatasetConfig.")
        else:
            return asdict(cls.__dataset_cfg) if asDict else cls.__dataset_cfg
    
    @classmethod
    def get_dataset_metadata(cls, key: str|None = None, asDict: bool = True) -> Union[DatasetMetadata, dict[str,Any], Any]:
        
        if key:
            if hasattr(cls.__dataset_metadata, key):
                return getattr(cls.__dataset_metadata, key)
            else:
                raise KeyError(f"Key '{key}' not found in DatasetMetadata.")
        else:
            return asdict(cls.__dataset_metadata) if asDict else cls.__dataset_metadata
    
    @classmethod
    def get_model_cfg(cls, key: str|None = None, asDict: bool = True) -> Union[ModelConfig, dict[str, Any], Any]:
        if key:
            if hasattr(cls.__model_cfg, key):
                return getattr(cls.__model_cfg, key)
            else:
                raise KeyError(f"Key '{key}' not found in ModelConfig.")
        else:
            return asdict(cls.__model_cfg) if asDict else cls.__model_cfg

    @classmethod
    def get_training_cfg(cls, key: str|None = None, asDict: bool = True) -> TrainingConfig | dict[str, Any] | Any:
        
        if key:
            if hasattr(cls.__training_cfg, key):
                return getattr(cls.__training_cfg, key)
            else:
                raise KeyError(f"Key '{key}' not found in TrainingConfig.")
        else:
            return asdict(cls.__training_cfg) if asDict else cls.__training_cfg

    @classmethod
    def get_preprocess_cfg(cls, key: str|None = None, asDict: bool = True) -> Union[PreprocessConfig, dict[str, Any], Any]:
        
        if key:
            if hasattr(cls.__preprocess_cfg, key):
                return getattr(cls.__preprocess_cfg, key)
            else:
                raise KeyError(f"Key '{key}' not found in PreprocessConfig.")
        else:
            return asdict(cls.__preprocess_cfg) if asDict else cls.__preprocess_cfg
