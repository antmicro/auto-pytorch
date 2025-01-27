from typing import Any, Generator, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
import numpy as np
import smac.configspace
import smac.optimizer.acquisition.maximizer

def convert_np_types(obj: Any) -> Any:
    """
    Converts NumPy object to JSON-friendly types.

    Args:
        obj (Any):
            Object to convert

    Returns:
        Any:
            Converted object
    """
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)

    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, (np.complex64, np.complex128)):
        return {"real": obj.real, "imag": obj.imag}

    elif isinstance(obj, (np.str_)):
        return str(obj)

    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    elif isinstance(obj, (np.bool_)):
        return bool(obj)

    elif isinstance(obj, (np.void)):
        return None

    elif isinstance(obj, list):
        return [convert_np_types(_o) for _o in obj]

    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}

    return obj

class CustomConfiguration(Configuration):
    """
    Custom Configuration making newer ConfigSpace versions compatible
    with the old one (0.6.1). It also returns value in JSON-friently format.
    """
    def __init__(
        self,
        configuration_space: ConfigurationSpace,
        values: Optional[Mapping[str, Any]] = None,
        vector: Optional[Array[f64]] = None,
        allow_inactive_with_values: bool = False,
        origin: Optional[Any] = None,
        config_id: Optional[int] = None,
    ):
        super().__init__(
            configuration_space=configuration_space,
            values=values,
            vector=vector,
            allow_inactive_with_values=allow_inactive_with_values,
            origin=origin,
            config_id=config_id,
        )
        self.configuration_space = self.config_space

    def __getitem__(self, key: str) -> Any:
        try:
            value = super().__getitem__(key)
        except KeyError:
            # Disable KeyError to enable configs comparison in SMAC
            return None
        return convert_np_types(value)

    def is_valid_configuration(self):
        self.check_valid_configuration()

    @classmethod
    def cast(cls, x: Configuration) -> "CustomConfiguration":
        # Set additional variables for backward compatibility
        x.configuration_space = x.config_space
        x.__class__ = CustomConfiguration
        return x

class CustomConfigurationSpace(ConfigurationSpace):
    """
    Custom Configuration Space making newer ConfigSpace versions compatible
    with the old one (0.6.1).
    It also returns CustomConfiguration instead of Configuration.
    """
    def __init__(
            self,
            name: Union[str, None] = None,
            seed: Union[int, None] = None,
            meta: Union[dict, None] = None,
            *,
            space=None
        ):
        super().__init__(name=name, seed=seed, meta=meta, space=space)
        self.origin = None
        
    def sample_configuration(self, size=None) -> Union[Configuration, list[Configuration]]:
        configs = super().sample_configuration(size=size)

        is_list = True
        if not isinstance(configs, list):
            is_list = False
            configs = [configs]

        for i, conf in enumerate(configs):
            configs[i] = CustomConfiguration.cast(conf)

        return configs if is_list else configs[0]

    def get_default_configuration(self):
        config = super().get_default_configuration()

        return CustomConfiguration.cast(config)


smac_get_one_exchange_neighbourhood = smac.configspace.get_one_exchange_neighbourhood

def get_one_exchange_neighbourhood(
    configuration: CustomConfiguration,
    seed: Union[int, np.random.RandomState],
) -> Generator[CustomConfiguration, None, None]:
    """
    Creates neighbours of the given configuration
    and casts them to CustomConfiguration.

    Args:
        configuration (CustomConfiguration):
            Configuration from which neighbours are generated
        seed (Union[int, np.random.RandomState])
            Seed for neighbour generation

    Yields:
        CustomConfiguration
            The neighbour of configuration
    """
    results = smac_get_one_exchange_neighbourhood(
        configuration=configuration,
        seed=seed,
    )

    for result in results:
        yield CustomConfiguration.cast(result)

smac.configspace.get_one_exchange_neighbourhood = get_one_exchange_neighbourhood
smac.optimizer.acquisition.maximizer.get_one_exchange_neighbourhood = get_one_exchange_neighbourhood
