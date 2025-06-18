from pydantic import BaseModel, model_validator
from typing import Optional
import mlflow

class ModelConfig(BaseModel):
    """
    Configuration model for loading an MLflow model from the Model Registry.

    Attributes:
        model_name (str): The registered name of the model in MLflow.
        version (Optional[str]): The specific version of the model to load. Mutually exclusive with `stage`.
        stage (Optional[str]): The stage of the model to load (e.g., 'Production', 'Staging'). Mutually exclusive with `version`.
        tracking_uri (Optional[str]): Optional URI for the MLflow tracking server. Defaults to local if not provided.
    """

    model_name: str
    version: Optional[str] = None
    stage: Optional[str] = None
    tracking_uri: Optional[str] = None

    @model_validator(mode="after")
    def validate_stage_or_version(self):
        """
        Validates that either 'stage' or 'version' is provided (but not both).
        Defaults to 'Production' stage if neither is specified.
        """
        if self.version and self.stage:
            raise ValueError("Specify only one of 'version' or 'stage', not both.")
        if not self.version and not self.stage:
            self.stage = "Production"
        return self


def load_model_from_registry(config: ModelConfig):
    """
    Loads a model from the MLflow Model Registry based on the provided configuration.

    Args:
        config (ModelConfig): A validated configuration object containing model name,
                              version or stage, and optional tracking URI.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded MLflow model, ready for inference.

    Raises:
        RuntimeError: If the model cannot be loaded from the registry.
        ValueError: If both 'stage' and 'version' are provided in the configuration.

    Example:
        >>> config = ModelConfig(model_name="HeartDiseaseModel", stage="Production")
        >>> model = load_model_from_registry(config)
        >>> model.predict(pd.DataFrame([[1, 2, 3, 4]]))
    """
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)

    model_uri = f"models:/{config.model_name}/{config.stage or config.version}"
    print(f"[INFO] Loading model from: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_uri}: {e}")

    return model
