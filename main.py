import yaml
from load_model_function import ModelConfig, load_model_from_registry

# Load the YAML file and access the 'mlflow' section
with open("config.yaml", "r") as f:
    full_config = yaml.safe_load(f)

mlflow_config = full_config.get("mlflow", {})

# Pass the inner config to Pydantic
config = ModelConfig(**mlflow_config)

# Load the model
model = load_model_from_registry(config)