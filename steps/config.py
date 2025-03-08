from zenml.config.base_settings import BaseSettings


class ModelNameConfig(BaseSettings):
    """Model Configs"""
    name: str = 'Linear Regression'
