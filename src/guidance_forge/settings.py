# ---INFO-----------------------------------------------------------------------
"""
Settings for the package.
"""

# ---DEPENDENCIES---------------------------------------------------------------
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ------------------------------------------------------------------------------
ENV_FILE = Path(__file__).parents[2] / "env" / ".env"


# ---GoogleAISettings-----------------------------------------------------------
class GoogleAISettings(BaseSettings):
    """
    Settings for the Google AI API.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="GOOGLEAI_",
        extra="ignore",
    )
    api_key: str
