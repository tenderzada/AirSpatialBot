"""
UAV Configuration

Defines configuration for H-UAV and L-UAV variants.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class UAVType(Enum):
    """UAV type enumeration."""
    HIGH_UAV = "h-uav"  # Resource-rich, can update memory
    LOW_UAV = "l-uav"   # Lightweight, inference-only


@dataclass
class UAVConfig:
    """
    Configuration for UAV models.

    This extends the base LLaVA configuration with UAV-specific settings.
    """

    # UAV type
    uav_type: UAVType = UAVType.HIGH_UAV

    # Base model configuration
    model_path: str = "./models/AirSpatialBot"
    vision_tower: str = "/mnt/data/clip-vit-large-patch14-336"
    model_base: Optional[str] = None

    # MAC memory configuration
    enable_mac: bool = True
    memory_dim: int = 4096
    memory_depth: int = 2
    num_persistent_tokens: int = 64
    num_memory_tokens: int = 128

    # Learning configuration (H-UAV only)
    enable_memory_update: bool = True
    surprise_eta: float = 0.9
    learning_theta: float = 0.1
    forgetting_alpha: float = 0.01

    # Communication configuration (L-UAV only)
    enable_communication: bool = False
    self_match_threshold: float = 0.7
    huav_address: Optional[str] = None  # e.g., "localhost:50051"

    # Device configuration
    device: str = "cuda:0"
    load_8bit: bool = False
    load_4bit: bool = False

    def __post_init__(self):
        """Post-initialization validation and adjustment."""
        if self.uav_type == UAVType.HIGH_UAV:
            # H-UAV configuration
            self.enable_memory_update = True
            self.enable_communication = False
        else:
            # L-UAV configuration
            self.enable_memory_update = False
            self.enable_communication = True

            if self.huav_address is None:
                raise ValueError("L-UAV requires huav_address to be set")

    @classmethod
    def create_huav_config(
        cls,
        model_path: str,
        vision_tower: str,
        device: str = "cuda:0",
        **kwargs
    ):
        """Create configuration for H-UAV."""
        return cls(
            uav_type=UAVType.HIGH_UAV,
            model_path=model_path,
            vision_tower=vision_tower,
            device=device,
            enable_memory_update=True,
            enable_communication=False,
            **kwargs
        )

    @classmethod
    def create_luav_config(
        cls,
        model_path: str,
        vision_tower: str,
        huav_address: str,
        device: str = "cuda:1",
        **kwargs
    ):
        """Create configuration for L-UAV."""
        return cls(
            uav_type=UAVType.LOW_UAV,
            model_path=model_path,
            vision_tower=vision_tower,
            huav_address=huav_address,
            device=device,
            enable_memory_update=False,
            enable_communication=True,
            **kwargs
        )


if __name__ == "__main__":
    print("Testing UAVConfig...")

    # H-UAV config
    h_config = UAVConfig.create_huav_config(
        model_path="./models/AirSpatialBot",
        vision_tower="/mnt/data/clip-vit-large-patch14-336"
    )
    print(f"H-UAV Config: {h_config.uav_type}")
    print(f"  - Memory update: {h_config.enable_memory_update}")
    print(f"  - Communication: {h_config.enable_communication}")

    # L-UAV config
    l_config = UAVConfig.create_luav_config(
        model_path="./models/AirSpatialBot",
        vision_tower="/mnt/data/clip-vit-large-patch14-336",
        huav_address="localhost:50051"
    )
    print(f"L-UAV Config: {l_config.uav_type}")
    print(f"  - Memory update: {l_config.enable_memory_update}")
    print(f"  - Communication: {l_config.enable_communication}")
    print(f"  - H-UAV address: {l_config.huav_address}")

    print("✓ UAVConfig tests passed!")
