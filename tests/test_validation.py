import pytest
from pydantic import BaseModel, Field, ValidationError


class RadiusInput(BaseModel):
    world_edge_radius: float = Field(default=0.03, gt=0)


def test_default_radius():
    request = RadiusInput()
    assert request.world_edge_radius == 0.03


def test_negative_radius():
    with pytest.raises(ValidationError):
        RadiusInput(world_edge_radius=-0.01)