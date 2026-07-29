"""Regression tests for CityLearn DHW storage actions."""

import numpy as np
import pytest

from citylearn.citylearn import CityLearnEnv


def _central_dhw_action(env: CityLearnEnv, value: float) -> list[np.ndarray]:
    action = np.zeros(env.action_space[0].shape, dtype=np.float32)

    for index, name in enumerate(env.action_names[0]):
        if name == "dhw_storage":
            action[index] = value

    return [action]


def test_challenge_2023_dhw_storage_charges_and_offsets_demand():
    env = CityLearnEnv("citylearn_challenge_2023_phase_1")

    try:
        env.reset(seed=0)
        building = env.buildings[0]
        assert building.dhw_storage.capacity > 0.0
        assert building.heating_storage.capacity == 0.0

        env.step(_central_dhw_action(env, 1.0))
        charged_energy = float(building.dhw_storage.energy_balance[0])
        charged_soc = float(building.dhw_storage.soc[0])
        assert charged_energy > 0.0
        assert charged_soc > 0.0

        env.step(_central_dhw_action(env, -1.0))
        demand = float(building.dhw_demand[1])
        discharged_energy = float(building.dhw_storage.energy_balance[1])
        assert demand > 0.0
        assert discharged_energy == pytest.approx(-demand)
        assert float(building.dhw_storage.soc[1]) < charged_soc
        assert float(building.dhw_device.electricity_consumption[1]) == pytest.approx(0.0)
    finally:
        env.close()
