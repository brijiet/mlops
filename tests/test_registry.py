from src.registry import load_champion_model


def test_load_champion():

    model = load_champion_model()

    assert model is not None