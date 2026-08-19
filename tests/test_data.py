from src.data_processing import load_data


def test_load_data():

    X, y = load_data()

    assert X is not None
    assert y is not None


def test_data_shape():

    X, y = load_data()

    assert X.shape[0] == 569
    assert X.shape[1] == 30


def test_target_length():

    X, y = load_data()

    assert len(X) == len(y)