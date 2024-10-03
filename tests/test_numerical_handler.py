import pytest
import pandas as pd
from numerical_handler import handle_numerical


def test_handle_numerical():
    df = pd.DataFrame({
        'num_feature': [1.0, 2.5, 3.7, 4.1, 5.6],
        'target': [10, 20, 30, 40, 50]
    })

    scaled_feature = handle_numerical(df['num_feature'], df['target'])

    # Check if the output is scaled correctly
    assert scaled_feature.shape == (5, 1)
    assert abs(scaled_feature.mean()) < 1e-5  # Check if scaled feature has zero mean
    assert abs(scaled_feature.std() - 1) < 1e-5  # Check if scaled feature has unit variance
