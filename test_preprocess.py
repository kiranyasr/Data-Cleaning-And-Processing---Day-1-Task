import pandas as pd
from pathlib import Path
import pytest
from preprocess import load_data, engineer_features

# Create a dummy CSV file for testing
@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing the load_data function."""
    data = {
        'Name': ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Allen, Mr. William Henry'],
        'Age': [22, 26, 35],
        'SibSp': [1, 0, 0],
        'Parch': [0, 0, 0],
        'Sex': ['male', 'female', 'male'],
        'Embarked': ['S', 'S', 'Q']
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    return filepath

def test_load_data(dummy_csv: Path):
    """Test that data is loaded correctly into a pandas DataFrame."""
    df = load_data(dummy_csv)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Name' in df.columns

def test_engineer_features():
    """Test that feature engineering creates the correct new columns."""
    data = {
        'Name': ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina'],
        'SibSp': [1, 0],
        'Parch': [0, 0]
    }
    df = pd.DataFrame(data)
    df_featured = engineer_features(df)

    # Test if new columns are created
    assert 'FamilySize' in df_featured.columns
    assert 'IsAlone' in df_featured.columns
    assert 'Title' in df_featured.columns

    # Test the logic of the new columns
    assert df_featured['FamilySize'].tolist() == [2, 1]
    assert df_featured['IsAlone'].tolist() == [0, 1]
    assert df_featured['Title'].tolist() == ['Mr', 'Miss']