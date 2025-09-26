import pandas as pd
from sklearn import preprocessing

def load_and_clean_data(file_path='data/mammographic_masses.data.txt'):
    masses_data = pd.read_csv(
        file_path,
        na_values=['?'],
        names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']
    )
    masses_data.dropna(inplace=True)
    all_features = masses_data[['age', 'shape', 'margin', 'density']].values
    all_classes = masses_data['severity'].values
    feature_names = ['age', 'shape', 'margin', 'density']
    return all_features, all_classes, feature_names

def scale_features(features, method='standard'):
    if method == 'standard':
        scaler = preprocessing.StandardScaler()
    elif method == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    scaled = scaler.fit_transform(features)
    return scaled
