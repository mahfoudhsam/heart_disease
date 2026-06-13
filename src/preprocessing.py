import pandas as pd
import numpy as np
 
def split_data(X,y,test_size=0.2,seed=42):

    np.random.seed(seed)
    n_samples=X.shape[0]
    indices=np.arange(n_samples)
    np.random.shuffle(indices)

    test_set_size=int(n_samples*test_size)
    test_indices=indices[:test_set_size]
    train_indices=indices[test_set_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Add this scaling helper to src/preprocessing.py
def standard_scale(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1, std)

    return (X - mean) / std

def load_data(data_path):
    df=pd.read_csv(data_path)
    df.drop(columns=['patient_id'])
    df['disease']=df['disease'].map({'Yes':1,'No':-1}) 
    y=df['disease'].values
    X_features=df.drop(columns=['disease'])
    binary_mapping={'Male':0,'Female':1,'No':0,'Yes':1}
    activity_mapping={'Low':0,'Medium':1,'High':2}
    for col in X_features.columns:
        if X_features[col].dtype=='object':
            unique_val=X_features[col].dropna().unique()
            if 'Low' in unique_val or 'High' in unique_val:
                X_features[col]=X_features[col].map(activity_mapping)
            else:
                X_features[col]=X_features[col].map(binary_mapping)
    X=X_features.values.astype(float)
    X_scaled = standard_scale(X)
    
    return split_data(X_scaled,y)

