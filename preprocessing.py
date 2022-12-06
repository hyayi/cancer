from sklearn.preprocessing import LabelEncoder,QuantileTransformer,StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

def get_values(value):
    return value.values.reshape(-1, 1)

def preprocessing(train_path,test_path):

    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    numeric_cols = ['나이', '암의 장경', 'KI-67_LI_percent', '암의 개수','NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']
    none_scale_cols = ['ER','PR','HER2','DCIS_or_LCIS_여부']
    categorical_cols = ['진단명', '암의 위치','BRCA_mutation','T_category']

    for col in train_df.columns:
        
        if col in ignore_cols:
            continue
        
        elif col in numeric_cols:
            scaler = StandardScaler()
            train_df[col] = scaler.fit_transform(get_values(train_df[col]))
            test_df[col] = scaler.transform(get_values(val_df[col]))
            
        elif col in none_scale_cols:
            continue
    
        elif col in categorical_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(get_values(train_df[col]))
            test_df[col] = le.transform(get_values(test_df[col]))
        
        else:
            continue
    
    return train_df, test_df
