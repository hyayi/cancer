from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

def get_values(value):
    return value.values.reshape(-1, 1)

def preprocessing(train_path,test_path):

    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, val_df, train_labels, val_labels = train_test_split(
                                                        train_df.drop(columns=['N_category']), 
                                                        train_df['N_category'], 
                                                        test_size=0.2, 
                                                        random_state=42
                                                    )
    target = ['ID', 'img_path', 'mask_path', '수술연월일','나이', '진단명', '암의 위치', '암의 개수', '암의 장경', 'NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3','DCIS_or_LCIS_여부', 'T_category', 'ER','PR', 'KI-67_LI_percent', 'HER2','BRCA_mutation']
    
    target_test = ['ID', 'img_path', '수술연월일','나이', '진단명', '암의 위치', '암의 개수',
    '암의 장경', 'NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3',
    'DCIS_or_LCIS_여부', 'T_category', 'ER','PR', 'KI-67_LI_percent', 'HER2','BRCA_mutation']
    
    train_df = train_df[target]
    val_df = val_df[target] 
    test_df = test_df[target_test] 
    
    
    numeric_cols = ['나이', '암의 장경', 'KI-67_LI_percent', '암의 개수','NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']
    none_scale_cols = ['ER','PR','HER2','DCIS_or_LCIS_여부']

    for col in train_df.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            scaler = StandardScaler()
            train_df[col] = scaler.fit_transform(get_values(train_df[col]))
            val_df[col] = scaler.transform(get_values(val_df[col]))
            test_df[col] = scaler.transform(get_values(test_df[col]))
            
        elif col in none_scale_cols:
            continue
        
        else:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(get_values(train_df[col]))
            val_df[col] = le.transform(get_values(val_df[col]))
            test_df[col] = le.transform(get_values(test_df[col]))
    
    return train_df, val_df, test_df, train_labels, val_labels
