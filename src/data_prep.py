import pandas as pd
import yaml
import os

def load_data(file_path: str)-> pd.DataFrame:
    data=pd.read_csv(file_path)
    return data

def clean_data(data: pd.DataFrame)->pd.DataFrame:
    try:
        for column in data.columns:
            if data[column].isnull().any():
                mean=data[column].mean()
                data[column].fillna(mean,inplace=True)
        return data

    except Exception as e:
        raise Exception(f"error in getting data{e}")
def save_data(data:pd.DataFrame,file_path:str)->None:
    data.to_csv(file_path)

def main():
    test_path=r"D:\Water_portability\mlops_wa_pre\data\raw\test.csv"
    train_path=r"D:\Water_portability\mlops_wa_pre\data\raw\train.csv"

    proc_data_path="./data/processed"
    os.makedirs(proc_data_path)

    test_load=load_data(test_path)
    train_load=load_data(train_path)
    proc_test=clean_data(test_load)
    proc_train=clean_data(train_load)
    save_data(proc_test,os.path.join(proc_data_path,"pro_test.csv"))
    save_data(proc_train,os.path.join(proc_data_path,"pro_train.csv"))
if __name__=="__main__":
    main()







     



