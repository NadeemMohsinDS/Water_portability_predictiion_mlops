import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

#test_size=yaml.safe_load(params.yaml)['data_collection']['test_size']
def load_para(file_path_para : str)-> float:
    try:
        with open(file_path_para,'r') as f:
            test_size=yaml.safe_load(f)['data_collecton']['test_size']
            return test_size
    except Exception as e:
        raise Exception (f"error in loading parameter path fro{file_path_para}:{e }")

        


def load_data(file_path:str) -> pd.DataFrame:
    try: 
        df = pd.read_csv(file_path)
        return df
                        
    except Exception as e:
        raise Exception(f'error in loading data from {file_path}:{e}')

def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        train_data,test_data=train_test_split(data,test_size=test_size,random_state=40)
        return train_data,test_data
    except ValueError as e:
        raise ValueError(f"error in data splitting {e }")

def save_data(data: pd.DataFrame,save_path: str) -> None:
    try:
        data.to_csv(save_path,index=False)

    except Exception as e:
        raise Exception(f"error in data saving {save_path}:{e}")


def main():
    file_path=r"C:\Users\Nadeem Mohsin\water_potability.csv"
    para_path=r"D:\Water_portability\mlops_wa_pre\params.yaml"
    to_save_path=os.path.join('data','raw')
    os.makedirs(to_save_path)

    data=load_data(file_path)
    test_size=load_para(para_path)
    train_data,test_data=split_data(data,test_size)
    save_data(train_data,os.path.join(to_save_path,'train.csv'))
    save_data(test_data,os.path.join(to_save_path,'test.csv'))

if  __name__ == "__main__":
    main()

    







#df = pd.read_csv(r"C:\Users\Nadeem Mohsin\water_potability.csv")

