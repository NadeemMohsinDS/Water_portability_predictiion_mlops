import pandas as pd 
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def param_load(file_path:str)-> int:
    with open(file_path,"r") as f:
        param=yaml.safe_load(f)["model_traning"]["n_estimaters"]
    return param

def load_data(file_path:str)->pd.DataFrame:
    data=pd.read_csv(file_path)
    return data
def data_split(data:pd.DataFrame)-> tuple[pd.DataFrame,pd.DataFrame]:
    x=data.drop('Potability',axis=1) 
    y=data['Potability']
    return x,y


def train_model(x: pd.DataFrame,y:pd.DataFrame,param:int)->RandomForestClassifier:
    clf=RandomForestClassifier(n_estimators=param)
    clf.fit(x,y)
    return clf
def save_model(clf:RandomForestClassifier,model_name:str)->None:
    """
    This function takes a trained model and a file path as input, and saves the model to the given file path.
    The model is saved using the pickle library.
    """
    import pickle
    with open(model_name,"wb") as f:
        pickle.dump(clf,f)

def main():
    """
    This function is the main entry point of the model training pipeline.
    It reads the hyperparameter from a yaml file, loads the data, splits it into features and labels and trains a model using the hyperparameter.
    The trained model is then saved to a file.
    """
    
    para_load=r"D:\Water_portability\mlops_wa_pre\params.yaml"
    data_load=r"D:\Water_portability\mlops_wa_pre\data\processed\pro_train.csv"
    model_path="model.pkl"
    param=param_load(para_load)
    data=load_data(data_load)    
    x,y=data_split(data)
    model=train_model(x,y,param)
    save_model(model,model_path)

if __name__=="__main__": 
    main()