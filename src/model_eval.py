import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import f1_score,precision_score,recall_score



def load_model(model_place:str):
    with open(model_place,"rb")as f:
        model=pickle.load(f)
    return model

def load_data(file_path:str)->pd.DataFrame:
    data=pd.read_csv(file_path)
    return data
def data_prep(data:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    x=data.drop("Potability",axis=1)
    y=data["Potability"]
    return x,y


def model_eval(model,x_test:pd.DataFrame,y_test:pd.DataFrame)-> dict:
    y_pred=model.predict(x_test)
    f1=f1_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)      
    output={
         "f1":f1,
         "precision":precision,
         "recall":recall

     }
    return output
def save_json(metrics:dict,file_path:str)->None:
    with open("matrics.json","w") as f:
        json.dump(metrics,f,indent=4)
    
def main():
    model_path=r"D:\Water_portability\mlops_wa_pre\model.pkl"
    test_path=r"D:\Water_portability\mlops_wa_pre\data\processed\pro_test.csv"
    json_path=r"D:\Water_portability\mlops_wa_pre"
    model=load_model(model_path)
    data=load_data(test_path)
    x_test,y_test=data_prep(data)
    metr=model_eval(model,x_test,y_test)
    save_json(metr,json_path)

if __name__=="__main__":
    main()
