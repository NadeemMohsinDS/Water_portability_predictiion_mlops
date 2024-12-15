import pandas as pd
import numpy as np
import pickle
import jason
from sklearn.metrics import f1_score,precision_score,recall_score


def load_model(model_place:str)-> randomForestClassifier:
    with open(model_place,"rb")as f:
        model=pickle.dump(f)

def load_data(file_path:str)->pd.DataFrame:
    data=pd.read_csv(file_path)
    return data
def data_prep(data:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    x=data.drop("Potability",axis=1)
    y=data["Potability"]
    return x,y


def model_eval(model:randomForestClassifier,x_test:pd.DataFrame,y_test:pd.DataFrame)->None:
    y_pred=model.predict(x_test)
    f1=f1_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)      
    print(f"f1_score:{f1}")
    print(f"precision_score:{precision}")
    print(f"recall_score:{recall}")

def main():
    model_place=r"D:\Water_portability\mlops_wa_pre\model.pkl"
    test_path=r"D:\Water_portability\mlops_wa_pre\data\processed\pro_test.csv"
    model=load_model(model_place)
    data=load_data(test_path)
    x_test,y_test=data_prep(data)
    model_eval(model,x_test,y_test)

if __name__=="__main__":
    main()
