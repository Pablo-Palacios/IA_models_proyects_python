from flask import Blueprint
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset/data_prome.cvs")
data_ = pd.read_csv("dataset/data_at_bats.cvs")

def entrenar_modelo(data1,data2):
    modelo = LinearRegression()
    modelo.fit(data1,data2)
    return modelo

x = data_[['PA','SO','BB','SF']]

y = data_['ATBS']

x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)



a = data[['AB','H']]

b = data['AVG']


a_train, a_test, b_train, b_test = train_test_split(a,b, test_size=0.2, random_state=42)



# modelo calcular at_bats
# model = entrenar_modelo(x_train,y_train)

# score = model.score(x_test,y_test)
# print(f"score del modelo: {score}")

# predicion = model.predict(x_test)
# print(f"prediccion modelo: {predicion} - dato real: {y_test}")



# modelo calcular avg
model_ = entrenar_modelo(a_train,b_train)

score_ = model_.score(a_test,b_test)
print(f"score del modelo: {score_}")

predicion_ = model_.predict(a_test)
print(f"prediccion modelo: {predicion_} - dato real: {b_test}")







# bp = Blueprint('avg',__name__,url_prefix="/avg")





# bp.route("/bateador", methods = ["POST"])
# def avgs():
#     pass