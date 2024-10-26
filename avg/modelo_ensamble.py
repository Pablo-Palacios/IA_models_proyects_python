from flask import Blueprint
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("dataset/data_prome.cvs")
data_ = pd.read_csv("dataset/data_at_bats.cvs")

def entrenar_modelo(data1,data2):
    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(data1,data2)
    return modelo

x = data_[['PA','SO','BB','SF']]

y = data_['ATBS']

x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)



a = data[['AB','H']]

b = data['AVG']


a_train, b_train, a_test, b_test = train_test_split(a,b, test_size=0.2, random_state=42)



# modelo calcular at_bats
model = entrenar_modelo(x_train,y_train)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

score_rf = model.score(x_test,y_test)
predicciones_rf = model.predict(x_test)

print("Score del modelo Random Forest:", score_rf)
print("Predicción modelo Random Forest:", predicciones_rf[:5])
print("Datos reales:", y_test[:5].values)