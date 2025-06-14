import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS. CASO DE MERMAS*")


df_producto = pd.read_csv(r"C:\Users\kanom\OneDrive\Desktop\Universidad\Programacion\mermas.csv")
df_tienda = pd.read_csv(r"C:\Users\kanom\OneDrive\Desktop\Universidad\Programacion\tienda.csv")
df_motivo_merma = pd.read_csv(r"C:\Users\kanom\OneDrive\Desktop\Universidad\Programacion\motivo_merma.csv")
df_mermas = pd.read_csv(r"C:\Users\kanom\OneDrive\Desktop\Universidad\Programacion\mermas.csv")
df_tiempo = pd.read_csv(r"C:\Users\kanom\OneDrive\Desktop\Universidad\Programacion\tiempo.csv")

data = df_mermas.merge(df_producto, on="nombre_producto", how="left")
data = df_mermas.merge(df_tienda, left_on="tienda", right_on="tienda", how="left")\
                .merge(df_motivo_merma, on="motivo", how="left")\
                .merge(df_tiempo, on="fecha", how="left")

data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
data['nombre_mes'] = data['fecha'].dt.month
data['año'] = data['fecha'].dt.year
data['semana'] = data['fecha'].dt.isocalendar().week

features = ['nombre_producto', 'categoria', 'motivo', 'tienda', 'comuna_x', 'region_x', 'nombre_mes', 'semana_x', 'año']
X = data[features]
y = data['valor_total_merma']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['nombre_producto', 'categoria', 'motivo', 'tienda', 'comuna_x', 'region_x']
numeric_features = ['nombre_mes', 'semana_x', 'año']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

print("Entrenando modelos...")
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
print("Modelos entrenados.")

y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)

metrics = {
    "Regresión Lineal": [mean_absolute_error(y_test, y_pred_lr), np.sqrt(mean_squared_error(y_test, y_pred_lr)), r2_score(y_test, y_pred_lr)],
    "Random Forest": [mean_absolute_error(y_test, y_pred_rf), np.sqrt(mean_squared_error(y_test, y_pred_rf)), r2_score(y_test, y_pred_rf)]
}

print("\n=== COMPARACIÓN DE MODELOS ===")
for model, values in metrics.items():
    print(f"{model} → MAE: {values[0]:.2f} | RMSE: {values[1]:.2f} | R²: {values[2]:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valor Real de Merma')
plt.ylabel('Predicción')
plt.title('Predicción de Mermas')
plt.savefig('predicciones_mermas_rf.png')
print("Gráfico guardado: predicciones_mermas_rf.png")