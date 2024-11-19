from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos y seleccionar columnas numéricas (1960-2023)
data = pd.read_csv(r"C:\Users\marmr\Downloads\budapest_weekends.csv")
data = data.drop(columns=['Unnamed: 0','room_type', 'room_shared', 'room_private', 'host_is_superhost','attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm'])

# Correlación
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Análisis de varianza
print("Varianza de las variables:")
print(data.var())

# Eliminar filas con valores faltantes
numerical_data = data.dropna()

# Estandarizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# PCA
pca = PCA()
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score = pd.DataFrame(pca.components_, columns=numerical_data.columns)

# Graficar el aporte de cada variable al primer componente principal
matrix_transform = pca.components_.T
plt.figure(figsize=(10, 6))
plt.bar(range(len(numerical_data.columns)), matrix_transform[:, 0])
plt.xticks(range(len(numerical_data.columns)), numerical_data.columns, rotation=90)
plt.ylabel('Loading Score')
plt.title('Aporte de las Variables al Primer Componente Principal')
plt.show()

top_vars_per_component = {}
for i in range(pca.n_components_):
    component = pd.Series(pca.components_[i], index=data.columns)
    sorted_component = component.abs().sort_values(ascending=False)
    top_vars_per_component[f'PCA{i+1}'] = sorted_component.index[:3].tolist()  # Top 3 variables
print("Variables de mayor aporte para los primeros componentes:", top_vars_per_component)


# Graficar varianza explicada
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(per_var) + 1), per_var, marker='o', linestyle='-', color='b')

for i, var in enumerate(per_var):
    plt.text(i + 1, var + 1, f"{var:.1f}%", ha='center', va='bottom', fontsize=8, color="blue")

plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada (%)')
plt.title('Scree Plot')
plt.show()

# Porcentaje de varianza acumulada
porcent_acum = np.cumsum(per_var)
porcent_acum = np.minimum(porcent_acum, 100) 

# Impresión de resultados
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(porcent_acum) + 1), porcent_acum, marker='o', linestyle='-', color='r')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada (%)')
plt.title('Varianza Acumulada')
plt.show()

# Selección del número de componentes principales para capturar al menos 85% de la varianza
threshold = 85
n_components = np.argmax(porcent_acum >= threshold) + 1
print(f"Número de componentes necesarios para capturar el {threshold}% de la varianza: {n_components}")
