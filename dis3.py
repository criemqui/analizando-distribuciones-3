import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# Cargar el archivo CSV
file_path = 'categorias.csv'
df = pd.read_csv(file_path)

# Extraer la columna 'is_dead' para crear un array unidimensional
death_event_array = df['is_dead'].values

# Definir la ruta del archivo para guardar el array numpy
output_death_event_path = 'death_event.npy'

# Guardar el array numpy en un archivo
np.save(output_death_event_path, death_event_array)

print(f'Archivo guardado en: {output_death_event_path}')

# Preparar los datos para TSNE
# Eliminar las columnas 'is_dead' y 'categoria_edad' para obtener X
X = df.drop(columns=['is_dead', 'categoria_edad']).values

# Ejecutar TSNE
X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)

# Verificar la forma del array resultante
print(X_embedded.shape)  # Debería ser (299, 3)

# Guardar el array resultante de TSNE
output_tsne_path = 'X_embedded.npy'
np.save(output_tsne_path, X_embedded)

print(f'Archivo TSNE guardado en: {output_tsne_path}')

# Crear un DataFrame con los resultados de TSNE y la columna de 'is_dead'
tsne_df = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2', 'TSNE3'])
tsne_df['is_dead'] = death_event_array

# Crear el gráfico de dispersión 3D con Plotly
fig = px.scatter_3d(
    tsne_df,
    x='TSNE1',
    y='TSNE2',
    z='TSNE3',
    color='is_dead',
    title='3D Scatter Plot of TSNE results',
    labels={'is_dead': 'Class (0: Alive, 1: Dead)'}
)

# Mostrar el gráfico
fig.show()
