# analizando-distribuciones-3

Para esta sección usaremos una técnica de reducción de dimensionalidad para tratar de visualizar aproximadamente la estructura de nuestros datos.

Los pasos a seguir para lograrlo son (partiendo del DataFrame anterior):

1. Exportar la una matriz con sólo los valores de los atributos en formato de numpy array.
   - Para esto deberás usar df.drop(columns[<columna-objetivo>]) para eliminar la colúmna que contiene la información si la persona murió o no, también elimina categoria_edad.
   - Puedes convertir un dataframe a un numpy array con su atributo df.values.

2. Exportar un array unidimensional y de sólo la colúmna objetivo DEATH_EVENT.
3. Ejecutar el siguiente fragmento de código (puede demorar unos segundos dependiendo de la capacidad de cómputo de tu PC)
X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)

  dónde X_embedded es un NumPy array de (299, 3)

4. Realizar un gráfico de dispersión 3D con Plotly donde los puntos de cada clase (vivo o muerto) tienen un color asignado para así poder diferenciarlos. (Para esto debes usar el vector y)
   

   ![Texto alternativo](https://github.com/criemqui/analizando-distribuciones-3/blob/main/newplot.png)
