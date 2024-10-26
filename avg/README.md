# Análisis de Estadísticas de Béisbol con Modelos de Regresión Lineal y Ensamble

El objetivo principal de nuestro proyecto es desarrollar modelos de inteligencia artificial que puedan predecir promedios de bateo en béisbol. Usamos información estadística real de jugadores (por ejemplo, turnos al bate, hits, y bases alcanzadas) y la organizamos en un dataset. Este dataset sirve de base para entrenar y evaluar diferentes tipos de modelos, que nos ayudan a entender los patrones en los datos y hacer predicciones precisas.

Para entrenar y evaluar los modelos, dividimos el dataset en dos partes: un conjunto de entrenamiento y un conjunto de prueba. Los datos de entrenamiento se utilizan para ajustar los modelos, mientras que los datos de prueba nos permiten evaluar el rendimiento y verificar si los modelos pueden generalizar bien a nuevos datos.

Al comparar las predicciones y los valores reales del promedio de bateo, observamos que:

La regresión lineal produce predicciones cercanas y es interpretativamente útil, pero puede ser menos precisa si los datos presentan relaciones no lineales.
Los modelos de ensamble tienden a ser más precisos en promedio, especialmente cuando los datos son complejos o contienen patrones difíciles de capturar.