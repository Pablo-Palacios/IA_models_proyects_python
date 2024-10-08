# ViTImageProcessor =  procesa las imagenes antes de ser enviadas al modelo. 
# Los convierte en tensores que el  modelos ViT entiende

#ViTForImageClassification = Clasifica las imagenes. Modelo pre entrenado


# A TENER EN CUENTA: 
"Pérdida media en entrenamiento (Average Training Loss)"
# La pérdida (o loss) es una métrica que mide qué tan mal lo está haciendo el modelo en una determinada tarea. 
# Se calcula utilizando una función de pérdida (por ejemplo, Cross-Entropy para clasificación) 
# que compara las predicciones del modelo con las etiquetas verdaderas.

" Pérdida en validación (Validation Loss)"
# La pérdida en validación se calcula de la misma manera que la pérdida en entrenamiento,
# pero en este caso se evalúa en un conjunto de datos de validación 
# (un conjunto de datos que no se usa durante el entrenamiento, pero sí para evaluar el rendimiento del modelo).

"Precisión en validación (Validation Accuracy)"
# La precisión es una métrica de rendimiento que mide el porcentaje de predicciones correctas del modelo en relación 
# con el número total de predicciones. 
# Es decir, mide cuántas veces el modelo predijo la clase correcta.

# Propósito: La precisión en validación te da una medida del rendimiento del modelo sobre 
# los datos que no ha visto antes. Junto con la pérdida en validación,
# te ayuda a entender si el modelo está funcionando correctamente fuera del conjunto de entrenamiento.









# Ajustes para mejorar el rendimiento:
# Reducir la Pérdida en Validación:

# Regularización: Agregar técnicas como Dropout o L2 regularization para evitar que el 
# modelo memorice los datos de entrenamiento.
# Early Stopping: Detener el entrenamiento cuando la pérdida en validación deje de mejorar.
# Más datos: Ampliar el conjunto de datos de entrenamiento para ayudar al modelo a aprender patrones más generales.
# Mejorar la Precisión en Validación:

# Mejora del Modelo: Probar modelos más complejos o ajustar hiperparámetros (learning rate, número de capas, etc.).
# Data Augmentation: Aumentar la variabilidad del conjunto de datos (rotaciones, cambios de brillo, etc.) 
# para que el modelo aprenda a generalizar mejor.
# Estas técnicas pueden ayudarte a reducir el sobreajuste y mejorar el rendimiento general 
# del modelo en el conjunto de validación.