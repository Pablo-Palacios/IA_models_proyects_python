from transformers import ViTImageProcessor,ViTForImageClassification
from PIL import Image
import requests
from datasets import load_dataset
import os
import torch




# Funcion que asegura un formato de tamaño para el modelo del preentreno

def prepro_image(image,target_side = (224, 224)):
     # Si la imagen ya es un objeto 'Image', no es necesario abrirla.
    if isinstance(image, Image.Image):
        resized_image = image.resize(target_side)
    else:
        resized_image = Image.open(image).resize(target_side)
    
    return resized_image





# creacion de evaluate_model()
def evaluate_model(model,val_dataset):
    model.eval()

    total_loss = 0
    correct_predi= 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dataset:
            # inputs = batch["pixel_values"]
            # labels = batch["label"]

            pixel_values = []
            # extraer cada img de la etiqueta
            for image in batch['pixel_values']:
                if isinstance(image, list):  # Si es una lista, convertir cada elemento
                    image_tensor = torch.tensor(image).to(model.device)
                    pixel_values.append(image_tensor)
                else:
                    pixel_values.append(image.to(model.device))

            # Apilar las imágenes
            inputs = torch.stack(pixel_values).to(model.device)

            # Extraer las etiquetas
            labels = batch['label']  # Esto debe ser un tensor o un valor único
            if isinstance(labels, list):  # Si es una lista, convertir a tensor
                labels = torch.tensor(labels).to(model.device)
            else:
                labels = torch.tensor([labels]).to(model.device) 
            
            # Si los inputs son un tensor con 3 dimensiones, probablemente deberías apilar
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0) 

            outputs = model(inputs)

            if len(labels.shape) > 1:
                labels = labels.squeeze() 

            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
                 
            # prediccion: busca el valor maximo osea 1, que es el representante de la clase mas probable 
            _, predicted = torch.max(outputs.logits, 1)
            # suma los true y los guarda
            correct_predi += (predicted == labels).sum().item()
            # obtiene el tamaño mas grande de la primera labels
            total_samples = labels.size(0)

    avg_loss = total_loss / len(val_dataset)
    accuracy= correct_predi / total_samples
    print(f"Pérdida en validación: {avg_loss:.4f}, Precisión: {accuracy * 100:.2f}%")

    model.train()

    return avg_loss,accuracy


# creacion de funcion de train_model
# Toma como parametro el modelo, el optimizador, la predicciones de perdida y las epocas "eporch" que sera la cantidad de veces que se genere el entrenamiento
def train_model(model,train_dataset,val_dataset,eporchs):
    # indicamos al modelo que arranque su entrenamiento
    model.train()

    # generamos un bucle de la cantidad de epocas que le indidcamos al principio y iniciamos la variable de total_loss en cero para saber cuantas veces falla al final
    for eporch in range(eporchs):
        total_loss = 0
    
    # generamos un bucle sobre los datos del train_dataset, los inputs(imgs) y labels(etiquetas reales)
        for batch in train_dataset:
             # Imprimir el contenido de batch
            #print("Contenido de batch:", batch)

            # Imprimir tipo y forma de pixel_values
            #print("Tipo de pixel_values:", type(batch['pixel_values']))
            if isinstance(batch['pixel_values'], torch.Tensor):
                pass
                #print("Forma de pixel_values:", batch['pixel_values'].shape)
            else:
                pass
                #print("Longitud de pixel_values:", len(batch['pixel_values']))


            # Convertir cada imagen en un tensor si es necesario
            pixel_values = []
            for image in batch['pixel_values']:
                if isinstance(image, list):  # Si es una lista, convertir cada elemento
                    image_tensor = torch.tensor(image).to(model.device)
                    pixel_values.append(image_tensor)
                else:
                    pixel_values.append(image.to(model.device))

            # Apilar las imágenes
            inputs = torch.stack(pixel_values).to(model.device)

            #inputs = barch["pixel_values"].to(model.device)
            # Extraer las etiquetas
            labels = batch['label']  # Esto debe ser un tensor o un valor único
            if isinstance(labels, list):  # Si es una lista, convertir a tensor
                labels = torch.tensor(labels).to(model.device)
            else:
                labels = torch.tensor([labels]).to(model.device) 
            
            # Si los inputs son un tensor con 3 dimensiones, probablemente deberías apilar
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)  # Asegúrate de agregar la dimensión de batch_size

        
        # iniciamos variable que mande al modelo los inputs(imgs) que utilizaremos 
            outputs = model(inputs)

            # Asegúrate de que las etiquetas estén en la forma correcta
            if len(labels.shape) > 1:
                labels = labels.squeeze()  # Asegúrate de que sea un tensor 1D

            #print(f"Forma de labels antes de calcular la pérdida: {labels.shape}")
            # variable loss: integramos a la funcion de las predicciones, los outputs que arroje el modelo (logist) y los comparamos con los labels
            loss = loss_fn(outputs.logits, labels)

            # luego le indicamos al optimizador que se reinicien los gradientes que se van acumulando
            optimizador.zero_grad()
            # comparamos los gradientes tomados y los comparamos con las perdidas generadas, desde aca se cranea el paso a seguir del entrenamiento (hace un ajuste)
            loss.backward()
            # le decimos al optimizador que genere una actualizacion de los gradientes arrojados anteriormente para continuar por el mejor camino
            optimizador.step()

            # actualizamos el contador de loss para poder saber las perdidas totales de la epoca iniciada
            total_loss += loss.item()

        # se genera un promedio general para saber como esta aprendiendo el modelo a medida que avanzan las epocas
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {eporch + 1}/{eporchs}, Pérdida media: {avg_loss:.4f}")

    evaluate_model(model, val_dataset)
    






model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# cargar el dataset para poder manipularlo
dataset = load_dataset("imagefolder", data_dir="/home/pablopalacios/code/langchain/traductor/dataset")

def preprocess_images(examples):
    # Aplica el extractor sobre las imágenes que son listas de rutas
    images = [prepro_image(image).convert("RGB") for image in examples['image']]  # Abrir las imágenes
    # Convertir a tensores utilizando el extractor
    processed_images = extractor(images, return_tensors='pt')
    return {"pixel_values": processed_images['pixel_values']} 

# Aplicar el proceso de conversión de tensores a cada imagen con map del dataset
dataset = dataset.map(preprocess_images, batched=True)


# dividir el dataset en entrenamiento y validacion
train_dataset = dataset['train']
split_dataset = train_dataset.train_test_split(test_size=0.2) # %80 de entrenamiento, %20 para validacion
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']


# definimos el optimizador de entrenamiento, en este caso utilizamos adamW.
# debemos reiniciar los parametros del modelo (pesos y sesgos) 
# Luego pasamos la tasa de aprendizaje "lr", un numero muy grande aprenderia rapido y no converja bien. Un numero muy bajo, el modelo aprende muy despacio
# lr=5e-5 = 0.00005
optimizador = torch.optim.AdamW(model.parameters(), lr=5e-5)

# aca se evaluan que tan alejado estan las predicciones del modelo sobre las respuestas reales
loss_fn = torch.nn.CrossEntropyLoss()

train_model(model,train_dataset,val_dataset,eporchs=4)