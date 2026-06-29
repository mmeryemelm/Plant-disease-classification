import time #mesurer le temps pris par chaque époque pendant l'entraînement.
import os #Utilisé pour interagir avec le système d'exploitation et gérer les chemins de fichiers.
import tensorflow as tf #La bibliothèque de deep learning utilisée pour construire et entraîner le modèle.

#Divers modules de tensorflow.keras sont importés pour créer le modèle et effectuer le prétraitement des données.
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_dataset(dataset_dir, image_size=(224, 224), batch_size=32):

    # ImageDataGenerator de Keras pour créer des générateurs de données à la fois pour l'entraînement et la validation.
    # augmentation des données d'entraînement pour augmenter la diversité des échantillons d'entraînement.
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input, #fonction de prétraitement à appliquer sur chaque image
        validation_split=0.2,  #20% des données sont réservées pour la validation, 80% des données sera utilisé pour l'entraînement du modèle
        rotation_range=40, #rotation
        width_shift_range=0.2,  #déplacement
        height_shift_range=0.2,  #déplacement
        horizontal_flip=True,  #retournement
        shear_range=0.2  #contrôle la quantité de cisaillement aléatoire appliquée aux images.
    )

    # charger les données d'entraînement à partir du répertoire spécifié par dataset_dir
    train_data = data_generator.flow_from_directory(
        dataset_dir,
        target_size=image_size, # image_size est un tuple contenant la taille souhaitée des images (par exemple, (224, 224) ou (299, 299))
        batch_size=batch_size,  # Le nombre d'images inclus dans chaque lot (batch) d'entraînement.
        class_mode='categorical',  #chaque classe aura une valeur de 1 dans le vecteur correspondant à la classe et  0 ailleurs.
        shuffle=True,  #les données d'entraînement seront mélangées aléatoirement à chaque époque.
        seed=123,  # La graine (seed) pour l'aléatoire, assurant la reproductibilité des résultats si nécessaire.
        subset='training'
    )

    # charger les données de validation à partir du répertoire spécifié par dataset_dir
    validation_data = data_generator.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
        subset='validation'
    )

    # Extraction des étiquettes de classe à partir des noms des sous-répertoires de la base de données

    #train_data.class_indices dictionnaire qui mappe les noms des sous-répertoires (noms de classe)
    # aux entiers qui leur sont associés en tant qu'étiquettes de classe.


    class_labels = sorted(train_data.class_indices, key=train_data.class_indices.get) #tri des classe dans l'ordre numérique.
    train_data.class_labels = class_labels #l'information des étiquettes est ajoutée à l'objet train_data pour une référence future.
    validation_data.class_labels = class_labels #l'information des étiquettes est ajoutée à l'objet validation_data pour une référence future.

    return train_data, validation_data

def fine_tune_cnn(model, train_data, validation_data):


    #Les 30 dernières couches sont décongelées pour permettre leur entraînement sur la nouvelle tâche de classification.
    for layer in model.layers[-30:]:
        layer.trainable = True

    # couche de Average pooling est ajoutée pour réduire les dimensions spatiales
    x = model.output #la couche de sortie qui contient les caractéristiques apprises par le modèle jusqu'à présent.
    x = GlobalAveragePooling2D()(x)  #calcule la moyenne des valeurs dans chaque canal tout en conservant les informations essentielles.
    # la sortie de la couche GlobalAveragePooling2D remplace ainsi la sortie précédente du modèle.

    # Ajout d'une couche Dense finale pour la classification
    predictions = Dense(39, activation='softmax')(x)
    # couche Dense de TensorFlow avec 39 unités (neurones) de sortie.
    #La fonction d'activation 'softmax' est utilisée ici pour obtenir des probabilités pour chaque classe.
    # La somme de toutes les probabilités pour toutes les classes est égale à 1.

    # la sortie de la couche de Global Average Pooling est utilisée comme entrée pour la couche Dense.

    # La variable predictions représente maintenant la sortie de la couche Dense.

    # Elle contient les probabilités pour chaque classe
    # permet de prédire la classe de l'image en fonction des probabilités les plus élevées.

    # créer un modèle séquentiel personnalisée avec Keras de TensorFlow
    updated_model = tf.keras.Model(inputs=model.input, outputs=predictions)
    # images en entrée du modèle et probabilités pour chaque classe en sortie.

    # Define the training loss function and optimizer.
    # La fonction de perte calcule la différence entre les probabilités prédites par le modèle et les vraies étiquettes de classe
    loss_function = CategoricalCrossentropy()

    # L'optimiseur SGD (Stochastic Gradient Descent) est utilisé pour ajuster les poids du modèle afin de minimiser la fonction de perte.
    optimizer = SGD(learning_rate=0.001)

    # Compilation du modèle mis à jour
    updated_model.compile(loss=loss_function, optimizer=optimizer, metrics=[CategoricalAccuracy()])
    updated_model.fit(train_data, epochs=20, validation_data=validation_data)

    # création de liste pour stocker les valeurs de perte d'entraînement et de validation.
    train_loss_values = []
    val_loss_values = []

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")  #"Epoch 1/20", "Epoch 2/20", etc.
        start_time = time.time() #le temps de début de l'époque actuelle


        # entrainer le modéle pour une époque
        # Pendant l'entraînement, la perte et l'exactitude, seront enregistrées dans l'objet history.
        history = updated_model.fit(train_data, epochs=1, validation_data=validation_data)

        # Enregistre la perte du training and validation pour l'affichage
        train_loss_values.append(history.history['loss'][0])

        # enregistre la valeur de perte de validation pour l'époque actuelle dans la liste val_loss_values
        # val_loss est la clé qui correspond à la perte de validation pour l'époque actuelle
        # L'index [0] renvoie la valeur de la perte de validation pour cette époque spécifique.

        val_loss_values.append(history.history['val_loss'][0])

        end_time = time.time()
        #enregistre le temps de fin de l'époque d'entraînement
        #permet de calculer la durée totale de l'époque en soustrayant le temps de début (start_time) du temps de fin (end_time).

        print(f"Time taken for epoch: {end_time - start_time:.2f} seconds")




    print("Training Accuracy:", updated_model.history.history['categorical_accuracy'][-1])
    # affiche la dernière valeur d'exactitude (accuracy) obtenue
    # L'index -1 renvoie la dernière valeur enregistrée (correspondant à la dernière époque)
    # L'attribut history de l'objet updated_model contient un dictionnaire qui enregistre les métriques d'entraînement à chaque époque.

    print("Training Loss:", updated_model.history.history['loss'][-1])
    print("Validation Accuracy:", updated_model.history.history['val_categorical_accuracy'][-1])
    print("Validation Loss:", updated_model.history.history['val_loss'][-1])

    # Plot accuracies
    plt.plot(updated_model.history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(updated_model.history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model.
    updated_model.save('ResNet50_model.h5')















def main():
    dataset_dir = 'C:/Users/DELL/Desktop/projetimage/PlantVillage3'

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #le modèle est créé en utilisant les poids pré-entraînés sur le jeu de données ImageNet.
    # include_top=False signifie que la couche entièrement connectée (top layer) du modèle DenseNet121 ne sera pas incluse.
    # Nnous pouvons ajouter nos propres couches personnalisées pour la classification des classes du jeu de données personnalisé.
    # les images doivent être de taille 224x224 pixels avec 3 canaux de couleur (RGB).

    train_data, validation_data = load_dataset(dataset_dir)

    fine_tune_cnn(model, train_data, validation_data)

if __name__ == '__main__':
    main()
