

!pip install tensorflow tensorflow-datasets tensorboard

import tensorflow as tf

# Vérifier la disponibilité du GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

from tensorflow.keras.datasets import cifar10

# Charger les données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisation des données
x_train, x_test = x_train / 255.0, x_test / 255.0

from tensorflow.keras import layers

# Définir le GhostModule
def ghost_module(x, out_channels, kernel_size=(3, 3), strides=(1, 1)):
    x1 = layers.Conv2D(out_channels // 2, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False)(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x = layers.Concatenate()([x1, x2])
    return x
	
# Définir le modèle GhostNet
def build_ghostnet(input_shape=(32, 32, 3), num_classes=10, kernel_size=(3, 3), strides=(1, 1)):
    inputs = layers.Input(shape=input_shape)

    # Première couche de convolution
    x = layers.Conv2D(32, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Empiler plusieurs GhostModules
    x = ghost_module(x, 64, kernel_size, strides)  # Premier GhostModule
    x = ghost_module(x, 128, kernel_size, strides) # Deuxième GhostModule
    x = ghost_module(x, 256, kernel_size, strides) # Troisième GhostModule
    x = ghost_module(x, 512, kernel_size, strides) # Quatrième GhostModule

    # Ajouter une couche Depthwise Separable
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classificateur final
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Créer le modèle
    model = tf.keras.Model(inputs, x)
    return model

# Compiler le modèle GhostNet
ghostnet_model = build_ghostnet(input_shape=(32, 32, 3), num_classes=10)
ghostnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Hyperparamètres à tester
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]

# Matrice pour stocker les résultats
import numpy as np
results_grid = np.zeros((len(learning_rates), len(batch_sizes)))

# Boucle pour tester chaque combinaison d'hyperparamètres
for i, lr in enumerate(learning_rates):
    for j, bs in enumerate(batch_sizes):
        # Compiler le modèle avec les nouveaux hyperparamètres
        ghostnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

        # Entraîner le modèle
        history = ghostnet_model.fit(
            x_train, y_train,
            epochs=10,
            batch_size=bs,
            validation_data=(x_test, y_test),
            verbose=0
        )

        # Stocker la meilleure précision de validation
        results_grid[i, j] = max(history.history['val_accuracy'])

# Afficher les résultats
print(results_grid)


from tensorflow.keras.callbacks import TensorBoard

# Supprimer les anciens logs pour éviter les conflits
!rm -rf ./logs

# Définir le callback TensorBoard
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True, write_images=True)

# Entraîner le modèle GhostNet sur GPU
best_lr = 0.001
best_bs = 32
ghostnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

ghostnet_history = ghostnet_model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    batch_size=best_bs,
    callbacks=[tensorboard_callback]  # Ajout de TensorBoard ici !
)

# Afficher la précision finale
ghostnet_test_loss, ghostnet_test_acc = ghostnet_model.evaluate(x_test, y_test)
print(f"GhostNet Test Accuracy: {ghostnet_test_acc * 100:.2f}%")


%load_ext tensorboard
%tensorboard --logdir ./logs
