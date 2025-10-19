#!/usr/bin/env python3
"""Transfer Learning on CIFAR-10 using MobileNetV2"""
import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for MobileNetV2"""
    X_p = X.astype('float32')
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Input: 32x32 images
    inputs = K.layers.Input(shape=(32, 32, 3))

    # Resize to 224x224 for pretrained network
    resize = K.layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(inputs)

    # Preprocess input according to MobileNetV2 requirements
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    x = K.layers.Lambda(preprocess_input)(resize)

    # Load base model (ImageNet weights, no top)
    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        input_shape=(224, 224, 3)
    )

    # Freeze base model initially
    base_model.trainable = False

    # Classification head
    x = K.layers.GlobalAveragePooling2D()(base_model.output)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Data augmentation for robustness
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    print("--- Training top layers ---")
    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              epochs=8,
              validation_data=(x_test, y_test))

    # Unfreeze top layers of the base model for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_cb = K.callbacks.ModelCheckpoint('cifar10.h5',
                                                save_best_only=True,
                                                monitor='val_accuracy')
    early_stop_cb = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=5,
                                              restore_best_weights=True)

    print("\n--- Fine-tuning full model ---")
    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              epochs=15,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint_cb, early_stop_cb])

    print("\n--- Saving final model ---")
    model.save('cifar10.h5')
