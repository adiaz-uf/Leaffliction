import os
import sys
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    # Load training subset
    train_data = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    # Load validation subset
    val_data = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    # Extract class names
    class_names = list(train_data.class_indices.keys())

    return train_data, val_data, class_names


def create_model(num_classes, img_size=(224, 224, 3)):
    base_model = MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights='imagenet'
    )

    # freeze pretrained layers
    base_model.trainable = False

    # build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_data, val_data, epochs=20):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'leaf_disease_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_and_save(model, val_data, train_data):
    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    if val_acc >= .90:
        print(f"Model meets 90% accuracy requirement!: {val_acc}")
    else:
        print("Model does not meet 90% accuracy requirement.")

    class_indices = train_data.class_indices
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    print("\nSaved:")
    print("- Model as 'leaf_disease_model.h5'")
    print("- Class indices as 'class_indices.json'\n")


def main(folder_path):
    print("\n-- Leaffliction Training --\n")

    print("\n[1/4] Loading dataset\n")
    train_data, val_data, class_names = load_dataset(folder_path)
    print("\n[2/4] Creating model\n")
    num_classes = len(class_names)
    model = create_model(num_classes)
    model.summary()

    print("\n[3/4] Training model\n")
    train_model(model, train_data, val_data, epochs=2)

    print(
        "\n[4/4] Training complete. Model saved as 'leaf_disease_model.h5'\n")
    evaluate_and_save(model, val_data, train_data)

    print("\n --Training Complete--\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Error! Usage is: ./train.py <folder_path>')
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print('Error! Path must be a directory')
        sys.exit(2)

    main(folder_path)
