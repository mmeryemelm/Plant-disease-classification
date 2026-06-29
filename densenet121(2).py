import time
import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt


def load_dataset(dataset_dir, image_size=(224, 224), batch_size=32):
    """Load dataset with augmentation"""
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    train_data = data_generator.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
        subset='training'
    )
    
    validation_data = data_generator.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
        subset='validation'
    )
    
    class_labels = sorted(train_data.class_indices, key=train_data.class_indices.get)
    train_data.class_labels = class_labels
    validation_data.class_labels = class_labels
    
    print(f"✓ Dataset loaded: {len(class_labels)} classes")
    return train_data, validation_data


def build_model(num_classes, input_shape=(224, 224, 3)):
    """Build DenseNet121 model with transfer learning"""
    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Unfreeze last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # DYNAMIC classes
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print(f"✓ DenseNet121 model built for {num_classes} classes")
    return model


def train_and_evaluate(dataset_dir='plant_disease_dataset'):
    """Complete training pipeline"""
    
    print("\n" + "="*60)
    print("Plant Disease Classification - DenseNet121")
    print("="*60)
    
    # Load data
    train_data, validation_data = load_dataset(dataset_dir)
    num_classes = len(train_data.class_indices)
    
    # Build and train
    model = build_model(num_classes=num_classes)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'densenet121_best.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    print("\n🔄 Training...")
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=validation_data,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(validation_data, verbose=0)
    print(f"\n✓ Validation Accuracy: {accuracy*100:.2f}%")
    
    # Save
    model.save('densenet121_final.h5')
    
    return model, history


if __name__ == "__main__":
    model, history = train_and_evaluate()
