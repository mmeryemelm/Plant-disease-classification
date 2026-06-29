import time
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(dataset_dir, image_size=(224, 224), batch_size=32):
    """
    Load training and validation data with augmentation
    
    Args:
        dataset_dir: Path to dataset folder with class subfolders
        image_size: Target image dimensions
        batch_size: Batch size for training
    
    Returns:
        train_data, validation_data generators
    """
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Data augmentation for training
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
    
    # Load training data
    train_data = data_generator.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
        subset='training'
    )
    
    # Load validation data
    validation_data = data_generator.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123,
        subset='validation'
    )
    
    # Get class labels
    class_labels = sorted(train_data.class_indices, key=train_data.class_indices.get)
    train_data.class_labels = class_labels
    validation_data.class_labels = class_labels
    
    print(f"✓ Dataset loaded: {len(class_labels)} classes")
    print(f"  Classes: {', '.join(class_labels[:5])}{'...' if len(class_labels) > 5 else ''}")
    
    return train_data, validation_data


def build_model(num_classes, input_shape=(224, 224, 3)):
    """
    Build ResNet50 model with transfer learning
    
    Args:
        num_classes: Number of disease classes
        input_shape: Input image shape
    
    Returns:
        Compiled Keras model
    """
    
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze most layers, unfreeze last 30 for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer - DYNAMIC number of classes
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Build full model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy', CategoricalAccuracy()]
    )
    
    print(f"✓ Model built for {num_classes} classes")
    return model


def train_model(model, train_data, validation_data, epochs=10, model_name='resnet50'):
    """
    Train the model
    
    Args:
        model: Compiled Keras model
        train_data: Training data generator
        validation_data: Validation data generator
        epochs: Number of epochs
        model_name: Name for saving
    
    Returns:
        Training history
    """
    
    # Create callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{model_name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train
    print("\n🔄 Training started...")
    start_time = time.time()
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Training completed in {elapsed/60:.2f} minutes")
    
    return history


def evaluate_model(model, validation_data, model_name='resnet50'):
    """
    Evaluate model on validation set
    """
    
    print("\n📊 Evaluating model...")
    loss, accuracy, cat_accuracy = model.evaluate(validation_data, verbose=0)
    
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    return loss, accuracy


def plot_history(history, model_name='resnet50'):
    """
    Plot training history
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png', dpi=150, bbox_inches='tight')
    print(f"✓ History plot saved as {model_name}_history.png")
    plt.show()


def main():
    """
    Main training pipeline
    """
    
    # Configuration
    DATASET_DIR = 'plant_disease_dataset'  # Change to your dataset path
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    MODEL_NAME = 'resnet50'
    
    print("="*60)
    print("Plant Disease Classification - ResNet50")
    print("="*60)
    
    # Check dataset
    if not os.path.exists(DATASET_DIR):
        print(f"✗ Dataset directory not found: {DATASET_DIR}")
        print("Create folder structure:")
        print(f"  {DATASET_DIR}/")
        print(f"    ├── class1/")
        print(f"    ├── class2/")
        print(f"    └── class3/")
        return
    
    try:
        # Load data
        print("\n📂 Loading dataset...")
        train_data, validation_data = load_dataset(
            DATASET_DIR, 
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE
        )
        
        # Get dynamic number of classes
        num_classes = len(train_data.class_indices)
        print(f"Found {num_classes} disease classes")
        
        # Build model
        print("\n🏗 Building model...")
        model = build_model(num_classes=num_classes, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        
        # Train
        print("\n📈 Training...")
        history = train_model(
            model,
            train_data,
            validation_data,
            epochs=EPOCHS,
            model_name=MODEL_NAME
        )
        
        # Evaluate
        loss, accuracy = evaluate_model(model, validation_data, MODEL_NAME)
        
        # Plot results
        plot_history(history, MODEL_NAME)
        
        # Save final model
        model.save(f'{MODEL_NAME}_final.h5')
        print(f"✓ Model saved as {MODEL_NAME}_final.h5")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Final Accuracy: {accuracy*100:.2f}%")
        print("="*60)
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
