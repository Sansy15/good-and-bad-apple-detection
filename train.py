import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

def build_model(input_shape=(224,224,3), dropout_rate=0.4):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout_rate)(x)
    # binary output
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    train_dir = os.path.join(args.dataset, "train")
    val_dir = os.path.join(args.dataset, "val")

    batch_size = args.batch_size
    img_size = (224, 224)

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       vertical_flip=False)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

    model = build_model(input_shape=(img_size[0], img_size[1], 3))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.save_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset directory with train/val subfolders")
    parser.add_argument("--save_path", default="models/model.h5", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    main(args)