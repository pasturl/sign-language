import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
import itertools


log = logging.getLogger("Signem")
model_version = "v1"

def get_input_size_and_handle(model_name):
    model_handle_map = {
        "efficientnetv2-s":
            "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
        "efficientnetv2-m":
            "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2"}
    model_image_size_map = {
        "efficientnetv2-s": 384,
        "efficientnetv2-m": 480}
    model_handle = model_handle_map.get(model_name)
    pixels = model_image_size_map.get(model_name, 224)

    log.info(f"Selected model: {model_name} : {model_handle}")

    image_size = (pixels, pixels)
    log.info(f"Input size {image_size}")

    return image_size, model_handle


def build_dataset(subset, data_dir, image_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
              data_dir,
              validation_split=.20,
              subset=subset,
              label_mode="categorical",
              # Seed needs to provided when using validation_split and shuffle = True.
              # A fixed seed is used so that the validation set is stable across runs.
              seed=123,
              image_size=image_size,
              batch_size=1)

    return train_ds


def get_class_names(dataset):
    class_names = tuple(dataset.class_names)
    return class_names


def get_ds_size(dataset):
    dataset_size = dataset.cardinality().numpy()
    return dataset_size


def build_preprocessing_model():
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    preprocessing_model = tf.keras.Sequential([normalization_layer])
    return preprocessing_model, normalization_layer


def add_data_augmentation(model):
    model.add(
        tf.keras.layers.RandomRotation(40))
    model.add(
        tf.keras.layers.RandomTranslation(0, 0.2))
    model.add(
        tf.keras.layers.RandomTranslation(0.2, 0))
    # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
    # image sizes are fixed when reading, and then a random zoom is applied.
    # If all training inputs are larger than image_size, one could also use
    # RandomCrop with a batch size of 1 and rebatch later.
    model.add(
        tf.keras.layers.RandomZoom(0.2, 0.2))
    model.add(
        tf.keras.layers.RandomFlip(mode="horizontal"))
    return model


def preprocess_dataset(ds, model):
    ds = ds.map(lambda images, labels:
                            (model(images), labels))
    return ds


def build_model(model_handle, do_fine_tuning, class_names, image_size):
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=image_size + (3,)),
        hub.KerasLayer(model_handle, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(len(class_names),
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + image_size + (3,))
    model.summary()
    return model


def train_model(model_handle, do_fine_tuning,
                class_names, image_size, batch_size,
                train_size, valid_size,
                train_ds, val_ds, model_name):
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=image_size + (3,)),
        hub.KerasLayer(model_handle, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(len(class_names),
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + image_size + (3,))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])
    steps_per_epoch = train_size // batch_size
    validation_steps = valid_size // batch_size
    hist = model.fit(
        train_ds,
        epochs=10, steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps).history
    saved_model_path = f"./trained_models/signs_model_{model_name}_{model_version}"
    tf.keras.models.save_model(model, saved_model_path)
    plot_loss_history(hist)
    return model


def plot_loss_history(hist):
    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])

    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])


def optimize_model_size(model_name, train_ds):
    # @title Optimization settings
    optimize_lite_model = True  # @param {type:"boolean"}
    # @markdown Setting a value greater than zero enables
    # quantization of neural network activations. A few dozen is already a useful amount.
    num_calibration_examples = 60  # @param {type:"slider", min:0, max:1000, step:1}
    representative_dataset = None
    if optimize_lite_model and num_calibration_examples:
        # Use a bounded number of training examples without labels for calibration.
        # TFLiteConverter expects a list of input tensors, each with batch size 1.
        representative_dataset = lambda: itertools.islice(
            ([image[None, ...]] for batch, _ in train_ds for image in batch),
            num_calibration_examples)
    saved_model_path = f"./trained_models/signs_model_{model_name}_{model_version}"
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    if optimize_lite_model:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset:  # This is optional, see above.
            converter.representative_dataset = representative_dataset
    lite_model_content = converter.convert()

    with open(f"./trained_models/signs_model_{model_name}_{model_version}.tflite", "wb") as f:
        f.write(lite_model_content)
    print("Wrote %sTFLite model of %d bytes." %
          ("optimized " if optimize_lite_model else "", len(lite_model_content)))
