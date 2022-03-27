import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from os import walk, listdir
import datetime


log = logging.getLogger("Signem")
model_version = "v9"
EPOCHS = 30
LEARNING_RATE = 0.1

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
        #tf.keras.layers.Dropout(rate=0.2),
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
        #tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(len(class_names),
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + image_size + (3,))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])
    steps_per_epoch = train_size // batch_size
    validation_steps = valid_size // batch_size
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1, write_images=False,)
    file_writer = tf.summary.create_file_writer(log_dir + '/cm')

    # def log_confusion_matrix(epoch, logs):
    #     # Use the model to predict the values from the validation dataset.
    #     test_pred_raw = model.predict(val_ds)
    #     test_pred = np.argmax(test_pred_raw, axis=1)
    #
    #     # Calculate the confusion matrix.
    #     cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    #     # Log the confusion matrix as an image summary.
    #     figure = plot_confusion_matrix(cm, class_names=class_names)
    #     cm_image = plot_to_image(figure)
    #
    #     # Log the confusion matrix as an image summary.
    #     with file_writer_cm.as_default():
    #         tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    #
    # # Define the per-epoch callback.
    # cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    hist = model.fit(
        train_ds,
        epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
          callbacks=[tensorboard_callback]).history
    print("Train eval")
    model.evaluate(train_ds, batch_size=batch_size, steps=steps_per_epoch)
    print("Validation eval")
    model.evaluate(val_ds, batch_size=batch_size, steps=steps_per_epoch)

    saved_model_path = f"./trained_models/signs_model_{model_name}_{model_version}"
    tf.keras.models.save_model(model, saved_model_path)
    plot_loss_history(hist)
    y_labels = []
    predictions = []
    for x, y in val_ds:
        prediction_x = model.predict(x)
        predict_label = np.argmax(prediction_x, axis=1)
        predictions.append(predict_label)
        y_true = np.argmax(y, axis=1)
        y_labels.append(y_true)

    y_labels = flatten(y_labels)
    predictions = flatten(predictions)
    df_eval = pd.DataFrame({'y_true': y_labels,
                             'y_pred': predictions})

    df_eval.to_csv("ds_pred_in_train.csv", index=False, sep=";")
    y_true = df_eval["y_true"].astype(int)
    y_pred = df_eval["y_pred"].astype(int)
    accuracy_train = accuracy_score(y_true, y_pred)
    print(f"Train accuracy {accuracy_train}")
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
    plt.savefig("plot_train_history.png")


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


def eval_model(model_trained, data_path):
    categories = pd.read_csv("category_mapping.csv", sep=";")
    categories["ID_str"] = categories["ID"].apply(lambda x: str(x).zfill(3))
    categories_to_ID = dict(zip(categories["ID_str"], categories["Name"]))
    video_categories = listdir(data_path)
    image_files = []
    for category in video_categories:
        category_path = data_path + "/" + category
        image_files_category = listdir(category_path)
        image_files = image_files + image_files_category
    dataset = pd.DataFrame()
    dataset["frames_files"] = image_files
    dataset["category_ID"] = dataset["frames_files"].apply(lambda x: x[:3])
    dataset["category_name"] = dataset["category_ID"].apply(lambda x: categories_to_ID[x])
    dataset["person_ID"] = dataset["frames_files"].apply(lambda x: x[4:7])

    width = 384
    height = 384
    dataset_tf = tf.keras.preprocessing.image_dataset_from_directory(data_path,
                                                                     image_size=(width, height),
                                                                     labels='inferred',
                                                                     label_mode='categorical')
    preprocessing_model, normalization_layer = build_preprocessing_model()
    dataset_tf = preprocess_dataset(dataset_tf, preprocessing_model)

    predictions = []
    y_labels = []
    for x, y in dataset_tf:
        prediction_x = model_trained.predict(x)
        predict_label = np.argmax(prediction_x, axis=1)
        predictions.append(predict_label)
        y_true = np.argmax(y, axis=1)
        y_labels.append(y_true)

    y_labels = flatten(y_labels)
    predictions = flatten(predictions)
    #predict_prob = model_trained.predict(dataset_tf)
    #predict = np.argmax(predict_prob, axis=1)
    tf.math.confusion_matrix(labels=y_labels, predictions=predictions).numpy()

    #y_true = dataset["category_ID"].astype(int).values
    df_eval = pd.DataFrame({'y_true': y_labels,
                             'y_pred': predictions})
    y_true = df_eval["y_true"]
    y_pred = df_eval["y_pred"]
    accuracy_train = accuracy_score(y_true, y_pred)
    print(f"Train accuracy {accuracy_train}")
    df_eval.to_csv("ds_pred.csv", index=False, sep=";")

    # Confusion matrix for actual and predicted values.
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    # Plotting the confusion matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig("heatmap.png")

def flatten(t):
    return [item for sublist in t for item in sublist]