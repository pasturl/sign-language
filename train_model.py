import data
import model
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Signem")
# Download data from Mega or gdrive
# https://mega.nz/file/kJBDxLSL#zamibF1KPtgQFHn3RM0L1WBuhcBUvo0N0Uec9hczK_M
# https://drive.google.com/file/d/1C7k_m2m4n5VzI4lljMoezc-uowDEgIUh/view
path_train_videos = "./data/train_videos_sample/"
path_train_frames = "./data/train_frames_sample/"
model_name = "efficientnetv2-s"
batch_size = 256
do_data_augmentation = False
do_fine_tuning = False

log.info("Creating folder by category and moving videos from /all folder")
#data.move_videos_to_folder_categories(path_train_videos)

log.info("Converting videos to frames")
# Use video-to-frame.py and handsegment function from repo https://github.com/hthuwal/sign-language-gesture-recognition
#data.convert_video_to_frames(path_train_videos, path_train_frames)
data.convert_video_to_frames_landmarks(path_train_videos, path_train_frames)

log.info("Gettting model input size")
image_size, model_handle = model.get_input_size_and_handle(model_name)

log.info("Building training dataset")
train_ds = model.build_dataset("training", path_train_frames, image_size)

log.info("Getting class names of training data")
class_names_train = model.get_class_names(train_ds)

log.info("Getting dataset size of training data")
ds_size_train = model.get_ds_size(train_ds)

# Is necessary this step?
train_ds = train_ds.unbatch().batch(batch_size)
train_ds = train_ds.repeat()

log.info("Building preprocessing model")
preprocessing_model, normalization_layer = model.build_preprocessing_model()
if do_data_augmentation:
    log.info("Adding layers to preprocessing model for data augmentation ")
    preprocessing_model = model.add_data_augmentation(preprocessing_model)

log.info("Preprocessing training dataset")
train_ds = model.preprocess_dataset(train_ds, preprocessing_model)

log.info("Building validation dataset")
val_ds = model.build_dataset("validation", path_train_frames, image_size)

log.info("Getting dataset size of validation data")
ds_size_val = model.get_ds_size(val_ds)

# Is necessary this step?
val_ds = val_ds.unbatch().batch(batch_size)

log.info("Preprocessing validation dataset")
val_ds = model.preprocess_dataset(val_ds, normalization_layer)

log.info(f"Building model: {model_name}")
#model_built = model.build_model(model_handle, do_fine_tuning,
#                               class_names_train, image_size)

log.info("Training model")
model.train_model(model_handle, do_fine_tuning,
                  class_names_train, image_size,
                  batch_size,
                  ds_size_train, ds_size_val,
                  train_ds, val_ds, model_name)

log.info("Plot loss history")
#model.plot_loss_history(hist)


