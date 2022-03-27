import os
from os import listdir
from os.path import isfile, join, exists
import cv2
import handsegment as hs
from tqdm import tqdm
import logging
import mediapipe as mp
import numpy as np
import pandas as pd

hc = []
log = logging.getLogger("Signem")


def move_videos_to_folder_categories(path):
    '''
    Organize image in folders, one per class
    :param path:
    :return:
    '''
    video_files = [f for f in listdir(path+"all") if isfile(join(path+"all", f))]
    file_categories = [i[:3] for i in video_files]
    categories = list(set(file_categories))
    for cat in categories:
        os.mkdir(path + cat)
    for file in listdir(path+"all"):
        os.replace(path +"all" + "/" + file, path + file[:3] + "/" + file)


def convert_video_to_frames_bw(path_videos, path_frames):
    rootPath = os.getcwd()
    majorData = os.path.abspath(path_frames)

    if not exists(majorData):
        os.makedirs(majorData)

    path_videos = os.path.abspath(path_videos)

    os.chdir(path_videos)
    gestures = os.listdir(os.getcwd())
    log.info("Source Directory containing gestures: %s" % (path_videos))
    log.info("Destination Directory containing frames: %s\n" % (majorData))

    for gesture in tqdm(gestures, unit='actions', ascii=True):
        gesture_path = os.path.join(path_videos, gesture)
        os.chdir(gesture_path)

        gesture_frames_path = os.path.join(majorData, gesture)
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]

        for video in tqdm(videos, unit='videos', ascii=True):
            name = os.path.abspath(video)
            cap = cv2.VideoCapture(name)  # capturing input video
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lastFrame = None

            os.chdir(gesture_frames_path)
            count = 0

            # assumption only first 200 frames are important
            while count < 201:
                ret, frame = cap.read()  # extract frame
                if ret is False:
                    break
                framename = os.path.splitext(video)[0]
                framename = framename + "_frame_" + str(count) + ".jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])

                if not os.path.exists(framename):
                    frame = hs.handsegment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1

            # repeat last frame untill we get 200 frames
            while count < 201:
                framename = os.path.splitext(video)[0]
                framename = framename + "_frame_" + str(count) + ".jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])
                if not os.path.exists(framename):
                    cv2.imwrite(framename, lastFrame)
                count += 1

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    os.chdir(rootPath)


def convert_video_to_frames_hands_landmarks(path_videos, path_frames):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    rootPath = os.getcwd()
    majorData = os.path.abspath(path_frames)

    if not exists(majorData):
        os.makedirs(majorData)

    path_videos = os.path.abspath(path_videos)

    os.chdir(path_videos)
    gestures = os.listdir(os.getcwd())
    log.info("Source Directory containing gestures: %s" % (path_videos))
    log.info("Destination Directory containing frames: %s\n" % (majorData))
    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        for gesture in tqdm(gestures, unit='actions', ascii=True):
            gesture_path = os.path.join(path_videos, gesture)
            os.chdir(gesture_path)

            gesture_frames_path = os.path.join(majorData, gesture)
            if not os.path.exists(gesture_frames_path):
                os.makedirs(gesture_frames_path)

            videos = os.listdir(os.getcwd())
            videos = [video for video in videos if(os.path.isfile(video))]

            for video in tqdm(videos, unit='videos', ascii=True):
                name = os.path.abspath(video)
                cap = cv2.VideoCapture(name)  # capturing input video
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                os.chdir(gesture_frames_path)
                count = 0

                # assumption only first 200 frames are important
                while count < frameCount:
                    ret, frame = cap.read()  # extract frame
                    if ret is False:
                        break
                    framename = os.path.splitext(video)[0]
                    framename = framename + "_frame_" + str(count) + ".jpeg"
                    hc.append([join(gesture_frames_path, framename), gesture, frameCount])

                    if not os.path.exists(framename):
                        # frame = hs.handsegment(frame)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(frame)
                        frame = np.zeros(frame.shape)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())
                        # Flip the image horizontally for a selfie-view display.

                        cv2.imwrite(framename, frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    count += 1

                os.chdir(gesture_path)
                cap.release()
                cv2.destroyAllWindows()

        os.chdir(rootPath)


def convert_video_to_frames_holistic_landmarks(path_videos, path_frames):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    rootPath = os.getcwd()
    majorData = os.path.abspath(path_frames)

    if not exists(majorData):
        os.makedirs(majorData)

    path_videos = os.path.abspath(path_videos)

    os.chdir(path_videos)
    gestures = os.listdir(os.getcwd())
    log.info("Source Directory containing gestures: %s" % (path_videos))
    log.info("Destination Directory containing frames: %s\n" % (majorData))
    with mp_holistic.Holistic(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        for gesture in tqdm(gestures, unit='actions', ascii=True):
            gesture_path = os.path.join(path_videos, gesture)
            os.chdir(gesture_path)

            gesture_frames_path = os.path.join(majorData, gesture)
            if not os.path.exists(gesture_frames_path):
                os.makedirs(gesture_frames_path)

            videos = os.listdir(os.getcwd())
            videos = [video for video in videos if(os.path.isfile(video))]

            for video in tqdm(videos, unit='videos', ascii=True):
                name = os.path.abspath(video)
                cap = cv2.VideoCapture(name)  # capturing input video
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                os.chdir(gesture_frames_path)
                count = 0

                # assumption only first 200 frames are important
                while count < frameCount:
                    ret, frame = cap.read()  # extract frame
                    if ret is False:
                        break
                    framename = os.path.splitext(video)[0]
                    framename = framename + "_frame_" + str(count).zfill(3) + ".jpeg"
                    hc.append([join(gesture_frames_path, framename), gesture, frameCount])

                    if not os.path.exists(framename):
                        # frame = hs.handsegment(frame)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(frame)
                        frame = np.zeros(frame.shape)
                        if results.face_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                results.face_landmarks,
                                mp_holistic.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                mp_holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                results.left_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                results.right_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                        cv2.imwrite(framename, frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    count += 1

                os.chdir(gesture_path)
                cap.release()
                cv2.destroyAllWindows()

        os.chdir(rootPath)


def blend(list_images):  # Blend images equally.

    equal_fraction = 1.0  # / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output


def blend_frames_to_one(path_input, path_output):
    categories = pd.read_csv("category_mapping.csv", sep=";")
    categories["ID_str"] = categories["ID"].apply(lambda x: str(x).zfill(3))
    categories_to_ID = dict(zip(categories["ID_str"], categories["Name"]))


    video_categories = listdir(path_input)
    image_files = []
    for category in video_categories:
        category_path = path_input + "/" + category
        image_files_category = listdir(category_path)
        image_files = image_files + image_files_category

    dataset = pd.DataFrame()
    dataset["frames_files"] = image_files
    dataset["category_ID"] = dataset["frames_files"].apply(lambda x: x[:3])
    dataset["category_name"] = dataset["category_ID"].apply(lambda x: categories_to_ID[x])
    dataset["person_ID"] = dataset["frames_files"].apply(lambda x: x[4:7])
    dataset["video_ID"] = dataset["frames_files"].apply(lambda x: x[:11])
    dataset["video_path"] = path_input + "/" + dataset["category_ID"] + "/" + dataset["frames_files"]
    dataset["frame_ID"] = dataset["frames_files"].apply(lambda x: int(x.split("frame_")[1].split(".")[0]))
    MAX_SEQ_LENGTH = dataset["frame_ID"].max() + 1
    print(MAX_SEQ_LENGTH)

    total_videos = len(dataset["video_ID"].unique())
    n_video = 0
    for video in dataset["video_ID"].unique():
        print(f"{n_video}/{total_videos}")
        list_images = []
        one_frame_path = path_output + video[:3] + "/"
        if not os.path.exists(one_frame_path):
            os.makedirs(one_frame_path)
        for frame in dataset[dataset["video_ID"] == video].sort_values("frame_ID")["video_path"]:
            img = cv2.imread(frame)
            list_images.append(img)
        output_images_blend = blend(list_images)
        output_path_video = path_output + video[:3] + "/" + video + ".jpeg"
        cv2.imwrite(output_path_video, output_images_blend)
        n_video += 1
