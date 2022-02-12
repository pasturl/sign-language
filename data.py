import os
from os import listdir
from os.path import isfile, join, exists
import cv2
import handsegment as hs
import argparse
from tqdm import tqdm
import logging

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


def convert_video_to_frames(path_videos, path_frames):
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
