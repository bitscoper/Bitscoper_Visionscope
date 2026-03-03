#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# By Abdullah As-Sadeed

from ultralytics import YOLO
import cv2
import PIL
import streamlit as st
import tempfile

import settings

if __name__ == "__main__":
    TITLE = "Bitscoper Visionscope"
    ICON = "🔬"

    OBJECT_DETECTION_MODEL = "Object Detection"
    OBB_OBJECT_DETECTION_MODEL = "Oriented Bounding Boxes (OBB) Object Detection"
    OBJECT_SEGMENTATION_MODEL = "Object Segmentation"
    POSE_DETECTION_MODEL = "Pose Detection"
    MODELS = [
        OBJECT_DETECTION_MODEL,
        OBB_OBJECT_DETECTION_MODEL,
        OBJECT_SEGMENTATION_MODEL,
        POSE_DETECTION_MODEL,
    ]

    NANO_WEIGHT = "Nano"
    SMALL_WEIGHT = "Small"
    MEDIUM_WEIGHT = "Medium"
    LARGE_WEIGHT = "Large"
    EXTRA_LARGE_WEIGHT = "Extra Large"
    WEIGHTS = [
        NANO_WEIGHT,
        SMALL_WEIGHT,
        MEDIUM_WEIGHT,
        LARGE_WEIGHT,
        EXTRA_LARGE_WEIGHT,
    ]

    IMAGE_FILE_SOURCE = "Image File"
    VIDEO_FILE_SOURCE = "Video File"
    WEBCAM_SOURCE = "Webcam"
    RTSP_SOURCE = "RTSP"
    SOURCES = [IMAGE_FILE_SOURCE, VIDEO_FILE_SOURCE, WEBCAM_SOURCE, RTSP_SOURCE]

    IMAGE_FILE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
    VIDEO_FILE_EXTENSIONS = ["mp4"]

    ASPECT_RATIO_16_9 = "16:9"
    ASPECT_RATIO_4_3 = "4:3"
    ASPECT_RATIO_CUSTOM = "Custom"
    ASPECT_RATIOS = [ASPECT_RATIO_16_9, ASPECT_RATIO_4_3, ASPECT_RATIO_CUSTOM]

    VIDEO_HEIGHTS = [144, 240, 360, 480, 720, 1080, 1440, 2160]

    def select_video_size(width=None, height=None):
        combined_widths = []

        for aspect_ratio in ASPECT_RATIOS:
            if aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9

            elif aspect_ratio == ASPECT_RATIO_4_3:
                aspect_ratio = 4 / 3

            elif aspect_ratio == ASPECT_RATIO_CUSTOM:
                continue

            else:
                st.error(body="Failed to determine aspect ratio!")

            for height in VIDEO_HEIGHTS:
                width = int(height * aspect_ratio)
                combined_widths.append(width)

        minimum_height = min(VIDEO_HEIGHTS)
        maximum_height = max(VIDEO_HEIGHTS)
        minimum_width = min(combined_widths)
        maximum_width = max(combined_widths)

        aspect_ratio = st.sidebar.radio(
            label="Select Aspect Ratio", options=ASPECT_RATIOS, horizontal=True
        )

        if (aspect_ratio == ASPECT_RATIO_16_9) or (aspect_ratio == ASPECT_RATIO_4_3):
            if aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9

            elif aspect_ratio == ASPECT_RATIO_4_3:
                aspect_ratio = 4 / 3

            else:
                st.error(body="Failed to determine aspect ratio!")

            widths = []
            resolutions = []

            for height in VIDEO_HEIGHTS:
                width = int(height * aspect_ratio)
                widths.append(width)
                resolutions.append(f"{height}p: {width} x {height}")

            resolution = st.sidebar.selectbox(
                label="Select Resolution", options=resolutions
            )

            width, height = map(int, resolution.split(":")[1].strip().split("x"))

        elif aspect_ratio == ASPECT_RATIO_CUSTOM:
            width = int(
                st.sidebar.number_input(
                    label="Set Video Width",
                    format="%d",
                    min_value=minimum_width,
                    max_value=maximum_width,
                    step=1,
                    value=settings.DEFAULT_VIDEO_WIDTH,
                )
            )

            height = int(
                st.sidebar.number_input(
                    label="Set Video Height",
                    format="%d",
                    min_value=minimum_height,
                    max_value=maximum_height,
                    step=1,
                    value=settings.DEFAULT_VIDEO_HEIGHT,
                )
            )

        else:
            st.error(body="Failed to select aspect ratio!")

        return width, height

    def display_result_frames(streamlit_frame, source_frame, width):
        if tracker == "No":
            model_output = model(source=source_frame, conf=confidence, verbose=False)

        elif (tracker == "bytetrack.yaml") or (tracker == "botsort.yaml"):
            model_output = model.track(
                source=source_frame,
                conf=confidence,
                persist=True,
                tracker=tracker,
                verbose=False,
            )

        else:
            st.error(body="Failed to determine tracker!")

        plotted_frame = model_output[0].plot(
            boxes=True,
            conf=True,
            kpt_line=True,
            labels=True,
            masks=True,
            probs=True,
        )

        streamlit_frame.image(
            image=plotted_frame,
            caption="Result Video",
            channels="BGR",
            width=width,
            output_format="auto",
        )

    st.set_page_config(
        page_title=TITLE,
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(body=TITLE, anchor=False)

    st.sidebar.header(body="Model Settings")

    model_type = st.sidebar.radio(label="Select Model", options=MODELS)

    model_weight = st.sidebar.selectbox(label="Select Weight", options=WEIGHTS)

    confidence = (
        float(
            st.sidebar.slider(
                label="Set Model Confidence",
                min_value=1,
                max_value=100,
                step=1,
                value=settings.DEFAULT_CONFIDENCE,
            )
        )
        / 100
    )

    tracker = st.sidebar.selectbox(
        label="Select Tracker", options=["bytetrack.yaml", "botsort.yaml", "No"]
    )

    if model_type == OBJECT_DETECTION_MODEL:
        model_suffix = ""
        task = "detect"

    elif model_type == OBB_OBJECT_DETECTION_MODEL:
        model_suffix = "-obb"
        task = "obb"

    elif model_type == OBJECT_SEGMENTATION_MODEL:
        model_suffix = "-seg"
        task = "segment"

    elif model_type == POSE_DETECTION_MODEL:
        model_suffix = "-pose"
        task = "pose"

    else:
        st.error(body="Failed to select model!")

    if model_weight == NANO_WEIGHT:
        model_weight_suffix = "n"

    elif model_weight == SMALL_WEIGHT:
        model_weight_suffix = "s"

    elif model_weight == MEDIUM_WEIGHT:
        model_weight_suffix = "m"

    elif model_weight == LARGE_WEIGHT:
        model_weight_suffix = "l"

    elif model_weight == EXTRA_LARGE_WEIGHT:
        model_weight_suffix = "x"

    else:
        st.error(body="Failed to select weight!")

    model_path = (
        str(settings.MODEL_DIRECTORY)
        + "/yolo26"
        + model_weight_suffix
        + model_suffix
        + ".pt"
    )

    try:
        model = YOLO(model=model_path, task=task, verbose=False)

    except Exception as exception:
        st.error(
            body=f"Failed to load model!\nCheck the path: {model_path}: {exception}"
        )

    st.sidebar.header(body="Input Settings")

    source_radio = st.sidebar.radio(label="Select Source", options=SOURCES)

    if source_radio == IMAGE_FILE_SOURCE:
        source_image_file = st.sidebar.file_uploader(
            label="Select an Image File",
            type=IMAGE_FILE_EXTENSIONS,
            accept_multiple_files=False,
        )

        column_1, column_2 = st.columns(2)

        with column_1:
            try:
                if source_image_file is not None:
                    uploaded_image = PIL.Image.open(fp=source_image_file)

                    st.image(
                        image=source_image_file,
                        caption="Uploaded Image",
                        width="stretch",
                        output_format="auto",
                    )

            except Exception as exception:
                st.error(body=f"Error occurred while opening the image: {exception}")

        with column_2:
            if source_image_file is not None:
                if st.sidebar.button(label="Run"):
                    if tracker == "No":
                        resource = model(
                            model=uploaded_image, conf=confidence, verbose=False
                        )

                    else:
                        resource = model.track(
                            source=uploaded_image,
                            conf=confidence,
                            persist=True,
                            tracker=tracker,
                            verbose=False,
                        )

                    boxes = resource[0].boxes
                    plotted_resource = resource[0].plot(
                        boxes=True,
                        conf=True,
                        kpt_line=True,
                        labels=True,
                        masks=True,
                        probs=True,
                    )[:, :, ::-1]

                    st.image(
                        image=plotted_resource,
                        caption="Result Image",
                        width="stretch",
                        output_format="auto",
                    )

                    st.snow()

                    try:
                        with st.expander(label="Results"):
                            for box in boxes:
                                st.write(box.data)

                    except Exception as exception:
                        st.error(body=f"No image is uploaded yet: {exception}")

    elif source_radio == VIDEO_FILE_SOURCE:
        source_video_file = st.sidebar.file_uploader(
            label="Select a Video File",
            type=VIDEO_FILE_EXTENSIONS,
            accept_multiple_files=False,
        )

        if source_video_file is not None:
            temporary_file = tempfile.NamedTemporaryFile(delete=False)
            temporary_file.write(source_video_file.read())
            video_capture = cv2.VideoCapture(temporary_file.name)

            column_1, column_2 = st.columns(2)

            with column_1:
                st.video(data=source_video_file)

            with column_2:
                if st.sidebar.button(label="Run"):
                    try:
                        st_frame = st.empty()

                        while video_capture.isOpened():
                            success, image = video_capture.read()

                            if success:
                                display_result_frames(
                                    streamlit_frame=st_frame,
                                    source_frame=image,
                                    width="stretch",
                                )

                            else:
                                video_capture.release()
                                break

                    except Exception as exception:
                        st.error(label=f"Error loading video: {exception}")

                    finally:
                        temporary_file.close()

    elif source_radio == WEBCAM_SOURCE:
        source_webcam = int(
            st.sidebar.number_input(
                label="Set Webcam Serial",
                format="%d",
                min_value=0,
                step=1,
                value=settings.DEFAULT_WEBCAM_NUMBER,
            )
        )

        source_width, source_height = select_video_size()

        if st.sidebar.button(label="Run"):
            try:
                stop_button = st.sidebar.button(label="Stop", key="stop")

                video_capture = cv2.VideoCapture(source_webcam)
                video_capture.set(propId=3, value=source_width)
                video_capture.set(propId=4, value=source_height)

                st_frame = st.empty()

                while video_capture.isOpened():
                    success, image = video_capture.read()

                    if stop_button:
                        video_capture.release()
                        break

                    if success:
                        display_result_frames(
                            streamlit_frame=st_frame,
                            source_frame=image,
                            width="content",
                        )

                    else:
                        video_capture.release()
                        break

                video_capture.release()

            except Exception as exception:
                st.error(body=f"Error loading webcam stream: {exception}")

    elif source_radio == RTSP_SOURCE:
        source_rtsp = st.sidebar.text_input(
            label="Set RTSP Stream URL", placeholder="Write a RTSP Stream URL"
        )

        if st.sidebar.button(label="Run"):
            try:
                stop_button = st.sidebar.button("Stop", key="stop")

                video_capture = cv2.VideoCapture(source_rtsp)

                st_frame = st.empty()

                while video_capture.isOpened():
                    success, image = video_capture.read()

                    if stop_button:
                        video_capture.release()
                        break

                    if success:
                        display_result_frames(
                            streamlit_frame=st_frame,
                            source_frame=image,
                            width="content",
                        )

                    else:
                        video_capture.release()
                        break

                video_capture.release()

            except Exception as exception:
                st.error(body=f"Error loading RTSP stream: {exception}")

    else:
        st.error(body="Failed to select source!")

else:
    print("Run as\nstreamlit run main.py")

exit()
