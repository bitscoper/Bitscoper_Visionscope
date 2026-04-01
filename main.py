#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""By Abdullah As-Sadeed"""

from pathlib import Path
import sys
import tempfile

from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st

import settings

if __name__ == "__main__":
    TITLE = "Bitscoper Visionscope"
    ICON = "🔬"

    OBJECT_DETECTION_MODEL = "Object Detection"
    OBB_OBJECT_DETECTION_MODEL = (
        "Oriented Bounding Boxes (OBB) Object Detection"
    )
    OBJECT_SEGMENTATION_MODEL = "Object Segmentation"
    POSE_DETECTION_MODEL = "Pose Detection"
    MODELS = [
        OBJECT_DETECTION_MODEL,
        OBB_OBJECT_DETECTION_MODEL,
        OBJECT_SEGMENTATION_MODEL,
        POSE_DETECTION_MODEL,
    ]

    NANO_MODEL_WEIGHT = "Nano"
    SMALL_MODEL_WEIGHT = "Small"
    MEDIUM_MODEL_WEIGHT = "Medium"
    LARGE_MODEL_WEIGHT = "Large"
    EXTRA_LARGE_MODEL_WEIGHT = "Extra Large"
    MODEL_WEIGHTS = [
        NANO_MODEL_WEIGHT,
        SMALL_MODEL_WEIGHT,
        MEDIUM_MODEL_WEIGHT,
        LARGE_MODEL_WEIGHT,
        EXTRA_LARGE_MODEL_WEIGHT,
    ]

    MODEL_SUFFIXES = {
        OBJECT_DETECTION_MODEL: "",
        OBB_OBJECT_DETECTION_MODEL: "-obb",
        OBJECT_SEGMENTATION_MODEL: "-seg",
        POSE_DETECTION_MODEL: "-pose",
    }

    MODEL_TASKS = {
        OBJECT_DETECTION_MODEL: "detect",
        OBB_OBJECT_DETECTION_MODEL: "obb",
        OBJECT_SEGMENTATION_MODEL: "segment",
        POSE_DETECTION_MODEL: "pose",
    }

    MODEL_WEIGHT_SUFFIXES = {
        NANO_MODEL_WEIGHT: "n",
        SMALL_MODEL_WEIGHT: "s",
        MEDIUM_MODEL_WEIGHT: "m",
        LARGE_MODEL_WEIGHT: "l",
        EXTRA_LARGE_MODEL_WEIGHT: "x",
    }

    BYTETRACK_TRACKER = "bytetrack.yaml"
    BOTSORT_TRACKER = "botsort.yaml"
    NO_TRACKER = "No"
    TRACKERS = [BYTETRACK_TRACKER, BOTSORT_TRACKER, NO_TRACKER]

    SOURCE_IMAGE_FILE = "Image File"
    SOURCE_VIDEO_FILE = "Video File"
    SOURCE_WEBCAM_STREAM = "Webcam Stream"
    SOURCE_RTSP_STREAM = "RTSP Stream"
    SOURCES = [
        SOURCE_IMAGE_FILE,
        SOURCE_VIDEO_FILE,
        SOURCE_WEBCAM_STREAM,
        SOURCE_RTSP_STREAM,
    ]

    IMAGE_FILE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
    VIDEO_FILE_EXTENSIONS = ["mp4"]

    ASPECT_RATIO_16_9 = "16:9"
    ASPECT_RATIO_4_3 = "4:3"
    CUSTOM_ASPECT_RATIO = "Custom"
    ASPECT_RATIOS = [ASPECT_RATIO_16_9, ASPECT_RATIO_4_3, CUSTOM_ASPECT_RATIO]

    WEBCAM_STREAM_HEIGHTS = [144, 240, 360, 480, 720, 1080, 1440, 2160]

    def select_webcam_stream_size() -> tuple[int, int]:
        """Selects Webcam Stream Size"""
        selected_aspect_ratio = st.sidebar.radio(
            label="Select Webcam Stream Aspect Ratio",
            options=ASPECT_RATIOS,
            key="webcam_stream_aspect_ratio",
            horizontal=True,
            help="Select Webcam Stream Aspect Ratio",
        )

        if selected_aspect_ratio in (ASPECT_RATIO_16_9, ASPECT_RATIO_4_3):
            resolutions = []
            resolution_map = {}

            if selected_aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9
            else:
                aspect_ratio = 4 / 3

            for height in WEBCAM_STREAM_HEIGHTS:
                width = int(height * aspect_ratio)
                resolution = f"{height}p: {width} x {height}"
                resolutions.append(resolution)
                resolution_map[resolution] = (width, height)

            selected_resolution = st.sidebar.selectbox(
                label="Select Webcam Stream Resolution",
                options=resolutions,
                key="webcam_stream_resolution",
                help="Select Webcam Stream Resolution",
            )

            selected_width, selected_height = resolution_map[
                selected_resolution
            ]

        else:
            widths = []

            for aspect_ratio_option in ASPECT_RATIOS:
                if aspect_ratio_option == ASPECT_RATIO_16_9:
                    aspect_ratio = 16 / 9

                elif aspect_ratio_option == ASPECT_RATIO_4_3:
                    aspect_ratio = 4 / 3

                else:
                    continue

                for height in WEBCAM_STREAM_HEIGHTS:
                    width = int(height * aspect_ratio)
                    widths.append(width)

            minimum_height = min(WEBCAM_STREAM_HEIGHTS)
            maximum_height = max(WEBCAM_STREAM_HEIGHTS)
            minimum_width = min(widths)
            maximum_width = max(widths)

            selected_width = int(
                st.sidebar.number_input(
                    label="Set Webcam Stream Width",
                    min_value=minimum_width,
                    max_value=maximum_width,
                    value=settings.DEFAULT_VIDEO_WIDTH,
                    step=1,
                    key="webcam_stream_width",
                    help="Set Webcam Stream Width",
                )
            )

            selected_height = int(
                st.sidebar.number_input(
                    label="Set Webcam Stream Height",
                    min_value=minimum_height,
                    max_value=maximum_height,
                    value=settings.DEFAULT_VIDEO_HEIGHT,
                    step=1,
                    key="webcam_stream_height",
                    help="Set Webcam Stream Height",
                )
            )

        return selected_width, selected_height

    def display_plotted_frames(streamlit_frame, source_frame, width) -> None:
        """Displays Plotted Frames"""
        if tracker == NO_TRACKER:
            model_output = model(
                source=source_frame, conf=confidence, verbose=False
            )

        elif tracker in (BYTETRACK_TRACKER, BOTSORT_TRACKER):
            model_output = model.track(
                source=source_frame,
                conf=confidence,
                persist=True,
                tracker=tracker,
                verbose=False,
            )

        else:
            st.error(body="Failed to determine tracker!")
            return

        plotted_frame = model_output[0].plot(
            boxes=True,
            conf=True,
            font_size=None,  # Scaled to Image Size
            kpt_line=True,
            labels=True,
            line_width=None,  # Scaled to Image Size
            masks=True,
            probs=True,
            txt_color=plot_text_color_bgr,
        )

        streamlit_frame.image(
            image=plotted_frame,
            caption="Plotted Video",
            channels="BGR",
            width=width,
            output_format="auto",
        )

    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_icon=ICON,
        page_title=TITLE,
    )

    st.title(anchor=False, body=TITLE, text_alignment="center")

    st.sidebar.header(body="Model Settings")

    model_type = st.sidebar.radio(
        disabled=False,
        help="Select Model",
        horizontal=False,
        key="model",
        label_visibility="visible",
        label="Select Model",
        options=MODELS,
    )

    model_weight = st.sidebar.selectbox(
        accept_new_options=False,
        disabled=False,
        help="Select Model Weight",
        key="model_weight",
        label_visibility="visible",
        label="Select Model Weight",
        options=MODEL_WEIGHTS,
        placeholder="Select Model Weight",
    )

    confidence = st.sidebar.slider(
        disabled=False,
        format="percent",
        help="Set Confidence",
        key="confidence",
        label_visibility="visible",
        label="Set Confidence",
        max_value=1.0,
        min_value=0.0,
        step=0.01,
        value=settings.DEFAULT_CONFIDENCE,
    )

    tracker = st.sidebar.selectbox(
        accept_new_options=False,
        disabled=False,
        help="Select Tracker",
        key="tracker",
        label_visibility="visible",
        label="Select Tracker",
        options=TRACKERS,
        placeholder="Select Tracker",
    )

    plot_text_color_hex = st.sidebar.color_picker(
        disabled=False,
        help="Pick Plot-Text Color",
        key="plot_text_color",
        label_visibility="visible",
        label="Pick Plot-Text Color",
    )
    plot_text_color_red = int(plot_text_color_hex[1:3], 16)
    plot_text_color_green = int(plot_text_color_hex[3:5], 16)
    plot_text_color_blue = int(plot_text_color_hex[5:7], 16)
    plot_text_color_bgr = (
        plot_text_color_blue,
        plot_text_color_green,
        plot_text_color_red,
    )

    root_path = Path(__file__).resolve().parent

    if root_path not in sys.path:
        sys.path.append(str(root_path))

    model_path = (
        Path(root_path.relative_to(Path.cwd()) / settings.MODEL_DIRECTORY)
        / f"yolo26{MODEL_WEIGHT_SUFFIXES[model_weight]}"
        f"{MODEL_SUFFIXES[model_type]}.pt"
    )

    try:
        model = YOLO(
            model=model_path, task=MODEL_TASKS.get(model_type), verbose=False
        )

    except Exception as exception:
        st.error(body=f"Failed to load model from {model_path}: {exception}")

    st.sidebar.header(body="Input Settings")

    source_radio = st.sidebar.radio(
        disabled=False,
        help="Select Source",
        horizontal=True,
        key="source",
        label_visibility="visible",
        label="Select Source",
        options=SOURCES,
    )

    if source_radio == SOURCE_IMAGE_FILE:
        source_image_file = st.sidebar.file_uploader(
            accept_multiple_files=False,
            disabled=False,
            help="Upload an Image File",
            key="source_image_file",
            label_visibility="visible",
            label="Upload an Image File",
            type=IMAGE_FILE_EXTENSIONS,
        )

        column_1, column_2 = st.columns(
            border=False, spec=2, vertical_alignment="top"
        )

        with column_1:
            try:
                if source_image_file is not None:
                    st.image(
                        caption="Source Image",
                        image=source_image_file,
                        output_format="auto",
                        width="stretch",
                    )

            except Exception as exception:
                st.error(
                    body=f"Error occurred while opening the image: {exception}"
                )

        with column_2:
            if source_image_file is not None:
                uploaded_image = Image.open(source_image_file)

                if tracker == NO_TRACKER:
                    resource = model(
                        source=uploaded_image, conf=confidence, verbose=False
                    )

                else:
                    resource = model.track(
                        source=np.array(uploaded_image)[:, :, ::-1],
                        conf=confidence,
                        persist=True,
                        tracker=tracker,
                        verbose=False,
                    )

                plotted_resource = resource[0].plot(
                    boxes=True,
                    conf=True,
                    font_size=None,  # Scaled to Image Size
                    kpt_line=True,
                    labels=True,
                    line_width=None,  # Scaled to Image Size
                    masks=True,
                    probs=True,
                    txt_color=plot_text_color_bgr,
                )[:, :, ::-1]

                st.image(
                    caption="Plotted Image",
                    image=plotted_resource,
                    output_format="auto",
                    width="stretch",
                )

                st.balloons()

                try:
                    with st.expander(label="Plots"):
                        result = resource[0]

                        if result.boxes is not None:
                            for box in result.boxes:
                                st.write(box.data)

                        elif hasattr(result, "obb") and result.obb is not None:
                            for obb in result.obb:
                                st.write(obb.data)

                        else:
                            st.warning("No detections found!")

                except Exception as exception:
                    st.error(body=f"No image is uploaded yet: {exception}")

    elif source_radio == SOURCE_VIDEO_FILE:
        source_video_file = st.sidebar.file_uploader(
            accept_multiple_files=False,
            disabled=False,
            help="Upload a Video File",
            key="source_video_file",
            label_visibility="visible",
            label="Upload a Video File",
            type=VIDEO_FILE_EXTENSIONS,
        )

        if source_video_file is not None:
            temporary_file = tempfile.NamedTemporaryFile(delete=False)
            temporary_file.write(source_video_file.read())
            video_capture = cv2.VideoCapture(temporary_file.name)

            column_1, column_2 = st.columns(
                border=False, spec=2, vertical_alignment="top"
            )

            with column_1:
                st.video(
                    autoplay=False,
                    data=source_video_file,
                    loop=False,
                    muted=False,
                )

            with column_2:
                try:
                    st_frame = st.empty()

                    while video_capture.isOpened():
                        success: bool
                        image: np.ndarray
                        success, image = video_capture.read()

                        if success:
                            display_plotted_frames(
                                streamlit_frame=st_frame,
                                source_frame=image,
                                width="stretch",
                            )

                        else:
                            video_capture.release()
                            break

                except Exception as exception:
                    st.error(body=f"Error loading video: {exception}")

                finally:
                    temporary_file.close()

    elif source_radio == SOURCE_WEBCAM_STREAM:
        source_webcam = int(
            st.sidebar.number_input(
                disabled=False,
                format="%d",
                help="Set Webcam Serial",
                key="webcam_serial",
                label_visibility="visible",
                label="Set Webcam Serial",
                min_value=0,
                placeholder="Set Webcam Serial",
                step=1,
                value=settings.DEFAULT_WEBCAM_NUMBER,
            )
        )

        source_width, source_height = select_webcam_stream_size()

        if st.sidebar.button(
            disabled=False,
            help="Run",
            key="run",
            label="Run",
        ):
            try:
                stop_button = st.sidebar.button(
                    disabled=False,
                    help="Stop",
                    key="stop",
                    label="Stop",
                )

                video_capture = cv2.VideoCapture(source_webcam)
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, source_width)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, source_height)

                st_frame = st.empty()

                while video_capture.isOpened():
                    success, image = video_capture.read()

                    if stop_button:
                        video_capture.release()
                        break

                    if success:
                        display_plotted_frames(
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

    elif source_radio == SOURCE_RTSP_STREAM:
        source_rtsp = st.sidebar.text_input(
            disabled=False,
            help="Write a RTSP Stream URL",
            key="rtsp_stream_url",
            label_visibility="visible",
            label="Set RTSP Stream URL",
            placeholder="Write a RTSP Stream URL",
        )

        if st.sidebar.button(
            disabled=False,
            help="Run",
            key="run",
            label="Run",
        ):
            try:
                stop_button = st.sidebar.button(
                    disabled=False,
                    help="Stop",
                    key="stop",
                    label="Stop",
                )

                video_capture = cv2.VideoCapture(source_rtsp)

                st_frame = st.empty()

                while video_capture.isOpened():
                    success, image = video_capture.read()

                    if stop_button:
                        video_capture.release()
                        break

                    if success:
                        display_plotted_frames(
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
