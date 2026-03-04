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

    def select_webcam_stream_size(width=None, height=None):
        combined_widths = []

        for aspect_ratio in ASPECT_RATIOS:
            if aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9

            elif aspect_ratio == ASPECT_RATIO_4_3:
                aspect_ratio = 4 / 3

            elif aspect_ratio == CUSTOM_ASPECT_RATIO:
                continue

            else:
                st.error(body="Failed to determine aspect ratio!")

            for height in WEBCAM_STREAM_HEIGHTS:
                width = int(height * aspect_ratio)
                combined_widths.append(width)

        minimum_height = min(WEBCAM_STREAM_HEIGHTS)
        maximum_height = max(WEBCAM_STREAM_HEIGHTS)
        minimum_width = min(combined_widths)
        maximum_width = max(combined_widths)

        aspect_ratio = st.sidebar.radio(
            disabled=False,
            help="Select Webcam Stream Aspect Ratio",
            horizontal=True,
            key="webcam_stream_aspect_ratio",
            label_visibility="visible",
            label="Select Webcam Stream Aspect Ratio",
            options=ASPECT_RATIOS,
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

            for height in WEBCAM_STREAM_HEIGHTS:
                width = int(height * aspect_ratio)
                widths.append(width)
                resolutions.append(f"{height}p: {width} x {height}")

            resolution = st.sidebar.selectbox(
                accept_new_options=False,
                disabled=False,
                help="Select Webcam Stream Resolution",
                key="webcam_stream_resolution",
                label_visibility="visible",
                label="Select Webcam Stream Resolution",
                options=resolutions,
                placeholder="Select Webcam Stream Resolution",
            )

            width, height = map(int, resolution.split(":")[1].strip().split("x"))

        elif aspect_ratio == CUSTOM_ASPECT_RATIO:
            width = int(
                st.sidebar.number_input(
                    disabled=False,
                    format="%d",
                    help="Set Webcam Stream Width",
                    key="webcam_stream_width",
                    label_visibility="visible",
                    label="Set Webcam Stream Width",
                    max_value=maximum_width,
                    min_value=minimum_width,
                    placeholder="Set Webcam Stream Width",
                    step=1,
                    value=settings.DEFAULT_VIDEO_WIDTH,
                )
            )

            height = int(
                st.sidebar.number_input(
                    disabled=False,
                    format="%d",
                    help="Set Webcam Stream Height",
                    key="webcam_stream_height",
                    label_visibility="visible",
                    label="Set Webcam Stream Height",
                    max_value=maximum_height,
                    min_value=minimum_height,
                    placeholder="Set Webcam Stream Height",
                    step=1,
                    value=settings.DEFAULT_VIDEO_HEIGHT,
                )
            )

        else:
            st.error(body="Failed to select aspect ratio!")

        return width, height

    def display_plotted_frames(streamlit_frame, source_frame, width):
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
        options=["bytetrack.yaml", "botsort.yaml", "No"],
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

    MODEL_SUFFIXES = {
        OBJECT_DETECTION_MODEL: "",
        OBB_OBJECT_DETECTION_MODEL: "-obb",
        OBJECT_SEGMENTATION_MODEL: "-segment",
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

    model_path = (
        str(settings.MODEL_DIRECTORY)
        + "/yolo26"
        + MODEL_WEIGHT_SUFFIXES.get(model_weight)
        + MODEL_SUFFIXES.get(model_type)
        + ".pt"
    )

    try:
        model = YOLO(model=model_path, task=MODEL_TASKS.get(model_type), verbose=False)

    except Exception as exception:
        st.error(
            body=f"Failed to load model!\nCheck the path: {model_path}: {exception}"
        )

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
            border=False, spec=2, vertical_alignment="center"
        )

        with column_1:
            try:
                if source_image_file is not None:
                    uploaded_image = PIL.Image.open(fp=source_image_file)

                    st.image(
                        caption="Source Image",
                        image=source_image_file,
                        output_format="auto",
                        width="stretch",
                    )

            except Exception as exception:
                st.error(body=f"Error occurred while opening the image: {exception}")

        with column_2:
            if source_image_file is not None:
                if st.sidebar.button(
                    disabled=False,
                    help="Run",
                    key="run",
                    label="Run",
                ):
                    if tracker == "No":
                        resource = model(
                            source=uploaded_image, conf=confidence, verbose=False
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

                    st.snow()

                    try:
                        with st.expander(label="Plots"):
                            for box in boxes:
                                st.write(box.data)

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
                border=False, spec=2, vertical_alignment="center"
            )

            with column_1:
                st.video(
                    autoplay=False,
                    data=source_video_file,
                    loop=False,
                    muted=False,
                )

            with column_2:
                if st.sidebar.button(
                    disabled=False,
                    help="Run",
                    key="run",
                    label="Run",
                ):
                    try:
                        st_frame = st.empty()

                        while video_capture.isOpened():
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
                        st.error(label=f"Error loading video: {exception}")

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
                video_capture.set(propId=3, value=source_width)
                video_capture.set(propId=4, value=source_height)

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
