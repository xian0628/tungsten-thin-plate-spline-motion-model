import os
import sys

sys.path.insert(0, "stylegan-encoder")
import subprocess
import tempfile
import warnings
from typing import List

import dlib
import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from skimage import img_as_ubyte
from skimage.transform import resize
from tungstenkit import BaseIO, Field, Image, Option, Video, define_model

from demo import load_checkpoints, make_animation

warnings.filterwarnings("ignore")


class Input(BaseIO):
    source_image: Image = Field(description="Input source image.")
    motion_video: Video = Field(description="video including target motion.")
    dataset_name: str = Option(
        description="Choose training dataset.",
        choices=["vox", "taichi", "ted", "mgif"],
        default="vox",
    )


class Output(BaseIO):
    output_video: Video


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    gpu_mem_gb=15,
    python_version="3.8",
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ninja-build",
        "build-essential",
    ],
    python_packages=[
        "ipython==7.21.0",
        "torch==1.10.1",
        "torchvision==0.11.2",
        "cffi==1.14.6",
        "cycler==0.10.0",
        "decorator==5.1.0",
        "face-alignment==1.3.5",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.5",
        "kiwisolver==1.3.2",
        "matplotlib==3.4.3",
        "networkx==2.6.3",
        "numpy==1.20.3",
        "pandas==1.3.3",
        "Pillow==8.3.2",
        "pycparser==2.20",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.2",
        "pytz==2023.3.post1",
        "PyWavelets==1.1.1",
        "PyYAML==5.4.1",
        "scikit-image==0.18.3",
        "scikit-learn==1.0",
        "scipy==1.7.1",
        "six==1.16.0",
        "tqdm==4.62.3",
        "cmake==3.21.3",
    ],
    dockerfile_commands=["RUN pip install dlib"],
    batch_size=1,
)
class ThinPlateSplineMotionModel:
    def setup(self):
        self.landmarks_detector = LandmarksDetector(
            "shape_predictor_68_face_landmarks.dat"
        )
        self.device = torch.device("cuda:0")
        datasets = ["vox", "taichi", "ted", "mgif"]
        (
            self.inpainting,
            self.kp_detector,
            self.dense_motion_network,
            self.avd_network,
        ) = ({}, {}, {}, {})
        for d in datasets:
            (
                self.inpainting[d],
                self.kp_detector[d],
                self.dense_motion_network[d],
                self.avd_network[d],
            ) = load_checkpoints(
                config_path=f"config/{d}-384.yaml"
                if d == "ted"
                else f"config/{d}-256.yaml",
                checkpoint_path=f"checkpoints/{d}.pth.tar",
                device=self.device,
            )

    def predict(
        self,
        inputs: List[Input],
    ) -> List[Output]:
        source_image = inputs[0].source_image.path
        driving_video = inputs[0].motion_video.path
        dataset_name = inputs[0].dataset_name
        predict_mode = "relative"  # ['standard', 'relative', 'avd']
        # find_best_frame = False

        pixel = 384 if dataset_name == "ted" else 256

        if dataset_name == "vox":
            # first run face alignment
            self.align_image(str(source_image), "aligned.png")
            source_image = imageio.imread("aligned.png")
        else:
            source_image = imageio.imread(str(source_image))
        reader = imageio.get_reader(str(driving_video))
        fps = reader.get_meta_data()["fps"]
        source_image = resize(source_image, (pixel, pixel))[..., :3]

        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [
            resize(frame, (pixel, pixel))[..., :3] for frame in driving_video
        ]

        inpainting, kp_detector, dense_motion_network, avd_network = (
            self.inpainting[dataset_name],
            self.kp_detector[dataset_name],
            self.dense_motion_network[dataset_name],
            self.avd_network[dataset_name],
        )

        predictions = make_animation(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device="cuda:0",
            mode=predict_mode,
        )

        # save resulting video
        out_path = "output.mp4"
        if os.path.exists(out_path):
            os.remove(out_path)
        imageio.mimsave(
            out_path, [img_as_ubyte(frame) for frame in predictions], fps=fps
        )
        return [Output(output_video=Video.from_path(out_path))]

    def align_image(self, raw_img_path, aligned_face_path):
        for i, face_landmarks in enumerate(
            self.landmarks_detector.get_landmarks(raw_img_path), start=1
        ):
            image_align(raw_img_path, aligned_face_path, face_landmarks)
