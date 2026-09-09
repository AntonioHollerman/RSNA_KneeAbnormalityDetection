import pydicom
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
import cv2
import psutil

target_dir = os.path.expanduser("~/.cache/kagglehub/competitions/rsna-knee-abnormality-detection")
os.chdir(target_dir)
target_columns = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
       'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion',
       'Fracture']
MIN_IMG_COUNT = 12
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / (1024 ** 2)

    print(f"Current RAM usage: {mem_mb:.2f} MB")

def get_img_count(folder_path):
    return len([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')])

def get_instance_number(filepath):
    try:
        return pydicom.dcmread(filepath, stop_before_pixels=True).InstanceNumber
    except AttributeError:
        return filepath

def display_dcm(dcm_file_path):
    dicom_data = pydicom.dcmread(dcm_file_path)
    image_array = dicom_data.pixel_array

    plt.imshow(image_array, cmap='gray')
    plt.title("Knee MRI Slice")
    plt.axis('off')
    plt.show()


def display_dcm_series_matrix(folder_path):
    dcm_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]

    if not dcm_files:
        print("No DICOM files found in the directory.")
        return

    dcm_files.sort(key=get_instance_number)
    num_images = len(dcm_files)
    grid_size = math.ceil(math.sqrt(num_images))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    axes = axes.flatten()
    for i, file_path in enumerate(dcm_files):
        dicom_data = pydicom.dcmread(file_path)
        image_array = dicom_data.pixel_array

        height, width = image_array.shape
        axes[i].imshow(image_array, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Slice {i + 1}\n{width}x{height} px", fontsize=8)

    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

    return pydicom.dcmread(dcm_files[int(len(dcm_files) / 2)])

def scale_img_array(image_array: np.ndarray):
    min_val = np.min(image_array)
    max_val = np.max(image_array)

    return (image_array - min_val) / (max_val - min_val + 1e-8)


def get_training_instance(folder_path):
    dcm_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]

    if not dcm_files:
        print("No DICOM files found in the directory.")
        return

    dcm_files.sort(key=get_instance_number)
    total_files = len(dcm_files)

    if total_files > MIN_IMG_COUNT:
        # Generate 18 evenly spaced indices
        # (target_count - 1) ensures we strictly hit the last index
        indices = [round(i * (total_files - 1) / (MIN_IMG_COUNT - 1)) for i in range(MIN_IMG_COUNT)]

        # Overwrite dcm_files with only the selected slices
        dcm_files = [dcm_files[idx] for idx in indices]

    processed_slices = []
    for file in dcm_files:
        # Read the raw pixel array
        img = pydicom.dcmread(file).pixel_array

        # Enforce 512x512 dimension (cv2.resize expects width x height)
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Apply your scaling step and append
        processed_slices.append(scale_img_array(img_resized))

    return np.array(processed_slices)


def get_data(df: pd.DataFrame, series, anatomical_plane, fluid_sensitive = None, fat_suppression = None,
             filter_len=True, get_target_columns=True):

    def check_len(path):
        return len(os.listdir(path)) >= MIN_IMG_COUNT

    filtered_df = df.copy()
    if filter_len:
        folder_paths = (series + "/" +
                        df["StudyInstanceUID"] + "/" +
                        df["SeriesInstanceUID"])
        correct_len = folder_paths.apply(check_len)
        filtered_df = df[correct_len]

    if fluid_sensitive is None and fat_suppression is None and anatomical_plane is None:
        filtered_df = filtered_df.copy()
    elif anatomical_plane is not None and fluid_sensitive is None and fat_suppression is None:
        filtered_df = filtered_df[filtered_df["Anatomical_Plane"] == anatomical_plane]
    else:
        filtered_df = filtered_df[(filtered_df["Fluid_Sensitive"] == fluid_sensitive) &
                         (filtered_df["Fat_Suppression"] == fat_suppression) &
                         (filtered_df["Anatomical_Plane"] == anatomical_plane)]


    folder_paths = (series + "/" +
                    filtered_df["StudyInstanceUID"] + "/" +
                    filtered_df["SeriesInstanceUID"])
    return (np.array([get_training_instance(path) for path in folder_paths]),
            np.array(filtered_df[target_columns]) if get_target_columns else None)