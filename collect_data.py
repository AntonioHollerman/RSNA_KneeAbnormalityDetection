import kagglehub
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
kagglehub.login()

# Download latest version
path = kagglehub.competition_download('rsna-knee-abnormality-detection')

print("Path to competition files:", path)