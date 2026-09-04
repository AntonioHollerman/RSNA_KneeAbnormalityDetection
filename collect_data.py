import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_11d286a48ad18aa0b42878afc9c3620d'
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

target_dir = os.path.expanduser("~/.cache/kagglehub/competitions/rsna-knee-abnormality-detection")
os.chdir(target_dir)
# 1. Data loading and identifying missing IDs
study_ids = os.listdir("train_series")
train_df = pd.read_csv("train.csv")
labeled_data = train_df.dropna()

ids_for_studies_not_downloaded = labeled_data[~labeled_data["StudyInstanceUID"].isin(study_ids)]["StudyInstanceUID"]
print(f"Missing folders: {len(ids_for_studies_not_downloaded)}")

# 2. Authenticate
api = KaggleApi()
api.authenticate()
competition_name = 'rsna-knee-abnormality-detection'

# 3. Retrieve ALL files using Pagination
print("Fetching list of all competition files... (This will take a few minutes due to pagination limits)")
file_paths = []
page_token = None

while True:
    # Request the current page of files
    response = api.competition_list_files(competition_name, page_token=page_token)

    # Extract the file objects from the response wrapper
    file_objects = getattr(response, 'files', getattr(response, 'data_files', []))

    # Save the file paths
    for f in file_objects:
        if hasattr(f, 'name'):
            file_paths.append(str(f.name))
        elif hasattr(f, 'ref'):
            file_paths.append(str(f.ref))
        else:
            file_paths.append(str(f))

    # Check for a next page token. If None, we've reached the end.
    page_token = getattr(response, 'next_page_token', None)

    if not page_token:
        break

print(f"Total files found on Kaggle: {len(file_paths)}")

# 4. Filter and download missing files
for study_id in ids_for_studies_not_downloaded:
    print(f"\nProcessing missing study: {study_id}")

    # Find all exact files (.dcm) nested under this study ID
    prefix = f"train_images/{study_id}/"
    files_to_download = [f for f in file_paths if f.startswith(prefix)]

    if not files_to_download:
        print(f"  No files found on Kaggle under {prefix}")
        continue

    for exact_file in files_to_download:
        api.competition_download_file(
            competition_name,
            exact_file,
            path='train_series'
        )