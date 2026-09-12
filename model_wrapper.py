import numpy as np
import pandas as pd

from helper_functions import *
from sklearn.model_selection import train_test_split
from typing import List, Dict
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

BATCH_SIZE = 5
AUC_THRESHOLD = 0.45

planes = ["Sagittal", "Axial", "Coronal"]

study_ids = os.listdir("train_series")

train_df = pd.read_csv("train.csv").drop(columns=["Report"]).dropna()
train_series_df = pd.read_csv("train_series.csv")

train_df = train_df[train_df["StudyInstanceUID"].isin(study_ids)]
train_series_df = train_series_df[train_series_df["StudyInstanceUID"].isin(study_ids)]
test_series_df = pd.read_csv("test_series.csv")
full_df = pd.merge(train_df, train_series_df, on='StudyInstanceUID', how='inner')

# 1. Initialize the multi-label stratifier
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# 2. Extract X (IDs) and Y (the 12 target columns)
X = train_df[["StudyInstanceUID"]].values
Y = train_df[target_columns].values

# 3. Generate the stratified splits
train_ids = []
test_ids = []
for train_index, test_index in msss.split(X, Y):
    train_ids = X[train_index].flatten()
    test_ids = X[test_index].flatten()

print(f"# of Training Instances: {len(train_ids)}")
print(f"# of Testing Instances: {len(test_ids)}")
# 4. Filter your full dataframe using the newly stratified IDs
train_full = full_df[full_df["StudyInstanceUID"].isin(train_ids)]
test_full = full_df[full_df["StudyInstanceUID"].isin(test_ids)]

print_memory_usage()

class Model:

    def __init__(self, anatomical_plane, fluid_sensitive = None, fat_suppression = None, full_train=True):
        self.auc_scores = np.zeros(len(target_columns))
        self.fluid_sensitive = fluid_sensitive
        self.fat_suppression = fat_suppression
        self.anatomical_plane = anatomical_plane

        print(f"""Training Model: 
\tPlane: {self.anatomical_plane}
\tFluid Sensitive: {self.fluid_sensitive}
\tFat Suppression: {self.fat_suppression}""")
        if full_train:
            X, y = get_data(train_full, "train_series", anatomical_plane, fluid_sensitive, fat_suppression)

            msss_ = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
            train_index, validation_index = next(msss_.split(X=X, y=y))

            X_train = X[train_index]
            X_validation = X[validation_index]

            y_train = y[train_index]
            y_validation = y[validation_index]

            print(f"\tTraining Shape: {X_train.shape}")
            print(f"\tValidation Shape: {X_validation.shape}")
            self.full_fit(X_train, y_train)
            y_pred = self.predict_batch(X_validation)
            self.auc_scores = roc_auc_score(y_validation, y_pred, average=None)

            print(f"\tAUC Score: {np.mean(self.auc_scores)}")
            for i in range(len(target_columns)):
                print(f"\t\t{target_columns[i]}: {self.auc_scores[i]}")
            print()
        else:
            folders, y = get_data(train_full, "train_series", anatomical_plane, fluid_sensitive, fat_suppression,
                                  get_folder_paths=True)

            msss_ = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
            train_index, validation_index = next(msss_.split(X=folders, y=y))

            X_train = folders.iloc[train_index]
            X_validation = folders.iloc[validation_index]

            y_train = y[train_index]
            y_validation = y[validation_index]

            print(f"\tTraining Size: {X_train.shape}")
            print(f"\tValidation Size: {X_validation.shape}")
            self.batch_fit(X_train, y_train, X_validation, y_validation)

            indices = np.arange(len(X_validation))
            all_batch_predictions = []
            for batch_idx in np.array_split(indices, 5):
                folders_batch = X_validation.iloc[batch_idx]

                X_batch = np.array([get_training_instance(f) for f in folders_batch])
                all_batch_predictions.append(self.predict_batch(X_batch))

            y_pred = np.vstack(all_batch_predictions)
            self.auc_scores = roc_auc_score(y_validation, y_pred, average=None)

            print(f"\tAUC Score: {np.mean(self.auc_scores)}")
            for i in range(len(target_columns)):
                print(f"\t\t{target_columns[i]}: {self.auc_scores[i]}")
            print()

    def full_fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def batch_fit(self, training_folders: pd.Series, y_train: np.ndarray, validation_folders: pd.Series, y_validation: np.ndarray):
        pass
    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        pass

    def predict_instance(self, x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def make_prediction(instance_ids: pd.Series, models: List['Model'], depth: int, series: str = "train_series") -> pd.DataFrame:
        """

        :param instance_ids: id for each instance being predicted
        :param models: ensemble of models
        :param depth: 1 -> models only care about anatomical plane of images, 2 -> fat, fluid, and anatomical planes now considered
        :param series: which folder instances from, train or test
        :return: A dictionary matching instance id to the predicted value
        """

        final_predictions = {}

        for id_ in instance_ids:
            auc_scores = []
            predictions = []

            if series == "train_series":
                studies = train_series_df[train_series_df["StudyInstanceUID"] == id_]
            else:
                studies = test_series_df[test_series_df["StudyInstanceUID"] == id_]

            for _, study in studies.iterrows():
                for m in models:
                    folder_path = (series + "/" +
                                   study["StudyInstanceUID"] + "/" +
                                   study["SeriesInstanceUID"])

                    img_count = len(os.listdir(folder_path))
                    if img_count < MIN_IMG_COUNT:
                        continue

                    if depth == 1:
                        if study["Anatomical_Plane"] == m.anatomical_plane:
                            auc_scores.append(np.copy(m.auc_scores))
                            predictions.append(m.predict_instance(get_training_instance(folder_path)))
                    else:
                        if (study["Anatomical_Plane"] == m.anatomical_plane and
                                study["Fluid_Sensitive"] == m.fluid_sensitive and
                                study["Fat_Suppression"] == m.fat_suppression):

                            auc_scores.append(np.copy(m.auc_scores))
                            predictions.append(m.predict_instance(get_training_instance(folder_path)))

            weights = Model.calculate_weights(auc_scores)
            final_predictions[id_] = Model.apply_weights(weights, predictions)

        return pd.DataFrame.from_dict(final_predictions, orient='index').reset_index()

    @staticmethod
    def get_ensemble_auc_score(models: List['Model'], depth: int):
        pred_ = Model.make_prediction(test_ids, models, depth).sort_values(by='index')
        true_ = train_df[train_df["StudyInstanceUID"].isin(test_ids)].sort_values(by='StudyInstanceUID')

        pred_ = pred_.drop(columns=["index"]).to_numpy()
        true_ = true_[target_columns].to_numpy()

        return roc_auc_score(true_, pred_, average=None)


    @staticmethod
    def calculate_weights(auc_scores: list):

        weights = [[] for _ in range(len(auc_scores))]

        for i in range(len(target_columns)):
            highest_auc = 0.0
            highest_pos = -1

            for j in range(len(auc_scores)):
                if auc_scores[j][i] > highest_auc:
                    highest_auc = auc_scores[j][i]
                    highest_pos = j

                if auc_scores[j][i] < AUC_THRESHOLD:
                    auc_scores[j][i] = 0.0

            if highest_auc < AUC_THRESHOLD:
                auc_scores[highest_pos][i] = AUC_THRESHOLD

        for i in range(len(target_columns)):
            auc_sum = 0

            for cur_scores in auc_scores:
                auc_sum += cur_scores[i]

            for j in range(len(auc_scores)):
                weights[j].append(auc_scores[j][i] / auc_sum)
        
        return weights

    @staticmethod
    def apply_weights(weights, predictions):
        final_predictions = []

        for i in range(len(target_columns)):
            target_prediction = 0

            for j in range(len(weights)):
                target_prediction += weights[j][i] * predictions[j][i]
            final_predictions.append(target_prediction)

        return final_predictions