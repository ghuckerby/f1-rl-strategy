import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import joblib
from scripts.fastf1_data_extraction import FastF1DataExtractor, COMPOUND_MAP

class LapPredictor:
    MODEL_DIR = "f1_gym/models/lap_predictors"

    def __init__(self):
        self.extractor = FastF1DataExtractor()
        self.models: dict[str, RandomForestRegressor] = {}

    @staticmethod
    def race_to_filename(race_name: str) -> str:
        return race_name.strip().replace(" ", "_").lower()

    def _model_path(self, race_name: str) -> str:
        return os.path.join(self.MODEL_DIR, f"{self.race_to_filename(race_name)}.pkl")

    def create_dataset(self, year: int, race_name: str) -> pd.DataFrame:
        session = self.extractor.load_session(year, race_name)
        laps = session.laps.copy()

        valid_laps = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna())
        ].copy()

        valid_laps['LapTimeSec'] = valid_laps['LapTime'].dt.total_seconds()
        valid_laps['CompoundID'] = valid_laps['Compound'].map(COMPOUND_MAP)
        valid_laps = valid_laps.sort_values(['Driver', 'LapNumber'])
        valid_laps['TyreAge'] = valid_laps.groupby(['Driver', 'Stint']).cumcount() + 1

        return valid_laps[['LapNumber', 'TyreAge', 'CompoundID', 'LapTimeSec']].dropna()

    def train(self, race_name: str, data: pd.DataFrame):
        model = RandomForestRegressor(n_estimators=100, random_state=6)
        X = data[['LapNumber', 'TyreAge', 'CompoundID']]
        y = data['LapTimeSec']
        model.fit(X, y)

        self.models[race_name] = model
        print(f"{race_name} Model trained")

    def train_races(self, year: int, race_list: list[str]):
        for race_name in race_list:
            data = self.create_dataset(year, race_name)

            print(f"\nTraining model for: {race_name} with {len(data)} samples\n")
            self.train(race_name, data)

    def save_models(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        for race_name, model in self.models.items():
            path = self._model_path(race_name)
            joblib.dump(model, path)
            print(f"Saved: {path}")

    @classmethod
    def load_model(cls, race_name: str) -> RandomForestRegressor:
        path = os.path.join(cls.MODEL_DIR, f"{cls.race_to_filename(race_name)}.pkl")
        return joblib.load(path)

    @staticmethod
    def test_model(race_name: str):
        model = LapPredictor.load_model(race_name)

        test_scenarios = [
            {'LapNumber': 5,  'TyreAge': 3,  'CompoundID': 1, 'Desc': 'Fresh Softs (Early Race)'},
            {'LapNumber': 30, 'TyreAge': 3,  'CompoundID': 1, 'Desc': 'Fresh Softs (Mid Race)'},
            {'LapNumber': 50, 'TyreAge': 3,  'CompoundID': 1, 'Desc': 'Fresh Softs (Late Race)'},
            {'LapNumber': 30, 'TyreAge': 20, 'CompoundID': 1, 'Desc': 'Old Softs (Mid Race)'},

            {'LapNumber': 5,  'TyreAge': 3,  'CompoundID': 2, 'Desc': 'Fresh Mediums (Early Race)'},
            {'LapNumber': 30, 'TyreAge': 3,  'CompoundID': 2, 'Desc': 'Fresh Mediums (Mid Race)'},
            {'LapNumber': 50, 'TyreAge': 3,  'CompoundID': 2, 'Desc': 'Fresh Mediums (Late Race)'},
            {'LapNumber': 30, 'TyreAge': 25, 'CompoundID': 2, 'Desc': 'Old Mediums (Mid Race)'},

            {'LapNumber': 5,  'TyreAge': 3,  'CompoundID': 3, 'Desc': 'Fresh Hards (Early Race)'},
            {'LapNumber': 30, 'TyreAge': 3,  'CompoundID': 3, 'Desc': 'Fresh Hards (Mid Race)'},
            {'LapNumber': 50, 'TyreAge': 3,  'CompoundID': 3, 'Desc': 'Fresh Hards (Late Race)'},
            {'LapNumber': 50, 'TyreAge': 35, 'CompoundID': 3, 'Desc': 'Old Hards (End of Race)'}
        ]

        df_test = pd.DataFrame(test_scenarios)
        predictions = model.predict(df_test[['LapNumber', 'TyreAge', 'CompoundID']])

        print(f"\n--- Predictions for: {race_name} ---")
        for i, pred in enumerate(predictions):
            print(f"  {test_scenarios[i]['Desc']}: {pred:.3f}s")


if __name__ == "__main__":
    predictor = LapPredictor()

    training_races = [
        'Bahrain Grand Prix',
        'Saudi Arabian Grand Prix',
        'Australian Grand Prix',
        'Japanese Grand Prix',
        'Chinese Grand Prix',
        'Miami Grand Prix',
        'Emilia Romagna Grand Prix',
        'Monaco Grand Prix',
        'Canadian Grand Prix',
        'Spanish Grand Prix',
        'Austrian Grand Prix',
        'British Grand Prix',
        'Hungarian Grand Prix',
        'Belgian Grand Prix',
        'Dutch Grand Prix',
        'Italian Grand Prix',
        'Azerbaijan Grand Prix',
        'Singapore Grand Prix',
    ]

    predictor.train_races(2024, training_races)
    predictor.save_models()
    for race in training_races:
        LapPredictor.test_model(race)