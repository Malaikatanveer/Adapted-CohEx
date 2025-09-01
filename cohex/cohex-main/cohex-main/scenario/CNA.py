import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import random

class GeneScenario:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # load dataset
        csv_path = r"scenario/Multivariate_CNA_filtered_by_hallmark.csv"
        self.data = pd.read_csv(csv_path)
        print("Shape is:", self.data.shape)

        # save gene identifiers separately
        self.gene_ids = self.data['Gene']

        # features & target
        self.X = self.data.drop(columns=['Gene', "Stouffer's Z"]).apply(pd.to_numeric, errors='coerce').fillna(0).values
        print("X shape: ", self.X.shape)
        self.y = self.data["Stouffer's Z"].values

        # model path
        self.model_path = './scenario/cohex_gene_model.json'

        if os.path.exists(self.model_path):
            print("Using existing model")
            self.model = xgb.XGBRegressor(tree_method='hist', random_state=self.random_seed)
            self.model.load_model(self.model_path)
        else:
            print("Creating and training model")
            self.train()
            model_save_path = os.path.abspath(self.model_path)
            self.model.save_model(model_save_path)
    

    @staticmethod
    def adjusted_r2(r2, n, p):
        """Calculate adjusted R² score."""
        if n - p - 1 == 0:
            return r2  # avoid division by zero
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def train(self):
        # 1. train-test split (85-15)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.15, random_state=self.random_seed)
        
        gene_ids_train, gene_ids_test = train_test_split(
            self.gene_ids, test_size=0.15, random_state=self.random_seed)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 2. define parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.1, 0.01, 0.001],
            'reg_alpha': [0, 0.1, 1],    # L1 regularisation
            'reg_lambda': [1, 5, 10],    # L2 regularizstion
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        # 3. randomised search with 3-fold CV
        base_model = xgb.XGBRegressor(tree_method='hist', random_state=self.random_seed, verbosity=0)
        randomized_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=30,
            scoring='r2',
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=self.random_seed
        )
        randomized_search.fit(X_train, y_train)

        best_params = randomized_search.best_params_
        print("Best Hyperparameters:", best_params)

        # 4. retrain best model on full training set using early stopping
        self.model = xgb.XGBRegressor(
            **best_params,
            tree_method='hist',
            random_state=self.random_seed,
            verbosity=1
        )

        # use small validation split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=self.random_seed)

        self.model = xgb.XGBRegressor(
            **best_params,
            tree_method='hist',
            random_state=self.random_seed,
            verbosity=1,
            eval_metric='rmse',
            early_stopping_rounds=10
        )

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        # 5. evaluate on test set
        y_test_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        
        # calculate adjusted R²
        n_samples = len(y_test)
        n_features = X_test.shape[1]
        adj_r2 = self.adjusted_r2(r2, n_samples, n_features)

        results_str = (
            f"Test R2: {r2:.4f}\n"
            f"Test Adjusted R2: {adj_r2:.4f}\n"
            f"Test RMSE: {rmse:.4f}\n"
            f"Test MAE: {mae:.4f}\n"
        )

        print(results_str)
        with open("results.txt", "a") as f:
            f.write(results_str + "\n")


        # 6. save predictions
        results_df = pd.DataFrame({
            'Gene': gene_ids_test.values,
            'Actual Z': y_test,
            'Predicted Z': y_test_pred
        })
        results_df.to_csv("gene_predictions.csv", index=False)
        print("Saved predictions to gene_predictions.csv")

if __name__ == '__main__':
    model = GeneScenario()
