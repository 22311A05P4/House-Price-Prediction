import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def main():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), X.columns.tolist())]
    )

    model = Pipeline(steps=[
        ('prep', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R²:", r2_score(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/housing_rf.pkl")
    print("Model saved to model/housing_rf.pkl")

if __name__ == "__main__":
    main()
