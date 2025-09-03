import joblib
import numpy as np

def main():
    model = joblib.load("model/housing_rf.pkl")
    example = np.array([[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]])
    pred = model.predict(example)
    print("Predicted house value:", pred[0])

if __name__ == "__main__":
    main()
