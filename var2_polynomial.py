import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

train = pd.read_csv("BT2024152_train_var2.csv")

print("Training data shape:", train.shape)
print("\nTraining columns:")
print(train.columns.tolist())


# ============================================================
# 2. SEPARATE INPUTS AND TARGET
# ============================================================

X = train.drop(columns=["y"])
y = train["y"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: y")


# ============================================================
# 3. CROSS-VALIDATION SETUP
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. TEST POLYNOMIAL DEGREES 1 TO 20
# ============================================================

results = []

for degree in range(1, 21):

    model = Pipeline([
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),
        (
            "linear_regression",
            LinearRegression()
        )
    ])

    # Cross-validation MSE
    mse_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    cv_mse = -mse_scores.mean()

    # Cross-validation R2
    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    cv_r2 = r2_scores.mean()

    results.append({
        "Degree": degree,
        "CV_MSE": cv_mse,
        "CV_R2": cv_r2
    })


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n==============================================")
print("POLYNOMIAL DEGREE COMPARISON")
print("==============================================")

print(results_df.to_string(index=False))


# ============================================================
# 6. SELECT BEST DEGREE
# ============================================================

# Best degree = lowest CV MSE
best_degree = results_df.loc[
    results_df["CV_MSE"].idxmin(),
    "Degree"
]

best_degree = int(best_degree)

best_cv_mse = results_df.loc[
    results_df["Degree"] == best_degree,
    "CV_MSE"
].iloc[0]

best_cv_r2 = results_df.loc[
    results_df["Degree"] == best_degree,
    "CV_R2"
].iloc[0]


print("\n==============================================")
print("BEST MODEL")
print("==============================================")

print("Best Polynomial Degree:", best_degree)
print("Best CV MSE:", best_cv_mse)
print("Best CV R2 :", best_cv_r2)


# ============================================================
# 7. TRAIN FINAL MODEL USING BEST DEGREE
# ============================================================

final_model = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(
            degree=best_degree,
            include_bias=False
        )
    ),
    (
        "linear_regression",
        LinearRegression()
    )
])

final_model.fit(X, y)


# ============================================================
# 8. TRAINING PERFORMANCE
# ============================================================

train_predictions = final_model.predict(X)

train_mse = mean_squared_error(
    y,
    train_predictions
)

train_r2 = r2_score(
    y,
    train_predictions
)

print("\n==============================================")
print("TRAINING PERFORMANCE")
print("==============================================")

print("Training MSE:", train_mse)
print("Training R2 :", train_r2)


# ============================================================
# 9. LOAD TEST DATA
# ============================================================

test = pd.read_csv("BT2024152_test_var2.csv")

X_test = test


# ============================================================
# 10. PREDICT TEST DATA
# ============================================================

test_predictions = final_model.predict(X_test)


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

prediction_file = "BT2024152_pred_var2.csv"

prediction_df = pd.DataFrame({
    "y": test_predictions
})

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\n==============================================")
print("PREDICTION")
print("==============================================")

print("Prediction file saved as:")
print(prediction_file)

print("\nFirst 10 predictions:")
print(prediction_df.head(10))