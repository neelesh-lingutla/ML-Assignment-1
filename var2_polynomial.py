import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

train_data = pd.read_csv("BT2024152_train_var2.csv")

X = train_data[["x1", "x2", "x3"]]
y = train_data["y"]


# ============================================================
# 2. CROSS-VALIDATION SETUP
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. ALPHA VALUES
# ============================================================

alphas = [
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
    10
]


# ============================================================
# 4. TRY EACH DEGREE
# ============================================================

results = []

for degree in range(1, 21):

    alpha_results = []

    for alpha in alphas:

        model = Pipeline([
            (
                "poly",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                )
            ),
            (
                "ridge",
                Ridge(alpha=alpha)
            )
        ])

        # CV MSE
        mse_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error"
        )

        cv_mse = -mse_scores.mean()

        # CV R2
        r2_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2"
        )

        cv_r2 = r2_scores.mean()

        # Store internally
        alpha_results.append({
            "alpha": alpha,
            "mse": cv_mse,
            "r2": cv_r2
        })


    # ========================================================
    # 5. FIND BEST ALPHA FOR THIS DEGREE
    # ========================================================

    best_alpha_result = min(
        alpha_results,
        key=lambda x: x["mse"]
    )

    best_alpha = best_alpha_result["alpha"]
    best_mse = best_alpha_result["mse"]
    best_r2 = best_alpha_result["r2"]


    results.append({
        "Degree": degree,
        "Best_Alpha": best_alpha,
        "CV_MSE": best_mse,
        "CV_R2": best_r2
    })


    # ========================================================
    # 6. IMMEDIATELY SHOW COMPLETED DEGREE
    # ========================================================

    print(
        f"Degree {degree} completed -> "
        f"Best Alpha: {best_alpha}, "
        f"CV MSE: {best_mse:.6f}, "
        f"CV R2: {best_r2:.6f}",
        flush=True
    )


# ============================================================
# 7. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n==================================================")
print("RESULTS FOR ALL DEGREES")
print("==================================================")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 8. SELECT BEST DEGREE OVERALL
# ============================================================

best_row = results_df.loc[
    results_df["CV_MSE"].idxmin()
]

best_degree = int(best_row["Degree"])
best_alpha = float(best_row["Best_Alpha"])
best_cv_mse = float(best_row["CV_MSE"])
best_cv_r2 = float(best_row["CV_R2"])


print("\n==================================================")
print("FINAL RESULT")
print("==================================================")

print(f"Best Polynomial Degree: {best_degree}")
print(f"Best Alpha: {best_alpha}")
print(f"Best CV MSE: {best_cv_mse:.6f}")
print(f"Best CV R2: {best_cv_r2:.6f}")


# ============================================================
# 9. TRAIN FINAL RIDGE MODEL
# ============================================================

final_poly = PolynomialFeatures(
    degree=best_degree,
    include_bias=False
)

X_poly = final_poly.fit_transform(X)

final_model = Ridge(
    alpha=best_alpha
)

final_model.fit(X_poly, y)


# ============================================================
# 10. GET WEIGHTS AND INTERCEPT INTERNALLY
# ============================================================

weights = final_model.coef_
intercept = final_model.intercept_


# ============================================================
# 11. LOAD TEST DATA
# ============================================================

test_data = pd.read_csv(
    "BT2024152_test_var2.csv"
)

X_test = test_data[["x1", "x2", "x3"]]


# ============================================================
# 12. POLYNOMIAL TRANSFORMATION
# ============================================================

X_test_poly = final_poly.transform(X_test)


# ============================================================
# 13. MANUAL PREDICTION
# ============================================================

predictions = (
    X_test_poly @ weights
) + intercept


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "y": predictions
})

prediction_df.to_csv(
    "BT2024152_pred_var2.csv",
    index=False
)


print("\n==================================================")
print("PREDICTION COMPLETE")
print("==================================================")

print(
    "Predictions saved to: BT2024152_pred_var2.csv"
)
