import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Car Price and Car Make Prediction""")
    return


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import (
        StandardScaler,
        MultiLabelBinarizer,
        LabelEncoder,
    )
    from sklearn.metrics import classification_report
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        median_absolute_error,
        mean_absolute_percentage_error,
    )

    import xgboost
    import joblib
    return (
        LabelEncoder,
        LinearRegression,
        StandardScaler,
        classification_report,
        joblib,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
        xgboost,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Data Understanding""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("sports-car-price.csv")

    columns = df.columns

    df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

    df["price_(in_usd)"] = df["price_(in_usd)"].str.replace(",", "").astype(float)

    cols_to_convert = [
        "engine_size_(l)",
        "horsepower",
        "torque_(lb-ft)",
        "0-60_mph_time_(seconds)",
    ]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

    df = df.drop("car_model", axis=1)
    df = df.dropna()
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df["car_make"].value_counts()
    return


@app.cell
def _(df):
    df.describe().T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Exploratory Data Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.1 Car Make Distribution""")
    return


@app.cell
def _(df, plt, sns):
    sns.barplot(data=df["year"].value_counts().reset_index(), x="year", y="count")
    plt.title("Number of Cars per year")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.1 Bar Plot by Car Make""")
    return


@app.cell
def _(df, plt, sns):
    car_make_counts = df["car_make"].value_counts().reset_index()
    car_make_counts.columns = ["car_make", "count"]  # Rename kolom

    # Plot horizontal barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=car_make_counts, x="count", y="car_make", orient="h")

    plt.xlabel("Count")
    plt.ylabel("Car Make")
    plt.title("Jumlah Mobil per Car Make")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(df, plt, sns):
    df_by_engine_size = (
        df[
            [
                "car_make",
                "engine_size_(l)",
                "horsepower",
                "torque_(lb-ft)",
                "0-60_mph_time_(seconds)",
                "price_(in_usd)",
            ]
        ]
        .groupby("car_make")
        .mean()
        .round(2)
        .sort_values(by="engine_size_(l)", ascending=False)
        .reset_index()
    )

    _numeric_columns = [
        "engine_size_(l)",
        "horsepower",
        "torque_(lb-ft)",
        "0-60_mph_time_(seconds)",
        "price_(in_usd)",
    ]

    # Loop untuk plotting tiap metrik
    for col in _numeric_columns:
        # Urutkan berdasarkan kolom saat ini
        df_sorted = df_by_engine_size.sort_values(by=col, ascending=False)

        # Buat plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_sorted,
            x=col,
            y="car_make",
            order=df_sorted["car_make"],  # jaga urutan y-axis sesuai sorting
        )

        # Label dan judul
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel("Car Make")
        plt.title(f"{col.replace('_', ' ').title()} by Car Make")
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Data Preprocessing""")
    return


@app.cell
def _(df, pd):
    car_make_one_hot = pd.get_dummies(df["car_make"]).astype(int)
    car_make_one_hot = pd.concat([df, car_make_one_hot], axis=1)
    car_make_one_hot.head()

    car_make_one_hot.to_csv("car_make_one_hot.csv", index=False)
    return (car_make_one_hot,)


@app.cell
def _(car_make_one_hot, train_test_split):
    X_reg = car_make_one_hot.drop(["price_(in_usd)", "car_make"], axis=1)
    y_reg = car_make_one_hot["price_(in_usd)"]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    return X_reg, X_reg_test, X_reg_train, y_reg_test, y_reg_train


@app.cell
def _(X_reg):
    X_reg.head()
    return


@app.cell
def _(
    StandardScaler,
    X_reg_test,
    X_reg_train,
    joblib,
    pd,
    y_reg_test,
    y_reg_train,
):
    reg_scaler = StandardScaler()
    X_reg_train_scaled = reg_scaler.fit_transform(X_reg_train)
    X_reg_test_scaled = reg_scaler.transform(X_reg_test)

    X_reg_train_scaled = pd.DataFrame(
        X_reg_train_scaled, columns=X_reg_train.columns
    )
    X_reg_test_scaled = pd.DataFrame(X_reg_test_scaled, columns=X_reg_test.columns)

    y_reg_scaler = StandardScaler()
    y_reg_train_scaled = y_reg_scaler.fit_transform(
        y_reg_train.values.reshape(-1, 1)
    )
    y_reg_test_scaled = y_reg_scaler.transform(y_reg_test.values.reshape(-1, 1))

    y_reg_train_scaled = pd.DataFrame(
        y_reg_train_scaled, columns=["price_(in_usd)"]
    )
    y_reg_test_scaled = pd.DataFrame(y_reg_test_scaled, columns=["price_(in_usd)"])

    joblib.dump(reg_scaler, "reg_scaler.pkl")
    joblib.dump(y_reg_scaler, "y_reg_scaler.pkl")
    return (
        X_reg_test_scaled,
        X_reg_train_scaled,
        y_reg_test_scaled,
        y_reg_train_scaled,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Regression Model - Predict Price""")
    return


@app.cell
def _(LinearRegression, X_reg_train_scaled, joblib, y_reg_train_scaled):
    reg_model = LinearRegression()
    reg_model.fit(X_reg_train_scaled, y_reg_train_scaled)

    joblib.dump(reg_model, "reg_model.pkl")
    return (reg_model,)


@app.cell
def _(X_reg_test_scaled, reg_model):
    reg_pred = reg_model.predict(X_reg_test_scaled)
    reg_pred.shape
    return (reg_pred,)


@app.cell
def _(
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    np,
    r2_score,
):
    def regression_report(y_true, y_pred):
        print("📊 Regression Report:")
        print(f"R² Score               : {r2_score(y_true, y_pred):.4f}")
        print(
            f"Mean Absolute Error    : {mean_absolute_error(y_true, y_pred):.4f}"
        )
        print(f"Mean Squared Error     : {mean_squared_error(y_true, y_pred):.4f}")
        print(
            f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}"
        )
        print(
            f"Median Absolute Error  : {median_absolute_error(y_true, y_pred):.4f}"
        )
        print(
            f"Mean Absolute % Error  : {mean_absolute_percentage_error(y_true, y_pred):.4f}"
        )
    return (regression_report,)


@app.cell
def _(reg_pred, regression_report, y_reg_test_scaled):
    regression_report(y_reg_test_scaled, reg_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Classification Model - Predict Car Make""")
    return


@app.cell
def _(LabelEncoder, df, joblib, train_test_split):
    X_clf = df.drop("car_make", axis=1)
    y_clf = df["car_make"]

    le = LabelEncoder()
    y_clf = le.fit_transform(y_clf)

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    joblib.dump(le, "car_make_label_encoder.pkl")
    return X_clf_test, X_clf_train, y_clf_test, y_clf_train


@app.cell
def _(StandardScaler, X_clf_test, X_clf_train, joblib, pd):
    clf_scaler = StandardScaler()
    X_clf_train_scaled = clf_scaler.fit_transform(X_clf_train)
    X_clf_test_scaled = clf_scaler.transform(X_clf_test)

    X_clf_train_scaled = pd.DataFrame(
        X_clf_train_scaled, columns=X_clf_train.columns
    )
    X_clf_test_scaled = pd.DataFrame(X_clf_test_scaled, columns=X_clf_test.columns)

    joblib.dump(clf_scaler, "clf_scaler.pkl")
    return X_clf_test_scaled, X_clf_train_scaled


@app.cell
def _(X_clf_train_scaled, joblib, xgboost, y_clf_train):
    clf_model = xgboost.XGBClassifier()
    clf_model.fit(X_clf_train_scaled, y_clf_train)

    joblib.dump(clf_model, "clf_model.pkl")
    return (clf_model,)


@app.cell
def _(X_clf_test_scaled, clf_model):
    clf_pred = clf_model.predict(X_clf_test_scaled)
    return (clf_pred,)


@app.cell
def _(classification_report, clf_pred, y_clf_test):
    print(classification_report(y_clf_test, clf_pred))
    return


@app.cell
def _(X_clf_train, clf_model, pd):
    feat_importance = pd.Series(
        clf_model.feature_importances_, index=X_clf_train.columns
    ).sort_values(ascending=False)

    feat_importance
    return


if __name__ == "__main__":
    app.run()
