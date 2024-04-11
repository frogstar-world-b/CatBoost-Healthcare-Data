from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


y_regrssion = "total_cost"
y_classification = "treatment"
date_col = "date"


def analyze_column_uniques(df):
    """Gets the dtype and number of unique values in each column.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        out (pandas.DataFrame): A DataFrame with three columns:
          * Column Name: The name of the column in the original DataFrame.
          * Dtype: The data type of the column.
          * Unique Values: The number of unique values in the column.
    """
    results = []
    for col in df.columns:
        col_name = col
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        results.append([col_name, dtype, n_unique])

    out = pd.DataFrame(data=results, columns=["Column Name", "Dtype", "Unique Values"])
    return out


def inspect_unique_values(df, col_stats, n_unique_values):
    """Tracks the unique values across columns with n_unique_values.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        col_stats (pandas.Series): Output of analyze_column_uniques().
        n_unique_values (int): Number of unique values

    Returns:
        values_dict (dict):
          * Keys: Tuple of unique values.
          * Values: Number of columns having these unique values.
    """
    cols = col_stats[col_stats["Unique Values"] == n_unique_values]["Column Name"]
    values_dict = {}
    for col in cols:
        u = tuple(np.sort(df[col].unique()))
        if u in values_dict.keys():
            values_dict[u] += 1
        else:
            values_dict[u] = 1
    return values_dict


def plot_x_by_labels(df, x, save_plot=False):
    """Plots column x by each of the two labels.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        x (str): Column name.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(x)

    ax1.scatter(df[x], df[y_regrssion], s=10)
    ax1.set_ylabel(y_regrssion)
    plt.setp(ax1.get_xticklabels(), rotation=50)

    ax2.scatter(df[x], df[y_classification], s=10)
    ax2.set_ylabel(y_classification)
    plt.setp(ax2.get_xticklabels(), rotation=50)

    # Don't obtain bar plot for predictors with more than 24 unique values
    if df[x].nunique() < 25:
        grouped_data = df.groupby(x)[y_classification].value_counts().unstack()
        grouped_data.plot(kind="bar", ax=ax3, rot=50)
        ax3.set_xlabel("")
        ax3.set_ylabel("Count")
        ax3.legend(title=y_classification)
    else:
        ax3.set_visible(False)

    plt.tight_layout()

    if save_plot:
        filename = f"{x}_vs_labels.png"
        filepath = "plots/" + filename
        plt.savefig(filepath)


def inspect_features_correlations(df, drop_columns):
    """Remves some columns then numerically encodes categorical columns and
    obtains the correlation matrix.

    Args:
        df (pd.DataFrame): A DataFrame to analyze.
        drop_columns (list): A list of column names (strings) to drop.

    Returns:
        corr (pd.DataFrame): The correlation matrix.
    """
    df = df.drop(drop_columns, axis=1)
    object_columns = df.select_dtypes(include=["object"]).columns
    print("Number of remaining columns:", len(object_columns))

    encoder = LabelEncoder()
    for col in object_columns:
        df[col] = encoder.fit_transform(df[col])

    corr = df.corr()
    plt.figure(figsize=(10, 7))  # Optional: Adjusts the size of the figure
    sns.heatmap(corr, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)

    return corr


def get_highly_correlated_features(corr, threshold=0.8):
    """Identifies pairs of features in a correlation matrix that are
    either highly correlated or highly anti-correlated beyond a threshold.
    The function filters out the lower triangle of the correlation matrix
    to avoid duplicate pairs of features.

    Args:
        corr (DataFrame): A square DataFrame representing a correlation
            matrix where the index and columns are feature names.
        threshold (float, optional): The correlation coefficient value above
            which a pair of features is considered highly correlated or below
            which it is considered highly anti-correlated.

    Returns:
        DataFrame: A DataFrame containing 3 columns: 'x1', 'x2', and 'corr',
        where 'x1' and 'x2' are the names of the pairs of features and 'corr'
        is their corresponding correlation coefficient.
    """
    # Filter upper triangle excluding diagonal
    corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))

    # Filter for correlations above and below thresholds
    high_corr = corr_triu[corr_triu > threshold].stack()
    low_corr = corr_triu[corr_triu < -1 * threshold].stack()

    # Print results
    if not high_corr.empty:
        print(f"Highly Correlated Features (above {threshold}):")
        print(high_corr)
    else:
        print(f"No features with correlation above {threshold}")

    if not low_corr.empty:
        print(f"Highly Anti-Correlated Features (below -{threshold}):")
        print(low_corr)
    else:
        print(f"No features with correlation below -{threshold}")

    # Stack the two dataframes
    out = pd.concat([high_corr, low_corr], axis=0).reset_index()
    out = out.rename(columns={"level_0": "x1", "level_1": "x2", 0: "corr"})

    return out


def get_train_test_sets(
    df,
    label_col,
    date_col=date_col,
    drop_cols=[y_regrssion, y_classification, date_col],
    split_type="Random",
    test_size=0.20,
    cutoff_date="2021-10-31",
):
    """Splits the data into training and testing sets udner two scenarios.
    The first is random split; the second is sequential, where the data before
    a cutoff date are for training and after the cutoff are for testing.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        label_col (str): The name of the column containing the target labels.
        date_col (str, optional): The name of the column with date values.
        drop_cols (list, optional): A list of column names to drop.
        split_type (str, optional): The type of split to perform. Must be
            either "Random" or "Sequential".
        cutoff_date (pandas.Timestamp, optional): The date used to separate
            the data into training and testing sets for the sequential split.

    Returns:
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]:
      A tuple containing the training and testing sets for features (X) and labels (y).
    """
    if split_type == "Random":
        y = df[label_col]
        X = df.drop(drop_cols, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    elif split_type == "Sequential":
        cutoff_date = pd.Timestamp(cutoff_date)
        train_df = df[df[date_col] <= cutoff_date]
        test_df = df[df[date_col] > cutoff_date]
        y_train = train_df[label_col]
        X_train = train_df.drop(drop_cols, axis=1)
        y_test = test_df[label_col]
        X_test = test_df.drop(drop_cols, axis=1)
    else:
        raise ValueError(
            'split_type parameter must be either "Random" or "Sequential".'
        )

    return X_train, X_test, y_train, y_test


def fit_default_catboostclassifier_model(
    X_train, X_test, y_train, y_test, categorical_features
):
    """Fits a CatBoostClassifier model, makes predictions on test data,
    and evaluates its performance.

    Args:
      X_train (pandas.DataFrame): The training set features.
      X_test (pandas.DataFrame): The testing set features.
      y_train (pandas.Series): The training set labels.
      y_test (pandas.Series): The testing set labels.
      categorical_features (list): A list of column names for categorical features.

    Returns:
      model (CatBoostClassifier model): fitted model
    """
    # Define the CatBoost model
    model = CatBoostClassifier(loss_function="Logloss", random_seed=42, silent=True)
    # Train the model
    model.fit(X_train, y_train, cat_features=categorical_features)
    # Make predictions on new data (replace with your actual data)
    y_pred = model.predict(X_test)

    acc, prec, recall, f1, auc = evaluate_classification_performance(
        model, y_test, y_pred, X_test
    )

    return model, acc, prec, recall, f1, auc


def evaluate_classification_performance(model, y_test, y_pred, X_test):
    """Evaluates the performance of a fitted  model.

    Args:
      model: The fitted machine learning model (must have a predict method).
      y_test (pandas.Series): The ground truth labels for the test data.
      y_pred (pandas.Series): The predicted labels on the test data.
      X_test (pandas.Series): The ground truth features for the test data.

    Returns:
      tuple[float, float, float, float, float]: A tuple containing the
          evaluation metrics (accuracy, precision, recall, F1-score, and AUC).
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Assuming the positive class is at index 1
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.4f}")

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC)")

    return accuracy, precision, recall, f1, auc


def plot_feature_importances(model, n_features=30):
    """Plots the n most important features.

    Args:
    model (CatBoostClassifier): The trained CatBoostClassifier model.
    n_features (int, optional): The number of top features to display (default: 50).
    plot_tag (str): A string tag to be included in the ROC curve title.

    Returns:
    feature_importances (pandas.DataFrame): A DataFrame containing the top n features.
    """
    # For feature names instead of indices use prettified=True
    feature_importances = model.get_feature_importance(prettified=True)
    feature_importances = feature_importances.sort_values(
        by="Importances", ascending=False
    )
    feature_importances_top_n = feature_importances.head(n_features)
    plt.figure(figsize=(10, 10))
    plt.barh(
        feature_importances_top_n["Feature Id"],
        feature_importances_top_n["Importances"],
    )
    plt.xlabel("Feature Importance")
    plt.title(f"Top {n_features} most important features")
    plt.grid()

    return feature_importances


def transform_regression_label(y, mu, sigma):
    """Transforms a regression label y using a log1p normalization with
    parameters mu (mean) and sigma (standard deviation).
    Log1p is used to handle cases where y might be close to zero,
    avoiding potential issues with logarithms.

    Args:
        y (numpy.ndarray): The regression label array to be transformed.
        mu (float): The mean value for normalization.
        sigma (float): The standard deviation value for normalization.

    Returns:
        numpy.ndarray: The transformed regression label array.
    """
    x = (np.log1p(y) - mu) / sigma
    return x


def inverse_transform_regression_label(x, mu, sigma):
    """Inverts a transformed regression label x back to the original scale."""
    y = np.exp(mu + sigma * x) - 1
    return y


def test_transform_and_inverse():
    z_in = np.array([0.0, 1.0, 10.0, 100.0])
    z_inv = transform_regression_label(z_in, 1, 2)
    z_out = inverse_transform_regression_label(z_inv, 1, 2)

    np.testing.assert_allclose(z_in, z_out)


def evaluate_regression_performance(y_test, y_pred, X_test, y_test_raw, y_pred_raw):
    """Evaluates the performance of a fitted regression model.

    Args:
      model: The fitted machine learning model (must have a predict method).
      y_test (pandas.Series): The transformed ground truth labels for the test data.
      y_pred (pandas.Series): The predicted transfomred labels on the test data.
      X_test (pandas.Series): The ground truth features for the test data.
      y_test_raw (pandas.Series): The (raw) ground truth labels for the test data.

    Returns:
      tuple[float, float, float, float]: A tuple containing the
          evaluation metrics (RMSE, MAE, R2, SpearmanR).
    """
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    SpearR = spearmanr(y_test, y_pred)[0]

    RMSE_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    MAE_raw = mean_absolute_error(y_test_raw, y_pred_raw)
    R2_raw = r2_score(y_test_raw, y_pred_raw)
    SpearR_raw = spearmanr(y_test_raw, y_pred_raw)[0]

    print(
        f"Transformed scale (RMSE: {RMSE:.4f}, MAE: {MAE:.4f}, R2: {R2:.4f}, SpearmanR: {SpearR:.4f})"
    )
    print(
        f"Original scale (RMSE: {RMSE_raw:.4f}, MAE: {MAE_raw:.4f}, R2: {R2_raw:.4f}, SpearmanR: {SpearR_raw:.4f})"
    )
    print("\n")

    # Plot predicted vs actual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.scatter(y_pred, y_test, s=10)
    ax1.set_xlabel("Predicted total_cost_future (Transformed)")
    ax1.set_ylabel("Actual total_cost_future (Transformed)")
    ax1.set_title(f"Predicted vs Actual (Transformed)")

    ax2.scatter(y_pred_raw, y_test_raw, s=10)
    ax2.set_xlabel("Predicted total_cost_future (Original scale)")
    ax2.set_ylabel("Actual total_cost_future (Original scale)")
    ax2.set_title(f"Predicted vs Actual (Original Scale)")

    # Add 45-degree lines
    ax1.plot(ax1.get_xlim(), ax1.get_ylim(), linestyle="--", color="gray", alpha=0.7)
    ax2.plot(ax2.get_xlim(), ax2.get_ylim(), linestyle="--", color="gray", alpha=0.7)

    plt.tight_layout()

    return RMSE_raw, MAE_raw, R2_raw, SpearR_raw


def fit_default_catboostregressor_model(
    X_train, X_test, y_train, y_test, y_test_raw, mu, sigma, categorical_features
):
    """Fits a CatBoostRegressor model, makes predictions on test data,
    and evaluates its performance.

    Args:
      X_train (pandas.DataFrame): The training set features.
      X_test (pandas.DataFrame): The testing set features.
      y_train (pandas.Series): The training set labels.
      y_test (pandas.Series): The testing set labels.
      mu (float): Tranformation parameter (Expected value on the log scale).
      sigma (float): Tranformation parameter (Std Dev value on the log scale).
      categorical_features (list): A list of column names for categorical features.

    Returns:
      tuple[model, float, float, float, float]: A tuple containing the
          model object and the evaluation metrics (RMSE, MAE, R2, SpearmanR).

    """
    # Define the CatBoost model
    model = CatBoostRegressor(loss_function="RMSE", random_seed=42, silent=True)
    # Train the model
    model.fit(X_train, y_train, cat_features=categorical_features)
    # Make predictions on new data (replace with your actual data)
    y_pred = model.predict(X_test)
    y_pred_raw = inverse_transform_regression_label(y_pred, mu, sigma)

    RMSE_raw, MAE_raw, R2_raw, SpearR_raw = evaluate_regression_performance(
        y_test, y_pred, X_test, y_test_raw, y_pred_raw
    )

    return model, RMSE_raw, MAE_raw, R2_raw, SpearR_raw
