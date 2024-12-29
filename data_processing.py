import logging
from datetime import datetime

import joblib
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')
# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create handlers
file_handler = logging.FileHandler(f'log/data_processing_{current_date}.log')
console_handler = logging.StreamHandler()
# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Load the data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        df.head()
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


# Preprocess the data
def preprocess_data(df):
    try:
        logging.info("Starting data preprocessing...")
        # Remove work year 2020 and 2021
        new_df = df[~df['work_year'].isin([2020, 2021])]
        new_df.head()
        # Drop unnecessary columns
        columns_to_drop = ['salary', 'salary_currency', 'remote_ratio']
        new_df = new_df.drop(columns=columns_to_drop)

        new_df.head()

        # Drop missing values and duplicates
        new_df = new_df.dropna()

        logging.info("Data preprocessing completed.")
        return new_df
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return None


def plot_predictions_vs_actual(y_true, y_pred_lr, y_pred_ridge, y_pred_dt, y_pred_rf, y_pred_poly, y_pred_combined):
    plt.figure(figsize=(14, 7))

    # Linear Regression
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=y_true, y=y_pred_lr)
    plt.title("Linear Regression: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Ridge Regression
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=y_true, y=y_pred_ridge)
    plt.title("Ridge Regression: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Decision Tree
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=y_true, y=y_pred_dt)
    plt.title("Decision Tree: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Random Forest
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=y_true, y=y_pred_rf)
    plt.title("Random Forest: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Polynomial Regression
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=y_true, y=y_pred_poly)
    plt.title("Polynomial Regression: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Combined Prediction
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=y_true, y=y_pred_combined)
    plt.title("Combined Prediction: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred_lr, y_pred_ridge, y_pred_dt, y_pred_rf, y_pred_poly, y_pred_combined):
    plt.figure(figsize=(14, 7))

    # Linear Regression Residuals
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=y_pred_lr, y=y_true - y_pred_lr)
    plt.title("Linear Regression: Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    # Ridge Regression Residuals
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=y_pred_ridge, y=y_true - y_pred_ridge)
    plt.title("Ridge Regression: Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    # Decision Tree Residuals
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=y_pred_dt, y=y_true - y_pred_dt)
    plt.title("Decision Tree: Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    # Random Forest Residuals
    plt.subplot(2, 3, 4)
    sns.scatterplot(x=y_pred_rf, y=y_true - y_pred_rf)
    plt.title("Random Forest: Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    # Polynomial Regression Residuals
    plt.subplot(2, 3, 5)
    sns.scatterplot(x=y_pred_poly, y=y_true - y_pred_poly)
    plt.title("Polynomial Regression: Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    # Combined Prediction Residuals
    plt.subplot(2, 3, 6)
    sns.scatterplot(x=y_pred_combined, y=y_true - y_pred_combined)
    plt.title("Combined Prediction: Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    plt.tight_layout()
    plt.show()


def encode_and_handle_outliers(df, categorical_columns, column_name, threshold=1.5):
    try:
        # Label encode the categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

        # Handle outliers for numerical columns
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        logging.info("Categorical columns encoded and outliers handled.")
        return filtered_df
    except Exception as e:
        logging.error(f"Error encoding categorical columns or handling outliers: {e}")
        return None


def determine_salary_ranges(df):
    max_salary = df['salary_in_usd'].max()
    min_salary = df['salary_in_usd'].min()

    num_subranges = 15

    subranges = np.linspace(min_salary, max_salary, num=num_subranges + 1, endpoint=True)
    range_labels = []
    for i in range(len(subranges) - 1):
        subrange_min = int(subranges[i])
        subrange_max = int(subranges[i + 1])
        range_label = f"{subrange_min:,} - {subrange_max:,}"
        range_labels.append(range_label)
    return range_labels


def save_data(df, file_path):
    try:
        # Save the processed data
        df.to_csv(file_path, index=False)
        logging.info("Processed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


# divide the dataset for training and testing
def divide_dataset(df):
    try:
        # Split the data into features and target
        x = df.drop('salary_in_usd', axis=1)
        y = df['salary_in_usd']
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info("Dataset divided into training and testing sets.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error dividing dataset: {e}")
        return None, None, None, None


def linear_regression(x_train, y_train):
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    logging.info("Linear regression model fitted successfully.")
    joblib.dump(linreg, 'model/linear_regression.pkl')
    return linreg


def decision_tree(x_train, y_train):
    destree = DecisionTreeRegressor()
    destree.fit(x_train, y_train)
    logging.info("Decision tree model fitted successfully.")
    joblib.dump(destree, 'model/decision_tree.pkl')
    return destree


def random_forest(x_train, y_train):
    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(x_train, y_train)
    logging.info("Random forest model fitted successfully.")
    joblib.dump(forest, 'model/random_forest.pkl')
    return forest


def ridge_regression(x_train, y_train):
    param_grid = {'alpha': [0.1, 1.0, 10.0],

                  'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, scoring='r2', cv=10)
    grid_search.fit(x_train, y_train)
    best_ridge = grid_search.best_estimator_
    logging.info("Ridge model fitted successfully.")
    joblib.dump(best_ridge, 'model/ridge_regression.pkl')
    return best_ridge


def predict(linreg, destree, randforest, ridgereg, poly, weights_all, x_train, model):
    # give a test input and see the predictions
    test_input = pd.DataFrame([[4, 3, 2, 225, 88, 82, 1]], columns=x_train.columns)
    test_actual_salary = 283780

    print("Predictions for the test input:")
    # Predict using Linear Regression
    test_pred_lr = linreg.predict(test_input)
    print(f"Linear Regression Prediction: {test_pred_lr[0]}, Actual Salary: {test_actual_salary}")

    # Predict using Decision Tree
    test_pred_destree = destree.predict(test_input)
    print(f"Decision Tree Prediction: {test_pred_destree[0]}, Actual Salary: {test_actual_salary}")

    # Predict using Random Forest
    test_pred_randforest = randforest.predict(test_input)
    print(f"Random Forest Prediction: {test_pred_randforest[0]}, Actual Salary: {test_actual_salary}")

    # Predict using Ridge Regression
    test_pred_ridge = ridgereg.predict(test_input)
    print(f"Ridge Regression Prediction: {test_pred_ridge[0]}, Actual Salary: {test_actual_salary}")

    # Predict using Polynomial Regression
    test_input_poly = poly.transform(test_input)
    test_pred_poly = model.predict(test_input_poly)
    print(f"Polynomial Regression Prediction: {test_pred_poly[0]}, Actual Salary: {test_actual_salary}")

    # Combined prediction from all models including Ridge Regression
    combined_test_pred_all_ridge = (weights_all[0] * test_pred_lr +
                                    weights_all[1] * test_pred_destree +
                                    weights_all[2] * test_pred_randforest +
                                    weights_all[3] * test_pred_poly +
                                    weights_all[4] * test_pred_ridge)
    print(
        f"Combined Model (All with Ridge) Prediction: {combined_test_pred_all_ridge[0]}, Actual Salary: {test_actual_salary}")


def main():
    # Path to the data
    file_path = 'data/salaries.csv'
    processed_file_path = 'data/processed_salary_data.csv'

    # Load the data
    df = load_data(file_path)

    # Check if the data is loaded successfully
    if df is not None:
        # Logging information about the data
        logging.info("General information about the data:\n" + str(df.head()))
        logging.info("Details of the data:\n" + str(df.describe()))
        logging.info("Missing values:\n" + str(df.isna().sum()))

        # Check properties of the data
        # data_properties(df)

        df = preprocess_data(df)

        if df is not None:
            categorical_columns = ['work_year', 'experience_level', 'employment_type', 'job_title', 'company_size',
                                   'employee_residence', 'company_location']
            # Encode categorical columns and handle outliers
            new_df = encode_and_handle_outliers(df, categorical_columns, 'salary_in_usd')
            # new_df.info()
            range_labels = determine_salary_ranges(new_df)
            print("Ranges: ")
            print(range_labels)
            new_df.info()
            # Save the processed data
            save_data(new_df, processed_file_path)

            '''
            # duplicate df for plotting
            df_plot = new_df.copy()
            # Plot the data
            plot_correlation_matrix(df_plot)
            plot_education_level(df_plot)
            plot_gender_vs_education(df_plot)
            plot_numerical_columns(df_plot)
            plot_gender_vs_salary(df_plot)
            plot_histogram(df_plot)'''

            x_train, x_test, y_train, y_test = divide_dataset(new_df)

            if x_train is not None:
                try:
                    # Linear regression model training
                    linreg = linear_regression(x_train, y_train)
                    pred_lr = linreg.predict(x_test)

                    mse_lr = mean_squared_error(y_test, pred_lr)
                    mae_lr = mean_absolute_error(y_test, pred_lr)
                    r2_lr = r2_score(y_test, pred_lr)

                    print("----------------------")
                    print("Mean Squared Error (MSE) of Linear Regression:", mse_lr)
                    print("Mean Absolute Error (MAE) of Linear Regression:", mae_lr)
                    print("R-squared (R2) Score of Linear Regression:", r2_lr)

                    """
                    #overfitting check
                    train_pre = linreg.predict(x_train)
                    mse_over_li = mean_squared_error(y_train ,train_pre)
                    mae_over_li = mean_absolute_error(y_train, train_pre)
                    r2_over_li = r2_score(y_train, train_pre)

                    print("Overfitting check for Linear Regression")
                    print("Mean Squared Error (MSE) of Linear Regression:", mse_over_li)
                    print("Mean Absolute Error (MAE) of Linear Regression:", mae_over_li)
                    print("R-squared (R2) Score of Linear Regression:", r2_over_li)
                    """

                    # Decision tree model training
                    destree = decision_tree(x_train, y_train)
                    pred_destree = destree.predict(x_test)

                    mse_destree = mean_squared_error(y_test, pred_destree)
                    mae_destree = mean_absolute_error(y_test, pred_destree)
                    r2_destree = r2_score(y_test, pred_destree)
                    print("----------------------")
                    print("Mean Squared Error (MSE) of Decision Tree:", mse_destree)
                    print("Mean Absolute Error (MAE) of Decision Tree:", mae_destree)
                    print("R-squared (R2) Score of Decision Tree:", r2_destree)

                    """
                    #overfitting check
                    train_pre_destree = destree.predict(x_train)
                    mse_over_destree = mean_squared_error(y_train, train_pre_destree)
                    mae_over_destree = mean_absolute_error(y_train, train_pre_destree)
                    r2_over_destree = r2_score(y_train, train_pre_destree)

                    print("Overfitting check for Decision Tree:")
                    print("Mean Squared Error (MSE):", mse_over_destree)
                    print("Mean Absolute Error (MAE):", mae_over_destree)
                    print("R-squared (R2) Score:", r2_over_destree)
                    """

                    # Random Forrest model training
                    randforest = random_forest(x_train, y_train)
                    pred_randforest = randforest.predict(x_test)

                    mse_randforest = mean_squared_error(y_test, pred_randforest)
                    mae_randforest = mean_absolute_error(y_test, pred_randforest)
                    r2_randforest = r2_score(y_test, pred_randforest)
                    print("----------------------")
                    print("Mean Squared Error (MSE) of Random Forrest:", mse_randforest)
                    print("Mean Absolute Error (MAE) of Random Forrest:", mae_randforest)
                    print("R-squared (R2) Score of Random Forrest:", r2_randforest)

                    """
                    #overfitting check
                    train_pre_randforest = randforest.predict(x_train)
                    mse_over_randforest = mean_squared_error(y_train, train_pre_randforest)
                    mae_over_randforest = mean_absolute_error(y_train, train_pre_randforest)
                    r2_over_randforest = r2_score(y_train, train_pre_randforest)

                    print("Overfitting check for Random Forest:")
                    print("Mean Squared Error (MSE):", mse_over_randforest)
                    print("Mean Absolute Error (MAE):", mae_over_randforest)
                    print("R-squared (R2) Score:", r2_over_randforest)
                    """

                    # Ridge Regression model training
                    ridgereg = ridge_regression(x_train, y_train)
                    pred_ridge = ridgereg.predict(x_test)

                    mse_ridge = mean_squared_error(y_test, pred_ridge)
                    mae_ridge = mean_absolute_error(y_test, pred_ridge)
                    r2_ridge = r2_score(y_test, pred_ridge)
                    print("----------------------")
                    print("Mean Squared Error (MSE) of Ridge Regression:", mse_ridge)
                    print("Mean Absolute Error (MAE) of Ridge Regression:", mae_ridge)
                    print("R-squared (R2) Score of Ridge Regression:", r2_ridge)

                    """
                    #overfitting check for Ridge Regression
                    train_pre_ridge = ridgereg.predict(x_train)
                    mse_over_ridge = mean_squared_error(y_train, train_pre_ridge)
                    mae_over_ridge = mean_absolute_error(y_train, train_pre_ridge)
                    r2_over_ridge = r2_score(y_train, train_pre_ridge)

                    print("Overfitting check for Ridge Regression:")
                    print("Mean Squared Error (MSE):", mse_over_ridge)
                    print("Mean Absolute Error (MAE):", mae_over_ridge)
                    print("R-squared (R2) Score:", r2_over_ridge)
                    """

                    # Polynomial Regression model training
                    degree = 2
                    poly = PolynomialFeatures(degree=degree)
                    X_train_poly = poly.fit_transform(x_train)
                    X_test_poly = poly.transform(x_test)
                    model = LinearRegression()
                    model.fit(X_train_poly, y_train)
                    model.predict(X_train_poly)
                    y_test_pred = model.predict(X_test_poly)

                    logging.info("Polynomial regression model fitted successfully.")

                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(y_test, y_test_pred)
                    print("----------------------")
                    print("Polynomial Regression Test MSE:", test_mse)
                    print("Polynomial Regression Test RMSE:", test_rmse)
                    print("Polynomial Regression Test R2:", test_r2)

                    # overfitting
                    train_pre_poly = model.predict(X_train_poly)
                    mse_over_poly = mean_squared_error(y_train, train_pre_poly)
                    mae_over_poly = mean_absolute_error(y_train, train_pre_poly)
                    r2_over_poly = r2_score(y_train, train_pre_poly)
                    print("----------------------")
                    print("Overfitting check for Polynomial Regression:")
                    print("Mean Squared Error (MSE):", mse_over_poly)
                    print("Mean Absolute Error (MAE):", mae_over_poly)
                    print("R-squared (R2) Score:", r2_over_poly)

                    # Combine predictions from all models using weighted average including Ridge Regression
                    # Calculate weights, according to R2 measurements
                    r2_scores = [r2_lr, r2_destree, r2_randforest, test_r2, r2_ridge]
                    total_r2 = sum(r2_scores)
                    weights_all = [r2 / total_r2 for r2 in r2_scores]

                    combined_pred_all = (weights_all[0] * pred_lr +
                                         weights_all[1] * pred_destree +
                                         weights_all[2] * pred_randforest +
                                         weights_all[3] * y_test_pred +
                                         weights_all[4] * pred_ridge)

                    mse_combined_all = mean_squared_error(y_test, combined_pred_all)
                    mae_combined_all = mean_absolute_error(y_test, combined_pred_all)
                    r2_combined_all = r2_score(y_test, combined_pred_all)
                    print("----------------------")
                    print("Mean Squared Error (MSE) of Combined Model (All with Ridge):", mse_combined_all)
                    print("Mean Absolute Error (MAE) of Combined Model (All with Ridge):", mae_combined_all)
                    print("R-squared (R2) Score of Combined Model (All with Ridge):", r2_combined_all)

                    # Plot the predictions vs. actual values
                    plot_predictions_vs_actual(y_test, pred_lr, pred_ridge, pred_destree, pred_randforest, y_test_pred,
                                               combined_pred_all)

                    # Plot residuals
                    plot_residuals(y_test, pred_lr, pred_ridge, pred_destree, pred_randforest, y_test_pred,
                                   combined_pred_all)

                    """

                    # Polynomial Regression cross-validation
                    cv_scores_polyreg = cross_val_score(model, x_train, y_train, cv=10, scoring='r2')
                    print("Cross-Validation R2 Scores for Polynomial Regression:", cv_scores_polyreg)
                    print("Mean Cross-Validation R2 Score for Polynomial Regression:", np.mean(cv_scores_polyreg))

                    # Linear Regression cross-validation
                    cv_scores_lr = cross_val_score(linreg, x_train, y_train, cv=10, scoring='r2')
                    print("Cross-Validation R2 Scores for Linear Regression:", cv_scores_lr)
                    print("Mean Cross-Validation R2 Score for Linear Regression:", np.mean(cv_scores_lr))

                    # Decision Tree cross-validation
                    cv_scores_destree = cross_val_score(destree, x_train, y_train, cv=10, scoring='r2')
                    print("Cross-Validation R2 Scores for Decision Tree:", cv_scores_destree)
                    print("Mean Cross-Validation R2 Score for Decision Tree:", np.mean(cv_scores_destree))

                    # Random Forest cross-validation
                    cv_scores_randforest = cross_val_score(randforest, x_train, y_train, cv=10, scoring='r2')
                    print("Cross-Validation R2 Scores for Random Forest:", cv_scores_randforest)
                    print("Mean Cross-Validation R2 Score for Random Forest:", np.mean(cv_scores_randforest))

                    # Ridge Regression cross-validation
                    cv_scores_ridge = cross_val_score(ridgereg, x_train, y_train, cv=10, scoring='r2')
                    print("Cross-Validation R2 Scores for Ridge Regression:", cv_scores_ridge)
                    print("Mean Cross-Validation R2 Score for Ridge Regression:", np.mean(cv_scores_ridge))
                    """

                    predict(linreg, destree, randforest, ridgereg, poly, weights_all, x_train, model)

                    logging.info("Models created and predictions made successfully.")
                except Exception as e:
                    logging.error(f"Error creating model or making predictions: {e}")


# Start the program
if __name__ == "__main__":
    main()
