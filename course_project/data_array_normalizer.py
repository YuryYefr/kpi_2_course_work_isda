import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def data_array_normalization(model, X_test, y_test, data_frame):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    # Ensure the arrays have the same length and index alignment
    y_pred_aligned = y_pred[:len(y_test)]  # Trim y_pred to match length of y_test
    df_aligned = data_frame.loc[y_test.index]  # Align the dataframe with y_test

    # If there are any missing values in y_pred or y_test, drop them
    mask = ~np.isnan(y_pred_aligned) & ~np.isnan(y_test)
    y_pred_aligned = y_pred_aligned[mask]
    y_test = y_test[mask]
    df_aligned = df_aligned[mask]
    return y_pred_aligned, y_test, df_aligned





