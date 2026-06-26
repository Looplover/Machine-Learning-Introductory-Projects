import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def preprocess_data_sklearn(train_df, val_df, test_df):
    X_train_raw, y_train = train_df.drop('result', axis=1), train_df['result']
    X_val_raw, y_val = val_df.drop('result', axis=1), val_df['result']
    X_test_raw, y_test = test_df.drop('result', axis=1), test_df['result']

    cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train_raw.select_dtypes(include=np.number).columns

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_train_cat_encoded = enc.fit_transform(X_train_raw[cat_cols])
    X_val_cat_encoded = enc.transform(X_val_raw[cat_cols])
    X_test_cat_encoded = enc.transform(X_test_raw[cat_cols])
    
    X_train = np.hstack((X_train_raw[num_cols].values, X_train_cat_encoded))
    X_val = np.hstack((X_val_raw[num_cols].values, X_val_cat_encoded))
    X_test = np.hstack((X_test_raw[num_cols].values, X_test_cat_encoded))

    return X_train, y_train, X_val, y_val, X_test, y_test

def main(train_path, val_path, test_path, output_csv_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_sklearn(train_df, val_df, test_df)

    print("Running Part F...")
    param_grid = {
        'n_estimators': [50, 150, 250, 350],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [2, 4, 6, 8, 10]
    }

    rf = RandomForestClassifier(criterion='entropy', random_state=42, oob_score=True)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("\nGrid search complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    
    best_rf = grid_search.best_estimator_
    train_pred = best_rf.predict(X_train)
    val_pred = best_rf.predict(X_val)
    test_pred = best_rf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    oob_acc = best_rf.oob_score_

    print(f"\nOptimal Model Performance:")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  OOB Accuracy: {oob_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test Accuracy:       {test_acc:.4f}")

    output_df = pd.DataFrame({'result': test_pred})
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions from best Random Forest model saved to {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python f.py <train_data_path> <val_data_path> <test_data_path> <output_csv_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    output_dir = os.path.dirname(output_csv_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(train_path, val_path, test_path, output_csv_path)