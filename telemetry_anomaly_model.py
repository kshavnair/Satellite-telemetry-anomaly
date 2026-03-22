"""Satellite Telemetry Anomaly Detection Model
================================================
AI model for detecting anomalies in satellite telemetry data.
Designed for monitoring and predictive maintenance.

References:
- Kaggle: devraai/satellite-telemetry-anomaly-prediction
- Dataset: orvile/satellite-telemetry-data-anomaly-prediction
"""

import os
import glob
import zipfile
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")


def load_data(data_path=None):
    """
    Load telemetry data. Supports multiple paths:
    - Extracted CSV in data folder
    - Downloaded zip from Kaggle
    """
    project_dir = Path(__file__).parent
    
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    
    # Check for CSV in data subfolder
    data_dir = project_dir / "satellite-telemetry-data-anomaly-prediction"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])
    
    # Check for zip file
    zip_files = list(project_dir.glob("*.zip"))
    for zf in zip_files:
        with zipfile.ZipFile(zf, 'r') as z:
            names = z.namelist()
            csv_name = next((n for n in names if n.endswith('.csv')), None)
            if csv_name:
                return pd.read_csv(z.open(csv_name))
    
    # Generate synthetic data for demonstration if no dataset found
    print("No dataset found. Generating synthetic ISRO-style telemetry data...")
    return generate_synthetic_telemetry()


def generate_synthetic_telemetry(n_samples=10000):
    """
    Generate synthetic satellite telemetry for demonstration.
    Simulates: temperature, voltage, current, gyro, pressure, power.
    """
    np.random.seed(42)
    n = n_samples
    
    # Normal telemetry (multivariate)
    temp = 25 + np.random.randn(n) * 3
    voltage = 28 + np.random.randn(n) * 0.5
    current = 2.5 + np.random.randn(n) * 0.3
    gyro_x = np.random.randn(n) * 0.1
    gyro_y = np.random.randn(n) * 0.1
    gyro_z = np.random.randn(n) * 0.1
    
    # Introduce anomalies (~5%)
    n_anomalies = int(n * 0.05)
    anomaly_idx = np.random.choice(n, n_anomalies, replace=False)
    
    temp[anomaly_idx] += np.random.choice([-1, 1], n_anomalies) * np.random.uniform(10, 25, n_anomalies)
    voltage[anomaly_idx] += np.random.uniform(-2, 2, n_anomalies)
    current[anomaly_idx] += np.random.uniform(-0.5, 0.5, n_anomalies)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='min'),
        'temperature': temp,
        'voltage': voltage,
        'current': current,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
    })
    
    # Create ground truth if not present
    if 'anomaly' not in df.columns and 'is_anomaly' not in df.columns:
        y_true = np.zeros(n)
        y_true[anomaly_idx] = 1
        df['anomaly'] = y_true
    
    return df


def prepare_features(df, feature_cols=None):
    """Prepare feature matrix for anomaly detection."""
    exclude = ['timestamp', 'date', 'time', 'id', 'anomaly', 'is_anomaly', 'label', 'target']
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.lower() not in exclude and df[c].dtype in ['int64', 'float64']]
    
    if not feature_cols:
        # Use all numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c.lower() not in exclude]
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    return X, feature_cols


def detect_anomalies(X, contamination=0.05, random_state=42):
    """
    Use Isolation Forest for anomaly detection.
    Well-suited for multivariate telemetry with no labeled anomalies.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
    )
    predictions = model.fit_predict(X_scaled)
    
    # Convert to binary: -1 (anomaly) -> 1, 1 (normal) -> 0
    anomaly_labels = (predictions == -1).astype(int)
    anomaly_scores = -model.score_samples(X_scaled)  # Higher = more anomalous
    
    return anomaly_labels, anomaly_scores, model, scaler


def evaluate(y_true, y_pred):
    """Compute evaluation metrics if ground truth is available."""
    if y_true is None or len(np.unique(y_true)) < 2:
        return None
    
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return cm


def plot_results(df, anomaly_labels, anomaly_scores, feature_cols, save_path=None):
    """Visualize telemetry and detected anomalies."""
    n_plots = min(4, len(feature_cols)) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot first few telemetry channels with anomalies highlighted
    for i, col in enumerate(feature_cols[:n_plots - 1]):
        ax = axes[i]
        ax.plot(df.index, df[col], 'b-', alpha=0.7, linewidth=0.8)
        anomaly_mask = anomaly_labels == 1
        ax.scatter(df.index[anomaly_mask], df[col][anomaly_mask], 
                   c='red', s=20, alpha=0.8, label='Anomaly', zorder=5)
        ax.set_ylabel(col)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot anomaly score
    ax = axes[-1]
    ax.plot(df.index, anomaly_scores, 'g-', alpha=0.8, linewidth=0.8)
    ax.axhline(y=np.percentile(anomaly_scores, 95), color='r', linestyle='--', alpha=0.7, label='95th percentile')
    ax.set_ylabel('Anomaly Score')
    ax.set_xlabel('Sample Index')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('ISRO Satellite Telemetry - Anomaly Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if os.environ.get("CI"):
        plt.close()
    else:
        plt.show()


def main():
    print("=" * 60)
    print("ISRO Satellite Telemetry Anomaly Detection")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} samples, {len(df.columns)} columns")
    
    # Prepare features
    X, feature_cols = prepare_features(df)
    print(f"Using features: {feature_cols}")
    
    # Check for ground truth
    y_true = None
    for col in ['anomaly', 'is_anomaly', 'label', 'target']:
        if col in df.columns:
            y_true = df[col].values
            break
    
    # Detect anomalies
    contamination = 0.05  # Expected proportion of anomalies (~5%)
    anomaly_labels, anomaly_scores, model, scaler = detect_anomalies(X, contamination=contamination)
    
    n_anomalies = anomaly_labels.sum()
    print(f"\nDetected {n_anomalies} anomalies ({100*n_anomalies/len(df):.2f}%)")
    
    # Evaluate if ground truth available
    if y_true is not None:
        evaluate(y_true, anomaly_labels)
    
    # Add predictions to dataframe
    df['predicted_anomaly'] = anomaly_labels
    df['anomaly_score'] = anomaly_scores
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "anomaly_predictions.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Plot
    plot_path = output_dir / "anomaly_visualization.png"
    plot_results(df, anomaly_labels, anomaly_scores, feature_cols, save_path=plot_path)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return df, model, scaler


if __name__ == "__main__":
    main()
