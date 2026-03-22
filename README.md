# ISRO Satellite Telemetry Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI model for detecting anomalies in satellite telemetry data, designed for **ISRO** mission monitoring and predictive maintenance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model](#model)
- [Output](#output)
- [License](#license)

## Overview

This project implements multivariate anomaly detection on satellite telemetry streams (temperature, voltage, current, gyroscope, etc.) using **Isolation Forest**, suitable for scenarios where labeled anomalies are scarce.

References:
- [Kaggle Kernel](https://www.kaggle.com/code/devraai/satellite-telemetry-anomaly-prediction) · devraai/satellite-telemetry-anomaly-prediction
- [Kaggle Dataset](https://www.kaggle.com/datasets/orvile/satellite-telemetry-data-anomaly-prediction) · orvile/satellite-telemetry-data-anomaly-prediction

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/telemetry-anomaly-detection.git
cd telemetry-anomaly-detection
pip install -r requirements.txt
```

## Quick Start

```bash
python telemetry_anomaly_model.py
```

If no dataset is found, the script generates synthetic ISRO-style telemetry for demonstration.

## Dataset

**Option A: Kaggle CLI**

```bash
pip install kaggle
# Add kaggle.json to ~/.kaggle/ or %USERPROFILE%\.kaggle\

kaggle kernels pull devraai/satellite-telemetry-anomaly-prediction -p .
kaggle datasets download orvile/satellite-telemetry-data-anomaly-prediction
```

**Option B: Setup script**

```bash
python setup_kaggle.py
```

**Option C: Manual**

1. Download from [Kaggle](https://www.kaggle.com/datasets/orvile/satellite-telemetry-data-anomaly-prediction)
2. Extract the CSV into the project folder or `satellite-telemetry-data-anomaly-prediction/`

## Project Structure

```
telemetry_anomaly_detection/
├── telemetry_anomaly_model.py   # Main model
├── setup_kaggle.py              # Kaggle data downloader
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── output/                      # Generated (gitignored)
    ├── anomaly_predictions.csv
    └── anomaly_visualization.png
```

## Model

| Component    | Choice                    |
|-------------|---------------------------|
| Algorithm   | Isolation Forest          |
| Features    | All numeric telemetry cols|
| Preprocessing | StandardScaler          |

Suitable for extension with LSTM, autoencoders, or other time-series approaches.

## Output

| File                     | Description                          |
|--------------------------|--------------------------------------|
| `output/anomaly_predictions.csv`  | Predictions + anomaly scores |
| `output/anomaly_visualization.png`| Plots of anomalies            |

## License

[MIT](LICENSE)
