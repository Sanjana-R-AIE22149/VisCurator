# VisCurator Testing & Analysis Suite

This directory contains testing scripts designed to help you generate rigorous data, metrics, and plots for your report or thesis.

## Prerequisites
Ensure your virtual environment is active and you have the required packages:
```bash
pip install pytest httpx requests pandas matplotlib seaborn
```

Ensure the VisCurator backend is running on `http://localhost:8000` for the API and performance tests.

## 1. API Integration Testing (`test_api_integration.py`)
This script uses `pytest` to automatically hit key endpoints of your application and verify their status. 
This is great for the **"System Testing"** section of your report to prove reliability.

**Run it:**
```bash
pytest test_api_integration.py -v
```

## 2. Performance Testing (`test_performance.py`)
This script simulates a series of API calls and measures the latency (response time). This provides quantitative data for a **"Performance Evaluation"** section.

**Run it:**
```bash
python test_performance.py
```

## 3. Results & Analysis Generator (`generate_analysis_charts.py`)
This script analyzes the training metrics (e.g. from `training_metrics.db`) and generates publication-ready charts (PNG files). If the database is missing, it will generate *simulated* template charts to show you how the plots will look.

**Run it:**
```bash
python generate_analysis_charts.py
```
Check the generated `report_charts/` directory for the plots to include in your document.
