$ErrorActionPreference = "Stop"

Write-Host "Generating synthetic CTG smoke-test data..."
python .\scripts\00_setup\generate_synthetic_ctg.py --out .\data\example\synthetic_ctg_long.csv --records 80 --points 1800

Write-Host "Extracting CTG features..."
python .\scripts\03_features\extract_ctg_features.py --input .\data\example\synthetic_ctg_long.csv --out .\data\processed\ctg_feature_table.csv

Write-Host "Running baseline smoke model..."
python .\scripts\04_models\run_baseline_smoke.py --features .\data\processed\ctg_feature_table.csv --out .\results\tables\baseline_smoke_metrics.csv

Write-Host "Writing project status..."
python .\scripts\05_reports\write_project_status.py

Write-Host "Pipeline complete."

