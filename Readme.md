# Feature Propagation for Missing Value Imputation in Graph Data

## Overview
This project implements a feature propagation algorithm to impute missing values in graph-structured data. 

## Usage
### Dependencies
```bash
pip install -r requirements.txt
```
### Running the script
Execute the script using:
```bash
python main.py
```

### Notes
- The `fill_data` function randomizes missing values for demonstration. If working with real-world missing data, remove this randomization.
- The script will load Cora dataset, in order to load your own dataset, you can modify the `main.py` file and change the `data` variable to your dataset (edges list and feature table).

