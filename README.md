# LoRa Network Optimization System

**Predictive Optimization of LoRa Network Deployment**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [ Overview](#🎯-overview)
- [Key Features](#✨-key-features)
- [System Architecture](#🏗️-system-architecture)
- [Installation](#📦-installation)
- [Configuration Guide](#⚙️-configuration-guide)
- [Parameter Reference](#📖-parameter-reference)
- [Usage Examples](#💡-usage-examples)
- [Model Training](#🎓-model-training)
- [Technical Details](#🔬-technical-details)
- [Troubleshooting](#🔧-troubleshooting)
- [Additional Resources](#📜-additional-resources)
- [Contact & Support](#📞-contact-and-support)
- [Contributing](#🤝-contributing)
- [FAQ](#❓-faq)
- [License](#📄-license)

---

## 🎯 Overview

This system uses **machine learning** and **real-time satellite data** to optimize LoRa network deployments by:

1. **Predicting signal quality** (RSSI, SNR, path loss) using ensemble ML models
2. **Calculating packet delivery rate** (PDR) using LoRaWAN physics
3. **Finding optimal beacon placement** using A* pathfinding algorithm
4. **Fetching real terrain data** from Google Earth Engine (elevation, land cover)
5. **Visualizing results** with interactive HTML maps

### **Problem It Solves**

Deploying LoRa networks over long distances (>1 km) requires intermediate relay beacons. Manual placement is:
- ❌ Time-consuming (trial-and-error)
- ❌ Suboptimal (misses best paths)
- ❌ Expensive (wrong beacon count/locations)

This system **automatically** finds the optimal beacon positions considering:
- ✅ Real terrain (mountains, valleys)
- ✅ Land cover (buildings, water, forests)
- ✅ LoRa physics (spreading factor, TX power)
- ✅ Signal quality requirements (PDR thresholds)

---

## ✨ Key Features

### **1. Intelligent Routing**
- **Short distances (<1 km)**: Direct path (no beacons)
- **Long distances (≥1 km)**: A* optimization with beacons
- **Adaptive grid**: Automatically adjusts density based on distance

### **2. Real-Time Satellite Data**
- **Elevation**: SRTM 30m resolution (mountains, valleys)
- **Land cover**: ESA WorldCover 10m resolution (buildings, water, forests)
- **Batch fetching**: Parallel workers (5-10x faster than sequential)
- **Disk caching**: Reuses previously fetched data

### **3. Machine Learning Models**
- **Random Forest**: Fast, interpretable
- **XGBoost**: High accuracy, gradient boosting
- **Neural Network**: Deep learning with early stopping
- **Ensemble**: Weighted combination of all models
- **Auto-selection**: Best model chosen based on R² score

### **4. Physics-Based PDR**
- Calculated using LoRaWAN specifications (not ML predicted)
- Considers spreading factor, SNR threshold, land cover
- Exponential decay model: `PDR = 1 - exp(-k × margin)`

### **5. Comprehensive Validation**
- All 14 parameters validated with helpful error messages
- Type checking (catches string instead of number)
- Range checking (catches out-of-bounds values)
- Coordinate validation (detects swapped lat/lon)

### **6. Visualization**
- Interactive HTML maps (Folium)
- Color-coded signal quality (red = poor, green = excellent)
- Transmitter/receiver/beacon markers
- Direct vs optimal path comparison

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                             │
│  (start, dest, SF, TX power, frequency, preferences)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              INPUT VALIDATION                               │
│  - Coordinates (-90 to 90, -180 to 180)                     │
│  - LoRa params (SF: 7-12, TX: 2-30 dBm, Freq: 100-1000 MHz) │
│  - Grid config (spacing, corridor width, adaptive)          │
│  - Optimization (PDR threshold, path deviation)             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│          DISTANCE-BASED ROUTING DECISION                    │
│                                                             │
│  Distance < 1 km? ──YES──> DIRECT PATH (no beacons)         │
│       │                                                     │
│       NO                                                    │
│       │                                                     │
│       └──────────────────> A* OPTIMIZATION (with beacons)   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│             ADAPTIVE GRID GENERATION                        │
│  • Calculates segments based on user spacing (1-2 km)       │
│  • Creates search corridor (±2-6 km width)                  │
│  • Generates candidate beacon positions                     │
│  • Adapts density based on total distance                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│        BATCH SPATIAL DATA FETCH (Google Earth Engine)       │
│  • Parallel workers (5-10 threads)                          │
│  • Fetches elevation (SRTM 30m)                             │
│  • Fetches land cover (ESA WorldCover 10m)                  │
│  • Calculates terrain penalties                             │
│  • Caches results to disk                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           15-FEATURE VECTOR CONSTRUCTION                    │
│  1. Elevation          9.  Path built-up fraction           │
│  2. Land cover         10. Path vegetation fraction         │
│  3. Terrain penalty    11. Path water fraction              │
│  4. Distance           12. Path avg penalty                 │
│  5. Spreading factor   13. Path elevation std               │
│  6. Frequency          14. Max terrain obstruction          │
│  7. TX power           15. Path dominant land cover         │
│  8. Elevation norm                                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              ML MODEL PREDICTION                            │
│  Input: 15 features → Model → Output: [RSSI, SNR, Path Loss]│
│  • Uses BEST model (RF/XGBoost/NN/Ensemble)                 │
│  • Scaled features (StandardScaler)                         │
│  • Validated output (clipped ranges)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│            PHYSICS-BASED PDR CALCULATION                    │
│  Formula: PDR = 1 - exp(-k × margin)                        │
│  • margin = SNR - SNR_threshold[SF]                         │
│  • k = decay constant based on land cover                   │
│  • NOT predicted by ML (uses LoRaWAN spec)                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│             A* PATHFINDING ALGORITHM                        │
│  • Cost function: PDR + distance + terrain                  │
│  • Prefers: high PDR, short paths, water bodies             │
│  • Avoids: low PDR, buildings, long detours                 │
│  • Memory-efficient: uses indices not objects               │
│  • Supports zigzag/curved paths                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                   │
│  • Beacon positions (lat/lon with metrics)                  │
│  • Quality metrics (avg/min PDR, RSSI, SNR)                 │
│  • Interactive HTML map (Folium)                            │
│  • JSON results file                                        │
│  • Comparison: direct vs optimal path                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### **Prerequisites**

- Python 3.8 or higher (recommended 3.10)
- CUDA-compatible GPU (optional, for faster training)
- Google Earth Engine account (free)

### **Step 1: Clone Repository**

```bash
git clone https://github.com/yourusername/lora-optimization.git
cd lora-optimization
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3: Configure Google Earth Engine**

1. **Create account**: [Google Earth Engine](https://earthengine.google.com/signup/)
2. **Authenticate**:
   ```bash
   earthengine authenticate
   ```
3. **Create `.env` file** (required):
   ```bash
   GEE_PROJECT_ID=your-project-id
   ```

### **Step 4: Verify Installation**

```python
python -c "import torch; import ee; print('Installation successful')"
```

---

## ⚙️ Configuration Guide

### **CONFIG Dictionary Structure**

```python
CONFIG = {
    # Data loading configuration
    'data': {
        'dataset1_path': r'../data/processed_data_1.csv',
        'dataset2_path': r'../data/processed_data_2.csv',
        'test_size': 0.2,
        'random_state': 42
    },
    
    # Hyperparameter tuning configuration
    'hyperparameter_tuning': {
        'enable': True,              # Set to True to enable tuning
        'nn_trials': 100,             # Number of trials for neural network
        'rf_n_iter': 500,             # Number of iterations for Random Forest
        'xgb_n_iter': 500,            # Number of iterations for XGBoost
        'cv_folds': 5,               # Number of cross-validation folds
        'tuning_data_ratio': 0.25     # Portion of training data to use for tuning
    },
    
    # Model hyperparameters (used only if hyperparameter tuning is disabled)
    'model_hyperparams': {
        'neural_network': {
            'hidden_sizes': [256, 128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'epochs': 400,
            'early_stopping_patience': 20,
            'gradient_clip': 1.0
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None
        },
        'xgboost': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'min_child_weight': 1
        }
    },
        
    
    # Google Earth Engine configuration
    'gee': {
        'batch_size': 50,               # Points per batch request
        'workers': 5,                   # Parallel threads (1-10)
        'retry_attempts': 3,            # Retry failed requests
        'fallback_to_individual': True, # Fallback if batch fails
        'cache_enabled': True,          # Cache GEE results to disk
        'cache_file': 'gee_cache.pkl',  # Cache filename
        'path_spatial_samples': 15      # Samples for path features
    },
    
    # Path optimization configuration
    'optimization': {
        'grid_spacing_km': 1.5,          # Distance between grid points (1-2 km)
        'corridor_width_km': 4.0,        # Search corridor width
        'adaptive_grid': True,           # Auto-adjust grid density
        'max_path_deviation': 0.5,       # Allow 50% longer than direct
        'min_pdr_threshold': 0.3,        # Minimum acceptable PDR (30%)
        'prefer_water': True,            # Prefer water bodies (best RF)
        'avoid_buildings': True,         # Avoid built-up areas
        'direct_path_threshold_km': 1.0  # Max direct path distance
    }
}

system = ImprovedLoRaSystem(config_dict=CONFIG)
```

---

## 📖 Parameter Reference

### **Coordinates (REQUIRED)**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `start_lat` | float | -90 to 90 | Transmitter latitude |
| `start_lon` | float | -180 to 180 | Transmitter longitude |
| `dest_lat` | float | -90 to 90 | Receiver latitude |
| `dest_lon` | float | -180 to 180 | Receiver longitude |

**Common Errors:**
- ❌ Swapped lat/lon → System detects and warns
- ❌ Out of range → `InvalidCoordinatesError`
- ❌ Identical start/dest → Error with clear message

---

### **LoRa Parameters (REQUIRED)**

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `spreading_factor` | int | 7-12 | 7 | **SF7** = short range/fast, **SF12** = long range/slow |
| `tx_power` | int | 2-30 | 14 | TX power in dBm (14 = standard, 20 = high) |
| `frequency` | int | 100-1000 | 868 | **868** = EU, **915** = US, **923** = Asia |

**Spreading Factor Guide:**
```
SF7:  Range ~2 km,  Data Rate 5470 bps  ← Urban, short distance
SF8:  Range ~4 km,  Data Rate 3125 bps
SF9:  Range ~6 km,  Data Rate 1760 bps
SF10: Range ~8 km,  Data Rate 980 bps   ← Balanced
SF11: Range ~11 km, Data Rate 440 bps
SF12: Range ~15 km, Data Rate 250 bps   ← Rural, maximum range
```

---

### **Grid Configuration (OPTIONAL)**

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `grid_spacing_km` | float | 0.2-10.0 | 1.5 | Distance between grid segments |
| `corridor_width_km` | float | 0.5-20.0 | 4.0 | Search area width  |
| `adaptive_grid` | bool | True/False | True | Auto-adjust density based on distance |

**Grid Spacing Impact:**
- **Small (0.5 km)**: Dense grid, more beacons, slower computation
- **Medium (1.5 km)**: Balanced, recommended
- **Large (3.0 km)**: Sparse grid, fewer beacons, may miss optimal paths

---

### **GEE Configuration (OPTIONAL)**

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `gee_workers` | int | 1-20 | 5 | Parallel workers for faster fetching |

**Workers Impact:**
- **1 worker**: Slowest, ~90 seconds for 225 points
- **5 workers**: Balanced, ~15 seconds (recommended)
- **10 workers**: Fastest, ~8 seconds, may hit API limits
- **20 workers**: Not recommended, API rate limits

---

### **Optimization Preferences (OPTIONAL)**

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_path_deviation` | float | 0.0-3.0 | 0.5 | Max path length vs direct (0.5 = 50% longer) |
| `min_pdr_threshold` | float | 0.0-1.0 | 0.3 | Minimum PDR to consider (30%) |
| `prefer_water` | bool | True/False | True | Prefer water bodies (best RF propagation) |
| `avoid_buildings` | bool | True/False | True | Avoid built-up areas (worst RF propagation) |
| `direct_path_threshold_km` | float | 0.1-10.0 | 1.0 | Distance below which to use direct path |

**max_path_deviation Examples:**
```python
max_path_deviation=0.2  # Path can be 20% longer (nearly straight)
max_path_deviation=0.5  # Path can be 50% longer (default)
max_path_deviation=1.0  # Path can be 100% longer (double, allows zigzag)
```

**min_pdr_threshold Examples:**
```python
min_pdr_threshold=0.2  # Accept 20% PDR (lenient, rural)
min_pdr_threshold=0.3  # Accept 30% PDR (default, balanced)
min_pdr_threshold=0.5  # Accept 50% PDR (strict, urban/critical)
```

---

## 💡 Usage Examples

```python
result = system.predict_and_optimize(
        # Coordinates (REQUIRED)
        # lat :-90 to 90, lon :-180 to 180
        start_lat=28.992, start_lon=50.851,
        dest_lat=28.992,dest_lon=50.951,
        
        # LoRa Parameters (REQUIRED)
        # SF 7-12 (higher = longer range, slower)
        spreading_factor=7,       
        # 2-30 dBm (higher = better signal, more power)
        tx_power=14,              
        # 100-1000 MHz (EU: 868, US: 915, AS: 923)
        frequency=868,            
        
        # Grid Configuration (OPTIONAL)
        # 0.1-10.0 km (larger = less beacon, less efficient)
        grid_spacing_km=1.5,       # 1.0-2.0 km recommended
        # 0.5-20.0 km (larger = wider search area, more points, more compute)
        corridor_width_km=4.0,     # 3.0-6.0 km recommended
        # True/False (adjust grid density based on distance)
        adaptive_grid=True,        # Auto-adjust based on distance
        
        # GEE Configuration (OPTIONAL)
        # 1-20 (higher = faster, more points, more compute)
        gee_workers=8,             # 5-10 for best speed/stability
        
        # Optimization Preferences (OPTIONAL)
        # 0.0-3.0 (larger = longer path, more compute)
        max_path_deviation=0.5,    # 0.3-1.0 recommended
        # 0.1-1.0 (larger = more strict, less points, less compute)
        min_pdr_threshold=0.3,     # 0.2-0.5 recommended
        # True/False (water = best RF, buildings = worst RF)
        prefer_water=True,         # Water = best RF propagation
        avoid_buildings=True,       # Buildings = worst RF propagation
        # 0.1-10 km (higher = less accuracy)
        direct_path_threshold_km= 1.0  # 0.5-2.0 recommended
    )

```

---

## 🎓 Model Training

### **Dataset Requirements**

Your training data should have **15 features**:

```csv
latitude,longitude,elevation,land_cover,terrain_penalty,distance_to_start,
spreading_factor,frequency,tx_power,elevation_normalized,
path_built_up_fraction,path_vegetation_fraction,path_water_fraction,
path_avg_penalty,path_elevation_std,max_terrain_obstruction_m,
path_dominant_land_cover,RSSI,SNR,observed_path_loss
```

**Target Variables (3):**
- `RSSI`: Received Signal Strength Indicator (dBm)
- `SNR`: Signal-to-Noise Ratio (dB)
- `observed_path_loss`: Path loss (dB)

**Note**: PDR is NOT in training data (calculated from SNR using physics).

---

### **Training Process**

```python
# 1. Load data
X_train, X_test, y_train, y_test, features = system.load_and_preprocess_data()

# 2. Train all models
best_model, model_name = system.train_models_and_select_best(
    X_train, X_test, y_train, y_test, features
)

# Models trained:
# - Random Forest (100 trees)
# - XGBoost (100 estimators)
# - Neural Network (256-128-64-32 architecture)
# - Ensemble (weighted combination)

# 3. Best model auto-selected based on R²
print(f"Best model: {model_name}")  # e.g., "Ensemble"

# 4. Models saved to ./models/
# - best_model.pkl
# - scaler.pkl
# - model_metadata.json
```

---

### **Training Performance**

**Expected Results (with good data):**

| Model | RSSI R² | SNR R² | Path Loss R² | Avg R² |
|-------|---------|--------|--------------|--------|
| Random Forest | 0.85 | 0.82 | 0.88 | 0.85 |
| XGBoost | 0.87 | 0.84 | 0.90 | 0.87 |
| Neural Network | 0.89 | 0.86 | 0.91 | 0.89 |
| **Ensemble** | **0.91** | **0.88** | **0.93** | **0.91** |

**Training Time:**
- Random Forest: ~2 minutes
- XGBoost: ~3 minutes
- Neural Network: ~10 minutes (400 epochs with early stopping)
- **Total: ~15 minutes**

---

## 🔬 Technical Details

### **A\* Cost Function**

```python
def calculate_cost(point, distance, lora_params):
    # PDR cost (exponential penalty for poor signal)
    if point.pdr < 0.3:
        pdr_cost = 1000.0  # Blocked
    elif point.pdr < 0.6:
        pdr_cost = 10.0
    else:
        pdr_cost = 0.1
    
    # Distance cost (prefer shorter hops)
    distance_cost = distance / 2000
    
    # Terrain cost (avoid obstacles)
    terrain_cost = point.terrain_penalty * 0.5
    
    # Total cost
    return pdr_cost + distance_cost * 0.2 + terrain_cost * 0.3
```

### **PDR Physics Formula**

```python
def calculate_pdr(snr, spreading_factor, land_cover):
    # Get SNR threshold from LoRaWAN spec
    snr_threshold = SNR_THRESHOLD[spreading_factor]
    
    # Calculate margin
    margin = snr - snr_threshold
    
    if margin <= 0:
        return 0.0  # No signal
    
    # Decay constant based on land cover
    k = LAND_COVER_K[land_cover]
    
    # Exponential recovery
    pdr = 1 - np.exp(-k * margin)
    
    return pdr
```

### **Feature Scaling**

All features are standardized before ML prediction:
```python
X_scaled = (X - mean) / std
```

This ensures:
- Equal weight to all features
- Faster neural network convergence
- Better gradient descent stability

### **Model Ensemble Weighting**

```python
# Calculate weights based on validation R²
weights = softmax(R² * 5)

# Example:
# NN:  R² = 0.89 → weight = 0.45
# XGB: R² = 0.87 → weight = 0.35
# RF:  R² = 0.85 → weight = 0.20
```
---

## 🔧 Troubleshooting

### **Common Errors**

#### **1. GEE Authentication Error**
```
Error: Google Earth Engine initialization failed
```
**Solution:**
```bash
earthengine authenticate
# Follow prompts to log in
```

---

#### **2. Invalid Coordinates**
```
InvalidCoordinatesError: Start latitude 150.841 out of range [-90, 90]. 
Did you swap latitude and longitude?
```
**Solution:** Check coordinate order (lat, lon not lon, lat)

---

#### **3. No Viable Path Found**
```
NoViablePathError: No viable path found with given parameters.
```
**Solutions:**
- Increase `max_path_deviation` (allow longer paths)
- Decrease `min_pdr_threshold` (accept lower quality)
- Increase `spreading_factor` (longer range)
- Increase `tx_power` (stronger signal)

---

#### **4. GEE Rate Limit**
```
Error: Too many requests
```
**Solutions:**
- Decrease `gee_workers` (fewer parallel requests)
- Enable caching: `cache_enabled=True`
- Wait 1 minute and retry

---

#### **5. Out of Memory (Training)**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `batch_size` (default 64 → try 32)
- Use CPU: Remove CUDA from PyTorch installation
- Reduce model size: `hidden_sizes=[128, 64, 32]`

---

### **Performance Issues**

**Problem: Slow GEE fetching (>60 seconds)**

**Solutions:**
1. Increase workers: `gee_workers=8` (faster)
2. Enable caching: `cache_enabled=True`
3. Reduce grid density: `grid_spacing_km=2.0`
4. Check internet connection

---

**Problem: A\* takes too long (>5 minutes)**

**Solutions:**
1. Increase grid spacing: `grid_spacing_km=2.0`
2. Narrow corridor: `corridor_width_km=3.0`
3. Stricter PDR: `min_pdr_threshold=0.4` (blocks more points)
4. Tighter deviation: `max_path_deviation=0.3`

---

## 📜 Additional Resources

### **Acknowledgments**
- **Google Earth Engine** for providing free satellite data access
- **LoRa Alliance** for LoRaWAN specifications
- **ESA WorldCover** for 10m land cover data (Zanaga et al., 2021)
- **NASA SRTM** for 30m elevation data
- **PyTorch**, **scikit-learn**, **XGBoost** communities
- 
### **LoRaWAN Documentation**
- [LoRa Alliance](https://lora-alliance.org/)
- [Semtech SX1276 Datasheet](https://www.semtech.com/products/wireless-rf/lora-core/sx1276)

### **Google Earth Engine**
- [GEE Documentation](https://developers.google.com/earth-engine)
- [ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200)
- [SRTM Elevation](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)

### **Machine Learning**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### **Dataset Sources**
- [data_1.xlsx](https://zenodo.org/records/10142174)
- [data_2.csv](https://doi.org/10.5281/zenodo.13835721)

---

## 📞 Contact & Support

- **Documentation**: This README
- **Issues**: [GitHub Issues](https://github.com/Nfx1z/Predictive_Optimization_LoRa/issues)
- **Email**: icr0x002@gmail.com

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### **Reporting Issues**

```markdown
## Issue Template

**Description:**
Brief description of the issue

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Environment:**
- Python version: 3.8
- PyTorch version: 2.0.1
- OS: Ubuntu 20.04

**Error Message:**
```

### **Pull Request Process**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with clear commits
4. Add tests if applicable
5. Update documentation
6. Submit pull request with description

---


## ❓ FAQ

### **Q1: Can I use this without Google Earth Engine?**
**A:** No, GEE is required for real terrain data. However, you can:
- Use cached data from previous runs
- Manually provide elevation/land cover arrays
- Contact us for offline dataset options

### **Q2: How accurate are the predictions?**
**A:** With good training data:
- RSSI: ±3 dBm (R² = 0.91)
- SNR: ±2 dB (R² = 0.88)
- PDR: Physics-based (deterministic)

### **Q3: Can it work for indoor deployments?**
**A:** No, system is designed for outdoor long-range. Indoor propagation is fundamentally different (multipath, walls, floors).

### **Q4: What's the maximum distance supported?**
**A:** Theoretically unlimited with multi-hop. Tested up to 50 km

### **Q5: How much does Google Earth Engine cost?**
**A:** Free for research/education (up to 250,000 requests/day). Commercial use may require paid account.

### **Q6: Can I train on my own data?**
**A:** Yes! Provide CSV with 15 features + 3 targets (RSSI, SNR, path_loss). Minimum 1000 samples recommended. Contact us for details.

### **Q7: Does it work in all countries?**
**A:** Yes, GEE has global coverage. Adjust frequency band:
- EU: 868 MHz
- US: 915 MHz
- Asia: 923 MHz
- India: 865 MHz
- or your continent's frequency band

### **Q8: What if A\* finds no path?**
**A:** Try:
1. Increase `max_path_deviation` → allow longer paths
2. Decrease `min_pdr_threshold` → accept lower quality
3. Increase `spreading_factor` → longer range
4. Increase `tx_power` → stronger signal
5. Widen corridor: `corridor_width_km=6.0`

### **Q9: How do I validate results in the field?**
**A:** Recommended validation process:
1. Run optimization
2. Deploy beacons at predicted locations
3. Measure actual RSSI/SNR/PDR
4. Compare with predictions
5. Adjust model if needed

### **Q10: Can I use custom propagation models?**
**A:** Yes, modify `LoRaPhysicsEngine.calculate_pdr()` with your formula. Keep ML predictions separate from physics.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Nfx1z]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
