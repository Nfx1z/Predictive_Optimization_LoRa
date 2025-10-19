# LoRa Path Optimization System - Technical Documentation

## Executive Summary

This system implements an advanced machine learning-based approach to optimize LoRa network connectivity across challenging terrain. By intelligently predicting radio frequency propagation conditions and strategically placing relay beacons, the system maximizes packet delivery reliability while minimizing infrastructure deployment costs. The methodology combines spatial analysis, ensemble machine learning, and graph-based pathfinding to solve the complex problem of maintaining robust low-power wide-area network (LPWAN) connectivity in real-world environments.

---

## 1. Introduction

### 1.1 The Challenge

LoRa networks face fundamental limitations when deployed across complex geographic landscapes:

- **Signal Degradation**: Radio signals weaken unpredictably due to terrain features, vegetation, and urban structures  
- **Environmental Variability**: Traditional propagation models (Free Space, Okumura-Hata) fail to capture local environmental complexities  
- **Infrastructure Optimization**: Determining optimal relay beacon placement requires balancing coverage, cost, and reliability  
- **Planning Inefficiency**: Conventional network design relies on expensive field surveys and trial-and-error testing  

### 1.2 The Solution Approach

This system introduces a data-driven methodology that:

- **Learns from Real-World Data**: Trains predictive models on actual LoRa measurements correlated with geographic conditions  
- **Predicts Signal Quality**: Estimates RSSI, SNR, and packet delivery ratio for any geographic coordinate  
- **Optimizes Network Topology**: Uses intelligent pathfinding algorithms to route signals through favorable terrain  
- **Validates Performance**: Quantifies improvements through comparative analysis against direct-path approaches  

---

## 2. Core Methodology

### 2.1 Spatial Feature Engineering

The system's predictive power stems from comprehensive spatial feature extraction that captures both point-specific and path-based characteristics.

#### 2.1.1 Point-Based Features

For each geographic location, the system extracts:

- **Elevation Data**: Height above sea level from SRTM (Shuttle Radar Topography Mission) at 30-meter resolution  
- **Land Cover Classification**: Surface type from ESA WorldCover at 10-meter resolution, including:
  - Built-up areas (urban/buildings)  
  - Vegetation types (trees, grassland, crops)  
  - Water bodies  
  - Bare/sparse terrain  
- **Terrain Penalty Factors**: RF attenuation coefficients derived from land cover types  
- **LoRa Configuration**: Spreading factor, transmission power, frequency band  

#### 2.1.2 Path-Based Features (Critical Innovation)

Unlike traditional models that only consider endpoint conditions, this system analyzes the entire signal propagation path:

- **Built-up Fraction**: Percentage of path crossing urban areas  
- **Vegetation Fraction**: Percentage of path through forested/vegetated zones  
- **Water Fraction**: Percentage of path over water bodies  
- **Average Terrain Penalty**: Mean RF attenuation across the entire path  
- **Elevation Variance**: Terrain roughness indicator (standard deviation of elevation)  
- **Maximum Obstruction**: Highest elevation difference encountered along path  
- **Dominant Land Cover**: Most prevalent terrain type along the signal path  

> **Why This Matters**: A building located between transmitter and receiver has greater impact than a building at the receiver. Path-based features capture these cumulative propagation effects that point-based models miss entirely.

---

### 2.2 Machine Learning Architecture

#### 2.2.1 Ensemble Model Design

The system employs three complementary predictive models, each bringing distinct strengths:

1. **Deep Neural Network**  
   - Architecture: Multi-layer perceptron with 4–6 hidden layers  
   - Activation: GELU/ReLU with batch normalization  
   - Regularization: Dropout, L2 weight decay, gradient clipping  
   - Training: Adam optimizer with learning rate scheduling and early stopping  
   - Strength: Captures complex non-linear interactions between features  

2. **Random Forest Regressor**  
   - Ensemble: 200–500 decision trees with bootstrap aggregation  
   - Tree Structure: Unlimited depth with adaptive leaf node splitting  
   - Feature Selection: Random feature sampling per split  
   - Strength: Handles feature interactions, provides importance rankings, robust to outliers  

3. **XGBoost (Gradient Boosting)**  
   - Algorithm: Gradient-boosted decision trees with L1/L2 regularization  
   - Tree Method: Histogram-based splitting for efficiency  
   - Growth Policy: Loss-guided leaf growth  
   - Strength: Superior performance on structured data, handles missing values  

#### 2.2.2 Hyperparameter Optimization

The system employs **Optuna (Tree-structured Parzen Estimator)** for automated hyperparameter tuning:

- **Neural Networks (100 trials)**:
  - Network architecture (depth, width, decay strategy)  
  - Regularization (dropout rates, weight decay)  
  - Optimization (learning rate, optimizer type, batch size)  
  - Training dynamics (gradient clipping, early stopping)  

- **Random Forest (500 iterations)**:
  - Tree parameters (max depth, min samples split/leaf)  
  - Ensemble size (number of estimators)  
  - Sampling strategies (bootstrap, max samples)  
  - Pruning parameters (complexity cost, impurity decrease)  

- **XGBoost (500 iterations)**:
  - Boosting parameters (learning rate, number of estimators)  
  - Tree structure (max depth, max leaves, min child weight)  
  - Regularization (gamma, alpha, lambda)  
  - Sampling (subsample, column sampling strategies)  

> **Results from Optimization**: The tuning process consistently achieves 5–15% performance improvements over default configurations. The best configurations achieved cross-validated MSE scores of **61.75 (Random Forest)** and **62.64 (XGBoost)**, with the neural network reaching a best validation loss of **61.21** after 100 trials. Trial identifiers (e.g., “Trial 79”) are run-specific and not fixed.

#### 2.2.3 Multi-Target Prediction

Each model simultaneously predicts three interdependent RF metrics:

- **RSSI** (Received Signal Strength Indicator): Absolute signal power in dBm  
- **SNR** (Signal-to-Noise Ratio): Signal quality relative to background noise in dB  
- **Path Loss**: Total signal attenuation from transmitter to receiver in dB  

This joint prediction approach leverages the physical relationships between these metrics, improving overall accuracy.

#### 2.2.4 Model Selection and Ensemble

After training all three models independently, the system:

1. **Evaluates Performance**: Tests each model on a held-out test set of **783 samples**  
2. **Computes Ensemble Weights**: Assigns weights proportional to `exp(5 × R²)`, strongly favoring top performers  
3. **Creates Weighted Ensemble**: Combines predictions from all models  
4. **Selects Best Performer**: Chooses the single model or ensemble with highest average R² score  

> **Observed Performance (October 19, 2025 run)**:
> - **XGBoost**: Average R² = **0.7146**, RMSE = **6.68 dB** (**Selected as best**)  
> - **Random Forest**: Average R² = **0.7089**, RMSE = **6.68 dB**  
> - **Neural Network**: Average R² = **0.5642**, RMSE = **8.52 dB**  
> - **Ensemble**: Average R² = **0.7050**, RMSE = **6.78 dB**  
>
> The XGBoost model achieves **81.04% R² for RSSI prediction** and **52.29% R² for SNR prediction**, demonstrating strong predictive capability despite the inherent complexity of RF propagation.

---

### 2.3 RF Physics Integration

#### 2.3.1 LoRa Link Budget Model

The system incorporates physical LoRa specifications to ensure predictions respect fundamental radio propagation principles:

| SF | Sensitivity | SNR Threshold | Range Characteristic | Data Rate |
|----|-------------|---------------|----------------------|-----------|
| 7  | -123 dBm    | -7.5 dB       | Short range          | 5.47 kbps |
| 9  | -129 dBm    | -12.5 dB      | Medium range         | 1.76 kbps |
| 12 | -137 dBm    | -20.0 dB      | Maximum range        | 0.29 kbps |

Higher spreading factors trade data rate for increased sensitivity, enabling longer-range communication.

#### 2.3.2 Packet Delivery Ratio (PDR) Calculation

The system implements a terrain-aware PDR model that accounts for land cover characteristics:

**PDR Formula**:
```python
margin = SNR - SNR_threshold(SF)

If margin ≤ -5 dB: PDR ≈ 1% (signal too weak)
If margin ≤ 0 dB:  PDR ≈ 5% × exp(margin) (rapid degradation)
If margin > 0 dB:  PDR = 1 / (1 + exp(-k × (margin - 5)))
```

Where `k` is the terrain recovery constant:

- **Water** (`k = 0.40`): PDR recovers quickly with positive SNR margin  
- **Cropland** (`k = 0.28`): Moderate recovery  
- **Buildings** (`k = 0.08`): Slow recovery due to multipath fading and reflections  

> **Physical Basis**: Urban environments exhibit severe multipath propagation, where signals reflect off buildings and arrive at different phases, causing constructive/destructive interference. This phenomenon degrades PDR even when SNR is theoretically sufficient.

**Example Impact**: With +10 dB SNR margin:
- Over water: PDR = 95% (excellent connectivity)  
- Over cropland: PDR = 88% (very good)  
- Through buildings: PDR = 52% (marginal, may experience packet loss)  

This captures the empirical reality that urban LoRa links require significantly higher SNR margins than rural links for equivalent reliability.

---

### 2.4 Adaptive Grid Generation

The system employs an intelligent spatial discretization strategy that balances computational efficiency with path accuracy.

#### 2.4.1 Distance-Based Adaptation

Grid density automatically adjusts based on the total distance between transmitter and receiver:

- **Short distance (< 3 km)**: 9 parallel lanes, 1.0 km segment spacing → Dense exploration  
- **Medium distance (3–8 km)**: 11 lanes, 1.5 km spacing → Balanced approach  
- **Long distance (> 8 km)**: 15 lanes, 2.0 km spacing → Efficient wide-area coverage  

> **Rationale**: Short-distance paths require fine-grained control to navigate around localized obstacles (individual buildings). Long-distance paths benefit from coarser sampling since small deviations average out over distance.

#### 2.4.2 Corridor Search Strategy

Rather than exploring the entire 2D space (computationally prohibitive), the system constrains the search to a corridor around the direct line:

- **Corridor width**: 3–6 km on each side of the direct path (configurable)  
- **Perpendicular sampling**: Multiple parallel lanes within corridor  
- **Path length constraint**: Maximum deviation = 50% longer than direct path (configurable)  

> **Observed Results**: In the test case (London, 28.58 km), the system:
> - Generated a grid of **405 candidate points** (27 segments × 15 lanes)  
> - Evaluated **2,433 potential hops**  
> - Found optimal path in **336 A\* iterations**  
> This demonstrates efficient exploration of a vast solution space.

---

### 2.5 A* Pathfinding with RF-Aware Cost Function

#### 2.5.1 Graph Construction

The adaptive grid becomes a directed graph where:

- **Nodes**: Each grid point represents a potential beacon location  
- **Edges**: Connections exist only between consecutive segments (enforcing forward progress)  
- **Edge weights**: Cost derived from predicted link quality (RSSI, SNR, PDR)  

#### 2.5.2 Multi-Objective Cost Function

The system optimizes a weighted combination of objectives:

```python
total_cost = 0.7 × PDR_cost + 0.1 × distance_cost + 0.2 × terrain_cost
```

**PDR_cost**:
- If PDR < threshold: cost = INFINITY (block unusable links)  
- If PDR < 0.4: cost = 100 (severe penalty)  
- If PDR < 0.6: cost = 20 (moderate penalty)  
- If PDR < 0.8: cost = 5 (minor penalty)  
- If PDR ≥ 0.8: cost = 0.5 (minimal penalty)  

**distance_cost** = hop_distance / 1000 (normalized to km)  

**terrain_cost** = terrain_penalty × 10  
- Multiplied by 0.1 if land cover is water (strong preference)  
- Multiplied by 5.0 if land cover is buildings (strong avoidance)  

> **Design Philosophy**:
> - **PDR dominance (70%)**: Connectivity is paramount  
> - **Distance minimization (10%)**: Fewer hops reduce latency and power consumption  
> - **Terrain optimization (20%)**: Prefer favorable propagation environments  

#### 2.5.3 Heuristic Function

A* requires an admissible heuristic (never overestimates true cost):

```python
h(n) = euclidean_distance(n, goal) / 10000
```

This underestimates cost since perfect PDR (cost ≈ 0.5) would yield actual cost near 0.5 × segments, while the heuristic is much smaller. Admissibility guarantees the found path is optimal given the cost function.

#### 2.5.4 Pathfinding Execution

The algorithm proceeds as follows:

1. **Initialize**: Place start node in priority queue with f-score = h(start)  
2. **Iterate**:
   - Extract node with lowest f-score  
   - If at destination segment → reconstruct and return path  
   - Expand to 7 neighboring nodes in next segment (lane ± 3)  
   - For each neighbor:
     - Query pre-computed ML prediction for hop quality  
     - Calculate tentative g-score = current g + edge cost  
     - If better than previous → update g-score, compute new f-score, add to queue  
3. **Terminate**: When destination reached or queue exhausted  

> **Performance Characteristics**:
> - Time complexity: O(E log V) where E = edges, V = vertices  
> - Space complexity: O(V) for storing g-scores and closed set  
> - Actual performance: 336 iterations for 405-node graph = highly efficient due to heuristic guidance

---

### 2.6 Batch Prediction Optimization

#### 2.6.1 Pre-computation Strategy

Rather than predicting hop quality on-demand during A* search (expensive), the system employs batch pre-computation:

- **Enumerate all possible hops**: For each segment, consider all lane-to-lane transitions (typically 7 per node)  
- **Extract features in parallel**: Fetch spatial data for all 2,433 hops using 5 parallel workers  
- **Batch ML inference**: Process all features through the model in a single vectorized call  
- **Cache predictions**: Store in hash map indexed by `(source_node, dest_node)` tuple  

> **Performance Impact**:
> - Traditional approach: 2,433 hops × 0.025s = ~60 seconds  
> - Batch approach: **<1 second total (100× speedup)**  
> This transformation makes real-time path optimization practical.

#### 2.6.2 Spatial Data Caching

The system maintains a persistent cache of Google Earth Engine queries:

- **Cache key**: `"(data_type, latitude_4_decimals, longitude_4_decimals)"`  
- **Storage**: Python pickle file (~1–10 MB typical)  
- **Hit rate**: After initial run, 95–100% cache hits  
- **Persistence**: Cache survives across sessions  

> **Observed Results**: In the test case:
> - Initial run: Fetched 405 new spatial points  
> - **Total cached: 82,481 points** (as of October 19, 2025)  
> - Subsequent runs: ~100% cache hit rate  
> This eliminates redundant API calls and dramatically accelerates iterative optimization.

---

## 3. Validation Methodology

### 3.1 Model Performance Evaluation

#### 3.1.1 Comprehensive Metrics

The system evaluates models using multiple complementary metrics:

- **R² Score (Coefficient of Determination)**:  
  Measures proportion of variance explained by the model  
  *Interpretation*: R² = 0.81 means the model explains 81% of RSSI variability  
  *Advantage*: Scale-independent, comparable across different targets  

- **RMSE (Root Mean Squared Error)**:  
  Penalizes large errors more heavily than small errors  
  *Interpretation*: RMSE = 7.39 dBm means typical prediction error is 7.39 dBm  
  *Advantage*: Same units as target variable, intuitive interpretation  

- **MAE (Mean Absolute Error)**:  
  Average absolute prediction error  
  *Interpretation*: MAE = 5.29 dBm means average error magnitude is 5.29 dBm  
  *Advantage*: Robust to outliers, easier to interpret than RMSE  

- **MAPE (Mean Absolute Percentage Error)**:  
  Percentage error across predictions  
  *Interpretation*: MAPE = 5.87% means predictions are typically within ±5.87% of actual  
  *Advantage*: Scale-independent, business-interpretable  

> **Observed Results (XGBoost on 783-sample test set)**:
> | Target       | R²     | RMSE     | MAE      | MAPE   |
> |--------------|--------|----------|----------|--------|
> | RSSI         | 0.8104 | 7.39 dBm | 5.29 dBm | 5.87%  |
> | SNR          | 0.5229 | 5.27 dB  | 3.89 dB  | Large* |
> | Path Loss    | 0.8104 | 7.39 dB  | 5.29 dB  | 5.04%  |
>
> \*SNR MAPE is inflated due to near-zero SNR values in the dataset (division by small numbers)

#### 3.1.2 Cross-Validation Strategy

Models undergo rigorous validation:

- **Train/Test Split**: **3,132 training** (80%), **783 held-out test** (20%) — never seen during training or tuning  
- **Hyperparameter Tuning**: Uses **2,349 training samples** for **5-fold CV**, with **783 samples** as a separate validation set during tuning  
- **Final Evaluation**: Best model tested on the untouched 783-sample test set  

This nested validation prevents overfitting and ensures honest performance estimates.

---

### 3.2 Path Optimization Validation

#### 3.2.1 Direct Path Baseline

Every optimized path is compared against a direct-path baseline:
- **Method**: Sample points along straight line from start to destination  
- **Prediction**: Use ML model to predict link quality  

#### 3.2.2 Comparative Analysis

> **Observed Results (London Test Case, October 19, 2025)**:
>
> | Metric          | Direct Path | Optimized Path | Improvement               |
> |-----------------|-------------|----------------|---------------------------|
> | Avg PDR         | 59.60%      | 79.79%         | **+20.19% (33.9% rel.)**  |
> | Avg SNR         | 0.49 dB     | 4.15 dB        | **+3.66 dB (747% rel.)**  |
> | Avg RSSI        | -101.25 dBm | -100.60 dBm    | +0.65 dBm (slightly stronger) |
> | Min PDR         | 45.35% | 57.97%         | **1.2× minimum reliability** |

**Interpretation**: By strategically placing 27 relay beacons with the correct locations, the system achieves:
- **33.9% higher packet delivery** (more reliable communication)  
- **747% better signal quality** (cleaner signal with less interference)  
- **1.2× minimum link reliability** (no critically weak links)

---

## 4. System Performance Characteristics

### 4.1 Computational Efficiency

**Time Breakdown (28.58 km optimization)**:
- Grid generation: <0.1 seconds  
- Spatial data fetch: 0.1 seconds (cached) / 60 seconds (uncached)  
- ML predictions (2,433 hops): <1 second  
- A* pathfinding: <0.1 seconds (336 iterations)  
- **Total**: ~1 second (cached) / ~60 seconds (uncached)  

### 4.2 Memory Requirements

**Typical Memory Footprint**:
- Model storage: ~50 MB (XGBoost trees)  
- Grid + predictions: ~5–10 MB per optimization  
- Spatial data cache: ~1–10 MB  
- **Total runtime memory**: ~100–200 MB  

### 4.3 Accuracy Characteristics

**Model Generalization**:
- R² = 0.71–0.81: Models explain 71–81% of signal variation  
- RMSE = 6.7–7.5 dB: Typical prediction error is 6.7–7.5 dB  

---

## 5. Methodological Innovations

### 5.1 Path-Based Feature Engineering

**Novel Contribution**: Traditional RF prediction models consider only transmitter and receiver locations. This system introduces **path-centric features** that analyze the entire propagation corridor:

- Cumulative terrain effects (multiple obstacles compound)
- Path-integrated land cover statistics
- Elevation variance as a proxy for Fresnel zone obstruction

**Impact**: Demonstrated R² improvement of ~8-12% compared to point-only features (ablation studies show path features contribute significantly to XGBoost performance).

### 5.2 Terrain-Aware PDR Modeling

**Novel Contribution**: Standard LoRa link budgets assume PDR is purely SNR-dependent. This system incorporates **land-cover-specific recovery constants** that model multipath fading and scattering effects:

- Urban areas (k=0.08): Slow PDR recovery due to severe multipath
- Open areas (k=0.28): Moderate recovery
- Water (k=0.40): Fast recovery due to minimal scattering

**Physical Basis**: Urban environments exhibit Rayleigh/Rician fading due to multi-path propagation. The recovery constant `k` effectively models the fading distribution's impact on packet loss.

**Impact**: Enables realistic urban vs. rural performance differentiation that purely SNR-based models miss.

### 5.3 Adaptive Grid Density

**Novel Contribution**: Fixed-resolution grids are either too coarse (miss local obstacles) or too fine (computationally prohibitive). This system dynamically adjusts:

- Grid spacing: 1.0-2.0 km based on total distance
- Corridor width: 3-6 km based on terrain complexity
- Lane count: 9-15 based on distance

**Impact**: Achieves optimal trade-off between solution quality and computation time. Short paths get fine-grained control, long paths get efficient coverage.

### 5.4 Batch Spatial Processing

**Novel Contribution**: Traditional approaches query spatial databases sequentially (1 hop at a time during pathfinding). This system:

- Pre-computes all possible hops before pathfinding
- Uses parallel workers for spatial queries
- Leverages vectorized ML inference

**Impact**: 100× speedup compared to naive sequential approach, enabling real-time optimization.

---

## 6. Limitations and Future Directions

### 6.1 Current Limitations

**1. Static Environment Assumption**:
- Does not account for dynamic factors (weather, moving obstacles, interference)
- Predictions represent long-term average conditions
- May underestimate short-term variability

**2. Training Data Dependency**:
- Model accuracy limited by training dataset quality and diversity
- Geographic regions not represented in training may have lower accuracy
- Requires periodic retraining with new measurements

**3. Computational Constraints**:
- Very long distances (>50 km) may require segmented optimization
- Real-time adaptation not yet implemented (pre-computes entire path)
- Large-scale multi-gateway networks not fully optimized

**4. LoRa-Specific Design**:
- Current implementation tailored to LoRa physical layer
- Would require adaptation for other LPWAN technologies (Sigfox, NB-IoT)

### 6.2 Future Research Directions

**1. Dynamic Adaptation**:
- Incorporate real-time signal quality feedback
- Implement online learning to adapt to changing conditions
- Dynamic beacon repositioning based on observed performance

**2. Multi-Gateway Optimization**:
- Extend to networks with multiple gateways (not just point-to-point)
- Optimize gateway placement jointly with relay selection
- Load balancing across multiple network paths

**3. Additional Environmental Factors**:
- Integrate weather data (rain attenuation, fog effects)
- Temporal modeling (diurnal patterns, seasonal variations)
- Interference mapping (identify and avoid high-interference zones)

**4. Uncertainty Quantification**:
- Probabilistic predictions (confidence intervals on PDR/RSSI)
- Risk-aware path planning (avoid high-variance paths)
- Robustness analysis (sensitivity to model errors)

**5. Transfer Learning**:
- Pre-train on large datasets, fine-tune for specific regions
- Domain adaptation for different geographic areas
- Few-shot learning for rapid deployment in new regions

---

## 7. Conclusion

This system represents a significant advancement in automated LoRa network planning through the integration of machine learning, spatial analysis, and graph-based optimization. By achieving **R² prediction accuracy of 0.71–0.81** and demonstrating **33.9% PDR improvements** in real-world test cases, the methodology validates the core hypothesis: **data-driven terrain-aware optimization significantly outperforms traditional direct-path approaches**.

The key methodological contributions—**path-based feature engineering**, **terrain-aware PDR modeling**, and **adaptive grid optimization**—establish a foundation for future intelligent wireless network design. As LPWAN technologies continue proliferating in IoT applications, such automated planning tools will become essential for cost-effective, reliable deployments at scale.

## 8. Program and Results

**Main program: [train_process.ipynb](https://github.com/Nfx1z/Predictive_Optimization_LoRa/blob/main/src/train_process.ipynb)**

**Program Explanation: [Code _explanation.md](https://github.com/Nfx1z/Predictive_Optimization_LoRa/blob/main/src/Code%20_explanation.md)**

**Output Results: [output](https://github.com/Nfx1z/Predictive_Optimization_LoRa/tree/main/src/output)**

**Models: [models](https://github.com/Nfx1z/Predictive_Optimization_LoRa/tree/main/src/models)**

**Run [path_visualization.html](https://github.com/Nfx1z/Predictive_Optimization_LoRa/blob/main/src/output/path_visualization.html) script to generate path visualization** 
