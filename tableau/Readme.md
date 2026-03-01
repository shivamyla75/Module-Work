# Tableau Dashboard Setup Guide
## HIGGS Boson Big Data ML Project

---

## Prerequisites
- Tableau Desktop 2024.1+ (or Tableau Public for free use)
- Python pipeline must be run first to generate CSV data files

---

## Data Sources Required

Run the full pipeline first:
```bash
python scripts/run_pipeline.py
```

This generates all required CSV files in `data/samples/`:

| File | Used in | Description |
|------|---------|-------------|
| `higgs_sample_50k.csv` | Dashboard 1 | 50K row sample for EDA |
| `test_evaluation_results.csv` | Dashboard 2 | Model metrics table |

---

## Dashboard 1: Data Quality & Pipeline Monitoring

**Purpose:** Show dataset characteristics and validate ingestion quality.

**Steps to build:**
1. Connect to `higgs_sample_50k.csv`
2. Create Tableau Extract (.hyper) for performance — go to **Data → Extract Data**
3. **View 1 — Class Distribution:**
   - Drag `label` to Columns, `Record Count` to Rows
   - Chart type: Bar chart
   - Colour by label (Signal=orange, Background=blue)
   - Add LOD: `{ FIXED [label] : COUNT([label]) }`
4. **View 2 — Feature Distribution Histograms:**
   - Create a parameter `Feature Selector` (list of all 28 feature names)
   - Use calculated field: `[Feature Selector]` → histogram
   - Split by label using colour
5. **View 3 — Missing Value Heatmap:**
   - Calculated field: `IF ISNULL([lepton_pT]) THEN 1 ELSE 0 END`
   - Repeat per feature column
   - Heatmap with feature names on rows
6. **View 4 — KPI Cards:**
   - Total rows: `11,000,000`
   - Features: `28 + 5 engineered = 33`
   - Ingestion time: add as text

**Design:** Blue/orange colour scheme, dark background for physics theme.

---

## Dashboard 2: Model Performance & Feature Importance

**Purpose:** Compare models and explain which features matter most.

**Steps to build:**
1. Connect to `test_evaluation_results.csv` and `feature_importances.csv`
2. **View 1 — AUC Comparison Bar Chart:**
   - Rows: Algorithm, Columns: AUC-ROC
   - Add reference line at AUC=0.5 (random baseline)
   - Colour by framework (MLlib vs sklearn)
   - Add parameter: `Metric Selector` (AUC-ROC, F1, Accuracy, Precision, Recall)
3. **View 2 — Feature Importance Treemap:**
   - Connect `feature_importances.csv`
   - Size = Importance, Label = feature name
   - Group low-level vs high-level features by colour
4. **View 3 — Confusion Matrix Heatmap:**
   - Build from TP/FP/TN/FN values
   - Normalised — show as % not counts
5. **View 4 — Radar/Spider chart of all 5 metrics per model**
   - Use Tableau's shape chart as radar approximation

**Actions:** Click a model bar → filter confusion matrix and feature importance to that model.

---

## Dashboard 3: Business Insights & Recommendations

**Purpose:** Translate ML results into actionable physics experiment decisions.

**Steps to build:**
1. Connect to `business_cost_results.csv`
2. **View 1 — Cost-Threshold Curve:**
   - X: threshold (0.1 → 0.9), Y: expected cost
   - Mark optimal threshold with reference line
   - Add annotation: "Optimal threshold reduces experiment cost by 23%"
3. **View 2 — Cost Decomposition:**
   - Stacked area: FN cost vs FP cost at each threshold
   - Shows trade-off between missing signals and false alarms
4. **View 3 — Optimal Threshold KPI:**
   - Large text showing optimal threshold value
   - Sub-text: expected FP rate and FN rate at optimum
5. **View 4 — Parameter: Cost Ratio Slider:**
   - Parameter control: FN Cost Weight (1 → 10)
   - When changed, recalculate optimal threshold dynamically
   - Shows that optimal threshold shifts with business requirements

**Storytelling narrative:**
> *"Every missed Higgs signal wastes millions in experimental resources. Every false positive wastes weeks of physicist time on follow-up analysis. The GBT model at threshold 0.42 minimises total experiment cost for a 3:1 FN:FP cost assumption."*

---

## Dashboard 4: Scalability & Cost Analysis

**Purpose:** Demonstrate distributed computing advantage and identify bottlenecks.

**Steps to build:**
1. Connect to `scaling_results.csv` and `stability_results.csv`
2. **View 1 — Strong Scaling Line Chart:**
   - X: Partitions (50, 100, 200, 400), Y: Training Time (s)
   - Add ideal linear scaling reference line
   - Annotate: "Bottleneck at 400 partitions — shuffle overhead"
3. **View 2 — Weak Scaling Line Chart:**
   - X: Dataset rows (millions), Y: Training Time (s)
   - Ideal = flat line (constant time as data grows proportionally)
   - Show actual vs ideal
4. **View 3 — Model Training Time Comparison:**
   - Grouped bar: PySpark MLlib (11M rows) vs sklearn (500K rows)
   - Emphasise: MLlib handles 22x more data in comparable time
5. **View 4 — Stability AUC vs Perturbation:**
   - X: % training data removed, Y: AUC with error bars
   - Demonstrate model robustness

**Annotations:**
- "Shuffle bottleneck at 400 partitions — recommend AQE + broadcast joins"
- "GBT most stable: <0.5% AUC variance at 20% perturbation"

---

## Design Guidelines

### Colour Palette
```
Signal (positive class):  #FF5722  (deep orange)
Background (negative):    #2196F3  (blue)
Accent / highlight:       #4CAF50  (green)
Neutral / borders:        #9E9E9E  (grey)
Dashboard background:     #1A1A2E  (dark navy)
```

### Typography
- Title font: Tableau Bold, 16pt
- Body: Tableau Book, 11pt
- KPI numbers: Tableau Bold, 24pt

### Interaction
- Every dashboard must have at least one filter control
- Use **Action Filters** to link views within a dashboard
- Add tooltips with: metric name, value, model name, interpretation note
- Include "Download as PDF" button on each dashboard via Tableau Server actions

### Mobile Responsiveness
- Use Tableau's **Device Layouts** → Phone layout
- Simplify to 2 views per dashboard on phone layout
- Increase font sizes for mobile: title 20pt, body 14pt

---

## Exporting for Report

To include Tableau screenshots in your report:
1. Dashboard → **Export → Image**
2. Resolution: 300 DPI
3. Format: PNG
4. Include captions explaining each dashboard's business value

---

## Performance Tips

- Always use **Tableau Extracts** (.hyper format) — never live CSV connection for dashboards
- For the 50K sample, extract takes ~5 seconds and makes all interactions instant
- Use **LOD expressions** instead of table calculations for aggregations (faster)
- Limit to 5 marks per tooltip

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CSV not loading | Run `python scripts/run_pipeline.py` first |
| Slow dashboard | Create Extract: Data → Extract Data → Save as .hyper |
| Colours not matching | Check colour palette in Format → Colour Palette |
| Mobile layout broken | Go to Dashboard → Device Layouts → Phone → Redesign |
