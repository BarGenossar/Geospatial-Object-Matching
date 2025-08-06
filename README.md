# 3dSAGER: Geospatial Entity Resolution over 3D Objects

**3dSAGER** (3D Spatial-Aware Geospatial Entity Resolution) is an end-to-end pipeline for geospatial entity resolution over 3D objects. Unlike traditional methods that rely on spatial proximity, textual metadata, or external identifiers, 3dSAGER captures *intrinsic geometric characteristics* to robustly match spatial objects across datasets, even when coordinate systems are incompatible.

A key component of 3dSAGER is **BKAFI**, a lightweight and interpretable blocking method that efficiently generates high-recall candidate sets.

> 📄 The paper is currently under review for [SIGMOD 2026](https://2026.sigmod.org/).

<p align="center">
  <img src="intro_fig.png" alt="3dSAGER Overview" width="600"/>
</p>

---

## 📦 Dataset

You can download the *The Hague* dataset using the following link:

🔗 [Download dataset](https://tinyurl.com/3dSAGERdataset)

We provide further instructions for working with the *The Hague* dataset in the [`Working_with_The_Hague_Dataset.ipynb`](Working_with_The_Hague_Dataset.ipynb) notebook.


---

## 🚀 Running Experiments

You can run the experiments using the provided shell script:

```bash
bash run_experiments.sh
```
In the run_experiments.sh script, you must define the evaluation mode by setting eval_mode to either "blocking" or "matching". For a single basic experimental configuration, we recommend using the "small" or "large" dataset size, setting bkafi_criterion="feature_importance", normalizations=True, and sdr_factor=False.



Here are some of the main variables to be configured in `config.py`:


| Variable                                   | Description                                                                                              |
| ------------------------------------------ |----------------------------------------------------------------------------------------------------------|
| `dataset_name`                             | Dataset key: `"Hague"`                                                                                   |
| `evaluation_mode`                          | `"blocking"` or `"matching"`                                                                             |
| `dataset_size_version`                     | Size variant: `"small"`, `"large"`                                                                       |
| `matching_cands_generation`                | How to generate candidate pairs (`"blocking-based"` or `"negative_sampling"`). Select `"blocking-based"` |
| `neg_samples_num`                          | Number of negative samples per positive. Select 2.                                                       |
| `seeds_num`                                | Number of seeds for experiments. Select 3.                                                               |
| `blocking_method`          | `'bkafi'`                                                                                                |
| `cand_pairs_per_item_list` | List of candidate counts per object                                                                      |
| `nn_param`                 | Number of nearest neighbors to retrieve. Select 20.                                                      |
| `sdr_factor`               | False                                                                                                    |
| `bkafi_criterion`          | Feature selection strategy: `'feature_importance'` or `'std'`. Select `'feature_importance'`                                   |
| `model_to_use`   | Default classifier used for prediction               |
| `model_list`     | Available models for matching                        |
| `blocking_model` | Classifier used in the blocking stage                |
| `params_dict`    | Hyperparameter grid per model (for cross-validation) |
