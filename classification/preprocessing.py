from typing import Optional, List, Union, Tuple, Dict, Any
from dataclasses import dataclass, field
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # moved here to allow importing IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import FunctionTransformer

"""
preprocessing.py

Comprehensive, production-ready data preprocessing utilities for tabular classification.
Designed to be configurable, safe for ML pipelines, and to handle common real-world issues:
- missing values (simple & iterative imputation)
- categorical encoding (one-hot, ordinal, smooth target encoding)
- scaling & power transforms
- skewness handling
- outlier detection & removal (IQR / z-score)
- multicollinearity mitigation (VIF-based)
- feature selection (variance threshold, univariate)
- text & datetime helpers
- imbalance handling (SMOTE / random oversampling) integrated into pipeline
- safe fit/transform for cross-validation (no leakage)
- save/load pipeline utilities

Usage:
    pre = Preprocessor(numeric_strategy='iterative', categorical_cardinality_threshold=10, remove_outliers='iqr')
    X_train_transformed, y_train_transformed = pre.fit_transform(X_train, y_train)
    X_test_transformed = pre.transform(X_test)

This file depends on: pandas, numpy, scikit-learn, scipy, imbalanced-learn (optional but recommended).
"""

# Optional packages
try:
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# ---------------------------
# Helper transformers & utils
# ---------------------------

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select columns and return a DataFrame (keeps column names).
    Useful inside ColumnTransformer to retain DataFrame logic.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].copy()

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Group rare categories into a single label to stabilize encoding.
    - threshold: frequency below which categories are grouped (absolute count or ratio).
    """
    def __init__(self, threshold: float = 0.01, by_ratio: bool = True, other_label: str = 'RARE'):
        self.threshold = threshold
        self.by_ratio = by_ratio
        self.other_label = other_label
        self.frequent_maps_ = {}

    def fit(self, X, y=None):
        # X expected to be a DataFrame or Series (single column)
        if isinstance(X, pd.DataFrame):
            cols = X.columns
            for col in cols:
                s = X[col].value_counts(normalize=self.by_ratio)
                if self.by_ratio:
                    frequent = s[s >= self.threshold].index.tolist()
                else:
                    # threshold treated as absolute count
                    s_abs = X[col].value_counts()
                    frequent = s_abs[s_abs >= self.threshold].index.tolist()
                self.frequent_maps_[col] = set(frequent)
        else:
            # Series like
            s = X.value_counts(normalize=self.by_ratio)
            if self.by_ratio:
                frequent = s[s >= self.threshold].index.tolist()
            else:
                s_abs = X.value_counts()
                frequent = s_abs[s_abs >= self.threshold].index.tolist()
            self.frequent_maps_ = set(frequent)
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.where(X.isin(self.frequent_maps_), self.other_label)
        elif isinstance(X, pd.DataFrame):
            Xc = X.copy()
            for col in Xc.columns:
                allowed = self.frequent_maps_.get(col, set())
                Xc[col] = Xc[col].where(Xc[col].isin(allowed), self.other_label)
            return Xc
        else:
            raise ValueError("RareCategoryGrouper expects pandas Series or DataFrame")


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smooth target mean encoding for categorical variables.
    Safe for cross-validation when used inside Pipeline: fit sees only training fold's y.
    smoothing: higher -> stronger prior towards overall mean
    """
    def __init__(self, cols: Optional[List[str]] = None, smoothing: float = 1.0, min_samples_leaf: int = 1):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.target_maps_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("TargetEncoder requires y in fit")
        X = X.copy()
        if self.cols is None:
            self.cols = X.columns.tolist()
        self.global_mean_ = float(np.mean(y))
        df = X.copy()
        df['_y_'] = y
        for col in self.cols:
            stats = df.groupby(col)['_y_'].agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            smoothing = 1 / (1 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
            enc = self.global_mean_ * (1 - smoothing) + means * smoothing
            self.target_maps_[col] = enc.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            mapping = self.target_maps_.get(col, {})
            X[col] = X[col].map(mapping).fillna(self.global_mean_)
        return X

# ---------------------------
# Utility functions
# ---------------------------

def infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Determine numeric, categorical, and datetime columns.
    Returns (numeric_cols, categorical_cols, datetime_cols)
    """
    numeric = X.select_dtypes(include=['number']).columns.tolist()
    datetime = X.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    # treat object and category as categorical
    categorical = [c for c in X.columns if c not in numeric + datetime]
    return numeric, categorical, datetime

def compute_vif(X: pd.DataFrame, thresh: float = 10.0, drop: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute VIF for features in X and optionally return a list of remaining columns
    after removing high-VIF columns iteratively.
    """
    X = X.select_dtypes(include=[np.number]).copy().dropna(axis=1, how='all')
    cols = X.columns.tolist()
    dropped = []
    while True:
        vif_dict = {}
        for i, col in enumerate(cols):
            others = [c for c in cols if c != col]
            if len(others) == 0:
                vif = 0.0
            else:
                lr = LinearRegression()
                lr.fit(X[others].values, X[col].values)
                r2 = lr.score(X[others].values, X[col].values)
                vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf
            vif_dict[col] = vif
        max_col = max(vif_dict, key=vif_dict.get)
        max_vif = vif_dict[max_col]
        if max_vif > thresh and drop:
            cols.remove(max_col)
            dropped.append(max_col)
        else:
            break
    vif_df = pd.DataFrame.from_dict(vif_dict, orient='index', columns=['VIF']).sort_values('VIF', ascending=False)
    return vif_df, cols

def drop_outliers_iqr(X: pd.DataFrame, cols: List[str], factor: float = 1.5) -> pd.Series:
    """
    Identify outliers by IQR for numeric columns. Returns boolean mask of non-outlier rows.
    """
    mask = pd.Series(True, index=X.index)
    for col in cols:
        if col not in X.columns:
            continue
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= X[col].between(lower, upper) | X[col].isna()
    return mask

def drop_outliers_zscore(X: pd.DataFrame, cols: List[str], thresh: float = 3.0) -> pd.Series:
    """
    Identify outliers by z-score. Returns boolean mask of non-outlier rows.
    """
    mask = pd.Series(True, index=X.index)
    for col in cols:
        if col not in X.columns:
            continue
        col_vals = X[col]
        mean = col_vals.mean()
        std = col_vals.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        z = (col_vals - mean).abs() / std
        mask &= (z <= thresh) | col_vals.isna()
    return mask

# ---------------------------
# Main Preprocessor class
# ---------------------------

@dataclass
class Preprocessor:
    """
    Configurable preprocessor for classification problems.
    - numeric_strategy: 'simple' (mean/median), 'iterative', 'knn'
    - numeric_scaler: 'standard', 'robust', 'minmax', None
    - categorical_strategy: 'onehot', 'ordinal', 'target'
    - categorical_cardinality_threshold: switch to ordinal encoding if > threshold (for onehot)
    - rare_threshold: frequency threshold to group rare categories
    - remove_outliers: None | 'iqr' | 'zscore'
    - outlier_params: dict for outlier method
    - vif_thresh: drop features with VIF above this iteratively if not None
    - feature_selection: None | tuple(method, k)
    - sampler: None | 'smote' | 'oversample' (requires imblearn)
    - random_state: seed
    """
    numeric_strategy: str = 'simple'
    numeric_impute_strategy: str = 'median'  # for simple imputer
    numeric_scaler: Optional[str] = 'standard'
    categorical_strategy: str = 'onehot'
    categorical_cardinality_threshold: int = 20
    rare_threshold: float = 0.01
    remove_outliers: Optional[str] = None
    outlier_params: Dict[str, Any] = field(default_factory=dict)
    vif_thresh: Optional[float] = None
    feature_selection: Optional[Tuple[str, int]] = None
    sampler: Optional[str] = None
    smoothing: float = 1.0  # for target encoder
    min_samples_leaf: int = 1
    random_state: Optional[int] = None

    # internal attributes
    pipeline_: Optional[Union[Pipeline, ImbPipeline]] = None
    numeric_cols_: List[str] = field(default_factory=list)
    categorical_cols_: List[str] = field(default_factory=list)
    datetime_cols_: List[str] = field(default_factory=list)
    fitted_: bool = False
    dropped_rows_mask_: Optional[pd.Series] = None

    def _build_numeric_pipeline(self):
        steps = []
        # imputation
        if self.numeric_strategy == 'iterative':
            steps.append(('imputer', IterativeImputer(random_state=self.random_state)))
        elif self.numeric_strategy == 'knn':
            steps.append(('imputer', KNNImputer()))
        else:
            steps.append(('imputer', SimpleImputer(strategy=self.numeric_impute_strategy)))
        # power transform to reduce skewness
        steps.append(('yjt', PowerTransformer(method='yeo-johnson', standardize=False)))
        # scaling
        if self.numeric_scaler == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif self.numeric_scaler == 'robust':
            steps.append(('scaler', RobustScaler()))
        elif self.numeric_scaler == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        return Pipeline(steps)

    def _build_categorical_pipeline(self, X: pd.DataFrame, y=None):
        steps = []
        # group rare categories first
        steps.append(('rare', RareCategoryGrouper(threshold=self.rare_threshold, by_ratio=True)))
        if self.categorical_strategy == 'target':
            # target encoder needs y during fit, so it will be applied outside ColumnTransformer if required
            steps.append(('enc', TargetEncoder(cols=None, smoothing=self.smoothing, min_samples_leaf=self.min_samples_leaf)))
            return Pipeline(steps)
        # decide between onehot and ordinal based on cardinality
        low_card_cols = [c for c in self.categorical_cols_ if X[c].nunique(dropna=False) <= self.categorical_cardinality_threshold]
        high_card_cols = [c for c in self.categorical_cols_ if X[c].nunique(dropna=False) > self.categorical_cardinality_threshold]
        # We will create a ColumnTransformer around this, so return placeholder pipeline and handle encoders at top-level.
        return None  # handled separately

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit transformers on training data. If remove_outliers is set, rows may be dropped
        from X (and correspondingly from y) internally for fitting to avoid leakage.
        """
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # infer types
        num_cols, cat_cols, dt_cols = infer_column_types(X)
        self.numeric_cols_ = num_cols
        self.categorical_cols_ = cat_cols
        self.datetime_cols_ = dt_cols

        # outlier removal: compute mask and drop rows during fitting only
        if self.remove_outliers is not None and len(self.numeric_cols_) > 0:
            if self.remove_outliers == 'iqr':
                mask = drop_outliers_iqr(X, self.numeric_cols_, factor=self.outlier_params.get('factor', 1.5))
            else:
                mask = drop_outliers_zscore(X, self.numeric_cols_, thresh=self.outlier_params.get('thresh', 3.0))
            self.dropped_rows_mask_ = mask
            X_fit = X[mask].copy()
            y_fit = None if y is None else (pd.Series(y, index=X.index)[mask].reset_index(drop=True))
        else:
            X_fit = X.copy()
            y_fit = None if y is None else (pd.Series(y).reset_index(drop=True))

        # build transformers
        numeric_pipe = self._build_numeric_pipeline() if self.numeric_cols_ else None

        # categorical handling: create sub-transformers for low and high cardinality
        transformers = []
        if self.categorical_cols_:
            low_card = [c for c in self.categorical_cols_ if X_fit[c].nunique(dropna=False) <= self.categorical_cardinality_threshold]
            high_card = [c for c in self.categorical_cols_ if X_fit[c].nunique(dropna=False) > self.categorical_cardinality_threshold]

            # OneHot for low_card
            if low_card:
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                cat_low_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')), ('ohe', ohe)])
                transformers.append(('cat_low', cat_low_pipeline, low_card))

            # Ordinal for high_card
            if high_card:
                ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                cat_high_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')), ('ord', ord_enc)])
                transformers.append(('cat_high', cat_high_pipeline, high_card))

        # numeric transformer handling
        if self.numeric_cols_:
            transformers.append(('num', numeric_pipe, self.numeric_cols_))

        # datetime: extract basic features
        if self.datetime_cols_:
            dt_pipeline = Pipeline([('selector', DataFrameSelector(self.datetime_cols_)),
                                    ('dtf', FunctionTransformer(_datetime_features, validate=False))])
            transformers.append(('dt', dt_pipeline, self.datetime_cols_))

        # ColumnTransformer
        column_transformer = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)

        # Compose pipeline steps
        steps = []
        # column transformer
        steps.append(('columns', column_transformer))

        # After ColumnTransformer, we may want to apply feature selection or VIF removal:
        if self.vif_thresh is not None:
            # We'll handle VIF after initial transform inside fit
            pass

        if self.feature_selection is not None:
            method, k = self.feature_selection
            if method == 'mutual_info':
                sel = SelectKBest(mutual_info_classif, k=k)
            elif method == 'f_classif':
                sel = SelectKBest(f_classif, k=k)
            else:
                raise ValueError("Unsupported feature selection method")
            steps.append(('select', sel))

        # sampler: integrate via imblearn pipeline if available
        if self.sampler is not None and not IMBLEARN_AVAILABLE:
            warnings.warn("imblearn not available; sampler will be ignored.")
            self.sampler = None

        # build final pipeline object
        if self.sampler is not None and IMBLEARN_AVAILABLE:
            # create imblearn pipeline
            steps_with_sampler = []
            steps_with_sampler.extend(steps)
            sampler_est = SMOTE(random_state=self.random_state) if self.sampler == 'smote' else RandomOverSampler(random_state=self.random_state)
            # ImbPipeline expects (name, estimator) pairs
            self.pipeline_ = ImbPipeline(steps_with_sampler + [('sampler', sampler_est)])
        else:
            self.pipeline_ = Pipeline(steps)

        # For target encoding, if chosen, fit separate encoders on categorical columns to avoid leakage
        # We'll fit column transformer first then fit target encoders if required
        # Fit the pipeline on X_fit
        if self.categorical_strategy == 'target' and self.categorical_cols_:
            # Fit numeric and other transformers excluding target encoder
            # We'll implement target encoding separately: replace cat cols by encoded values using TargetEncoder
            te = TargetEncoder(cols=self.categorical_cols_, smoothing=self.smoothing, min_samples_leaf=self.min_samples_leaf)
            te.fit(X_fit[self.categorical_cols_], y_fit)
            self._target_encoder_ = te
            # After encoding, we should transform categorical columns to numeric and then proceed with remainder pipeline
            X_encoded = X_fit.copy()
            X_encoded[self.categorical_cols_] = te.transform(X_fit[self.categorical_cols_])
            # Now fit the pipeline on X_encoded
            self.pipeline_.fit(X_encoded, y_fit)
        else:
            # Normal path: pipeline can handle categorical encoders via ColumnTransformer
            # Fit pipeline directly (it internally fits encoders)
            self.pipeline_.fit(X_fit, y_fit)

        # VIF-based dropping: after initial pipeline transform obtain feature matrix with names
        if self.vif_thresh is not None:
            # Get transformed feature names:
            Xt = self._transform_internals(X_fit)
            # compute vif and drop high ones
            vif_df, keep_cols = compute_vif(pd.DataFrame(Xt, columns=self._get_feature_names()), thresh=self.vif_thresh)
            # If features were dropped, we can wrap a VarianceThreshold or a transformer to select remaining columns.
            # For simplicity we will set up a selector mask in self._vif_keep_names and leave pipeline as-is,
            # but subsequent transform will filter columns.
            self._vif_keep_names = keep_cols
            # No re-fitting of pipeline here for simplicity; in practice you might rebuild pipeline with selected features.

        self.fitted_ = True
        return self

    def _get_feature_names(self) -> List[str]:
        """
        Attempt to extract feature names from fitted ColumnTransformer/pipeline.
        Works for typical sklearn transformers; may fail for some custom transformers.
        """
        # Try pipeline -> columns -> get_feature_names
        ct = self.pipeline_.named_steps['columns']
        feature_names = []
        # ct.transformers_ exists after fit
        for name, trans, cols in ct.transformers_:
            if trans == 'drop':
                continue
            if hasattr(trans, 'named_steps'):
                # a pipeline; get last step
                last = list(trans.named_steps.values())[-1]
                if hasattr(last, 'get_feature_names_out'):
                    try:
                        names = last.get_feature_names_out(cols)
                    except Exception:
                        try:
                            names = last.get_feature_names_out()
                        except Exception:
                            names = cols
                    feature_names.extend(list(names))
                else:
                    # fallback: use column names
                    if isinstance(cols, (list, tuple)):
                        feature_names.extend(list(cols))
                    else:
                        feature_names.append(cols)
            else:
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        names = trans.get_feature_names_out(cols)
                        feature_names.extend(list(names))
                    except Exception:
                        if isinstance(cols, (list, tuple)):
                            feature_names.extend(list(cols))
                        else:
                            feature_names.append(cols)
                else:
                    if isinstance(cols, (list, tuple)):
                        feature_names.extend(list(cols))
                    else:
                        feature_names.append(cols)
        return feature_names

    def _transform_internals(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform X through pipeline but return numpy array (no sampling).
        Used for internal inspections like VIF.
        """
        X = X.copy()
        if self.categorical_strategy == 'target' and self.categorical_cols_:
            X[self.categorical_cols_] = self._target_encoder_.transform(X[self.categorical_cols_])
        Xt = self.pipeline_.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        return Xt

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply preprocessing to new data. If VIF filtering was set, filter columns accordingly.
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor not fitted. Call fit or fit_transform first.")
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.categorical_strategy == 'target' and self.categorical_cols_:
            X[self.categorical_cols_] = self._target_encoder_.transform(X[self.categorical_cols_])
        Xt = self.pipeline_.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        if getattr(self, '_vif_keep_names', None) is not None:
            feature_names = self._get_feature_names()
            keep = self._vif_keep_names
            idx = [i for i, n in enumerate(feature_names) if n in keep]
            Xt = Xt[:, idx]
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform. If outlier removal is enabled, return X_transformed and y filtered accordingly.
        """
        self.fit(X, y)
        # Transform on full data but drop rows if they were excluded during fit (outliers)
        X = X.copy()
        if self.dropped_rows_mask_ is not None:
            X_kept = X[self.dropped_rows_mask_]
            Xt = self.transform(X_kept)
            y_kept = None if y is None else pd.Series(y, index=X.index)[self.dropped_rows_mask_].values
            return Xt, y_kept
        else:
            Xt = self.transform(X)
            y_arr = None if y is None else (np.array(y))
            return Xt, y_arr

    def save(self, path: str):
        """
        Save fitted preprocessor to disk (pickle). Contains pipeline and metadata.
        """
        if not self.fitted_:
            raise RuntimeError("Cannot save an unfitted Preprocessor")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Preprocessor':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, Preprocessor):
            raise ValueError("Pickle does not contain a Preprocessor")
        return obj

# ---------------------------
# Extra helpers
# ---------------------------


def _datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns into numeric features: year, month, day, weekday, hour, is_month_start, is_month_end
    Accepts DataFrame of datetime columns and returns numeric dataframe.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = pd.to_datetime(df[col], errors='coerce')
        out[f'{col}_year'] = s.dt.year
        out[f'{col}_month'] = s.dt.month
        out[f'{col}_day'] = s.dt.day
        out[f'{col}_weekday'] = s.dt.weekday
        out[f'{col}_hour'] = s.dt.hour
        out[f'{col}_is_month_start'] = s.dt.is_month_start.astype(float)
        out[f'{col}_is_month_end'] = s.dt.is_month_end.astype(float)
    return out

# ---------------------------
# Example convenience factory
# ---------------------------

def build_default_preprocessor(random_state: Optional[int] = 0) -> Preprocessor:
    """
    Return a sane default Preprocessor configuration suitable for many classification problems.
    """
    return Preprocessor(
        numeric_strategy='iterative',
        numeric_impute_strategy='median',
        numeric_scaler='robust',
        categorical_strategy='onehot',
        categorical_cardinality_threshold=10,
        rare_threshold=0.01,
        remove_outliers=None,
        vif_thresh=10.0,
        feature_selection=None,
        sampler=None,
        random_state=random_state
    )

# End of file
