import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
import json
import random
import pickle
import tempfile
import zipfile
import builtins
import warnings
import re
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import InconsistentVersionWarning

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Concatenate,
    Conv1D,
    Add,
    Attention,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    Reshape,
    Softmax,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# =========================================================
# 0. Config
# =========================================================
def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(float(value))


def env_optional_int(name: str, default=None):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(float(value))


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float_list(name: str, default: List[float]) -> List[float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [float(part.strip()) for part in value.split(",") if part.strip()]


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = env_str("TEE_JSON_DIR", os.path.join(PROJECT_ROOT, "data", "hybrid_json"))
JSON_PATH = env_str("TEE_JSON_PATH", os.path.join(JSON_DIR, "模型1.json"))
OUTPUT_DIR = env_str("TEE_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "results", "new_run"))

RANDOM_SEED = env_int("TEE_RANDOM_SEED", 2031)
TIME_STEPS = env_int("TEE_TIME_STEPS", 10)
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TEST_JSON_INDEX = 0
PHASE_SPLIT_BLOCKS = env_int("TEE_PHASE_SPLIT_BLOCKS", 5)
RANDOM_CHUNK_SPLIT_COUNT = env_int("TEE_RANDOM_CHUNK_SPLIT_COUNT", 20)
SPLIT_MODE = env_str("TEE_SPLIT_MODE", "phase_blocks")  # "phase_blocks", "random_phase_chunks", "random_frames", or "leave_one_json_out"
PREDICT_SOURCE_INDEX = env_optional_int("TEE_PREDICT_SOURCE_INDEX", None)
PREDICT_SOURCE_NAME = env_str("TEE_PREDICT_SOURCE_NAME", "")
IMPORTED_SOURCE_INDEX_BY_NAME: Dict[str, int] = {}
SELECTED_FILE_INDEX_OFFSET = 0
SELECTED_JSON_PATHS_FOR_OUTPUT: List[str] = []
BATCH_SIZE = env_int("TEE_BATCH_SIZE", 32)
EPOCHS = env_int("TEE_EPOCHS", 400)
PATIENCE = env_int("TEE_PATIENCE", 60)
EARLY_STOP_MIN_DELTA = 1e-5
REDUCE_LR_PATIENCE = env_int("TEE_REDUCE_LR_PATIENCE", 12)
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
L2_REG = env_float("TEE_L2_REG", 2e-4)
STATIC_DENSE_UNITS = env_float_list("TEE_STATIC_DENSE_UNITS", [256, 160, 96, 48])
STATIC_DROPOUT_RATES = env_float_list("TEE_STATIC_DROPOUT_RATES", [0.18, 0.15])
TEMPORAL_UNITS = env_float_list("TEE_TEMPORAL_UNITS", [96, 48])
SEQ_DROPOUT_RATE = env_float("TEE_SEQ_DROPOUT_RATE", 0.35)
SEQ_DENSE_UNITS = env_int("TEE_SEQ_DENSE_UNITS", 24)
FUSED_DENSE_UNITS = env_float_list("TEE_FUSED_DENSE_UNITS", [160, 80])
FUSED_DROPOUT_RATE = env_float("TEE_FUSED_DROPOUT_RATE", 0.16)
AXIS_HEAD_UNITS = env_int("TEE_AXIS_HEAD_UNITS", 32)
USE_MOE_OUTPUT_HEAD = env_bool("TEE_USE_MOE_OUTPUT_HEAD", False)
MOE_EXPERT_COUNT = env_int("TEE_MOE_EXPERT_COUNT", 4)
MOE_EXPERT_UNITS = env_int("TEE_MOE_EXPERT_UNITS", 24)
USE_CONDITION_GUIDED_MOE = env_bool("TEE_USE_CONDITION_GUIDED_MOE", True)
MOE_EXPERT_HIDDEN_UNITS = env_float_list("TEE_MOE_EXPERT_HIDDEN_UNITS", [64, 32])
MOE_GATE_HIDDEN_UNITS = env_float_list("TEE_MOE_GATE_HIDDEN_UNITS", [32, 16])
ERR_THRESHOLD = 0.04
POSITION_DISTANCE_THRESHOLD = 0.04
AXIS_FINE_THRESHOLD = env_float("TEE_AXIS_FINE_THRESHOLD", 0.023)
ERROR_CLASS_EDGES = [-0.12, -0.10, -0.08, -0.06, -0.04, -0.03, -0.02, -0.01,
                     0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12]
ERROR_CLASS_COUNT = len(ERROR_CLASS_EDGES) + 1
ERROR_CLASS_LOSS_WEIGHT = 0.1
DISTANCE_AUX_LOSS_WEIGHT = env_float("TEE_DISTANCE_AUX_LOSS_WEIGHT", 0.0)
RISK_AUX_LOSS_WEIGHT = env_float("TEE_RISK_AUX_LOSS_WEIGHT", 0.0)
EUCLIDEAN_DISTANCE_LOSS_WEIGHT = env_float("TEE_EUCLIDEAN_DISTANCE_LOSS_WEIGHT", 4.2)
POSITION_THRESHOLD_LOSS_WEIGHT = env_float("TEE_POSITION_THRESHOLD_LOSS_WEIGHT", 7.5)
AXIS_FINE_THRESHOLD_LOSS_WEIGHT = env_float("TEE_AXIS_FINE_THRESHOLD_LOSS_WEIGHT", 1.0)
DISTANCE_PRIMARY_AXIS_LOSS_WEIGHT = env_float("TEE_DISTANCE_PRIMARY_AXIS_LOSS_WEIGHT", 0.15)
DISTANCE_MARGIN_SHARPNESS = env_float("TEE_DISTANCE_MARGIN_SHARPNESS", 80.0)
REGRESSION_LOSS_MODE = env_str("TEE_REGRESSION_LOSS_MODE", "axis_aware")
Y_AXIS_LOSS_WEIGHT = 1.0
Y_AXIS_THRESHOLD_LOSS_WEIGHT = 0.0
HIGH_VALUE_TOP_CLASSES = 3
HIGH_VALUE_REG_LOSS_WEIGHT = 0.3
USE_RESIDUAL_CORRECTION = env_bool("TEE_USE_RESIDUAL_CORRECTION", False)
RESIDUAL_MAX_ABS_CORRECTION = env_float("TEE_RESIDUAL_MAX_ABS_CORRECTION", 0.04)
RESIDUAL_SCALE_CANDIDATES = env_float_list("TEE_RESIDUAL_SCALE_CANDIDATES", [0.0])
USE_NEAR_BOUNDARY_RESIDUAL = env_bool("TEE_USE_NEAR_BOUNDARY_RESIDUAL", False)
USE_PREDICTED_DISTANCE_RESIDUAL_GATE = env_bool("TEE_USE_PREDICTED_DISTANCE_RESIDUAL_GATE", True)
USE_LATE_STAGE_RESIDUAL = env_bool("TEE_USE_LATE_STAGE_RESIDUAL", False)
LATE_STAGE_RESIDUAL_TRAIN_LOW = env_float("TEE_LATE_STAGE_RESIDUAL_TRAIN_LOW", 0.80)
LATE_STAGE_RESIDUAL_APPLY_LOW = env_float("TEE_LATE_STAGE_RESIDUAL_APPLY_LOW", 0.85)
LATE_STAGE_RESIDUAL_APPLY_FULL = env_float("TEE_LATE_STAGE_RESIDUAL_APPLY_FULL", 0.95)
LATE_STAGE_RESIDUAL_MAX_ABS = env_float("TEE_LATE_STAGE_RESIDUAL_MAX_ABS", 0.020)
LATE_STAGE_RESIDUAL_MIN_SAMPLES = env_int("TEE_LATE_STAGE_RESIDUAL_MIN_SAMPLES", 24)
LATE_STAGE_RESIDUAL_SCALE_CANDIDATES = env_float_list("TEE_LATE_STAGE_RESIDUAL_SCALE_CANDIDATES", [0.0, 0.25, 0.5, 0.75, 1.0])
NEAR_BOUNDARY_RESIDUAL_TRAIN_LOW = env_float("TEE_NEAR_BOUNDARY_RESIDUAL_TRAIN_LOW", 0.030)
NEAR_BOUNDARY_RESIDUAL_TRAIN_HIGH = env_float("TEE_NEAR_BOUNDARY_RESIDUAL_TRAIN_HIGH", 0.070)
NEAR_BOUNDARY_RESIDUAL_APPLY_LOW = env_float("TEE_NEAR_BOUNDARY_RESIDUAL_APPLY_LOW", 0.035)
NEAR_BOUNDARY_RESIDUAL_APPLY_HIGH = env_float("TEE_NEAR_BOUNDARY_RESIDUAL_APPLY_HIGH", 0.060)
NEAR_BOUNDARY_RESIDUAL_HARD_HIGH = env_float("TEE_NEAR_BOUNDARY_RESIDUAL_HARD_HIGH", 0.080)
NEAR_BOUNDARY_RESIDUAL_MAX_ABS = env_float("TEE_NEAR_BOUNDARY_RESIDUAL_MAX_ABS", 0.015)
NEAR_BOUNDARY_RESIDUAL_SCALE_CANDIDATES = env_float_list("TEE_NEAR_BOUNDARY_RESIDUAL_SCALE_CANDIDATES", [0.0, 0.25, 0.5, 0.75, 1.0])
USE_NEAR_THRESHOLD_REWEIGHTING = env_bool("TEE_USE_NEAR_THRESHOLD_REWEIGHTING", False)
NEAR_THRESHOLD_FINE_TUNE_EPOCHS = env_int("TEE_NEAR_THRESHOLD_FINE_TUNE_EPOCHS", 60)
NEAR_THRESHOLD_FINE_TUNE_PATIENCE = env_int("TEE_NEAR_THRESHOLD_FINE_TUNE_PATIENCE", 15)
NEAR_THRESHOLD_EASY_DISTANCE = env_float("TEE_NEAR_THRESHOLD_EASY_DISTANCE", 0.025)
NEAR_THRESHOLD_PROTECT_DISTANCE = env_float("TEE_NEAR_THRESHOLD_PROTECT_DISTANCE", 0.035)
NEAR_THRESHOLD_RESCUE_DISTANCE = env_float("TEE_NEAR_THRESHOLD_RESCUE_DISTANCE", 0.045)
NEAR_THRESHOLD_EASY_WEIGHT = env_float("TEE_NEAR_THRESHOLD_EASY_WEIGHT", 0.8)
NEAR_THRESHOLD_STABLE_WEIGHT = env_float("TEE_NEAR_THRESHOLD_STABLE_WEIGHT", 1.0)
NEAR_THRESHOLD_PROTECT_WEIGHT = env_float("TEE_NEAR_THRESHOLD_PROTECT_WEIGHT", 1.0)
NEAR_THRESHOLD_RESCUE_WEIGHT = env_float("TEE_NEAR_THRESHOLD_RESCUE_WEIGHT", 1.8)
NEAR_THRESHOLD_FAR_WEIGHT = env_float("TEE_NEAR_THRESHOLD_FAR_WEIGHT", 1.0)
NEAR_THRESHOLD_FINE_TUNE_LR = env_float("TEE_NEAR_THRESHOLD_FINE_TUNE_LR", 2e-4)
PER_SOURCE_RESIDUAL_MIN_SAMPLES = 12
USE_SOURCE_OFFSET_CALIBRATION = False
USE_SOURCE_REGION_CALIBRATION = False
USE_PER_SOURCE_RESIDUAL = False
USE_SOURCE_PROGRESS_FEATURES = env_bool("TEE_USE_SOURCE_PROGRESS_FEATURES", True)
USE_THERMAL_ACCUMULATION_FEATURES = env_bool("TEE_USE_THERMAL_ACCUMULATION_FEATURES", True)
FEATURE_SET_MODE = env_str("TEE_FEATURE_SET_MODE", "minimal")
THERMAL_ACCUMULATION_MODE = env_str("TEE_THERMAL_ACCUMULATION_MODE", "summary")
USE_LONG_ROLLING_FEATURES = env_bool("TEE_USE_LONG_ROLLING_FEATURES", False)
LONG_ROLLING_WINDOWS = [
    int(round(v))
    for v in env_float_list("TEE_LONG_ROLLING_WINDOWS", [8, 20, 40])
    if int(round(v)) > 1
]
MODEL_BACKEND = "hybrid_nn"
RUN_MODE = env_str("TEE_RUN_MODE", "predict_only")       # "train" trains then predicts; "predict_only" loads saved Hybrid model
HYBRID_MODEL_FILE = env_str(
    "TEE_HYBRID_MODEL_FILE",
    os.path.join(
        PROJECT_ROOT,
        "models",
        "hybrid_v10",
        "best_full_dataset",
        "hybrid_spatiotemporal_model_v10_612_current_loss_e4.2_p7.5_a1.keras",
    ),
)
HYBRID_ARTIFACT_FILE = env_str(
    "TEE_HYBRID_ARTIFACT_FILE",
    os.path.join(
        PROJECT_ROOT,
        "models",
        "hybrid_v10",
        "best_full_dataset",
        "hybrid_spatiotemporal_artifacts_v10_612_current_loss_e4.2_p7.5_a1.pkl",
    ),
)
HYBRID_MODEL_PATH = env_str("TEE_HYBRID_MODEL_PATH", "")
HYBRID_ARTIFACT_PATH = env_str("TEE_HYBRID_ARTIFACT_PATH", "")
STOP_AFTER_SOURCE_SUMMARY = env_bool("TEE_STOP_AFTER_SOURCE_SUMMARY", False)
CONSOLE_OUTPUT_MODE = env_str("TEE_CONSOLE_OUTPUT", "predict_report")  # "predict_report", "full", "quiet", or "until_source_summary"
OPEN_PLOTS_AFTER_RUN = env_bool("TEE_OPEN_PLOTS_AFTER_RUN", False)
USE_Y_AXIS_TARGETED_FEATURES = False
USE_SOURCE_ONEHOT_FEATURES = env_bool("TEE_USE_SOURCE_ONEHOT_FEATURES", False)
USE_PHYSICAL_CONDITION_STATIC_FEATURES = env_bool("TEE_USE_PHYSICAL_CONDITION_STATIC_FEATURES", True)
TEMPORAL_BACKBONE = env_str("TEE_TEMPORAL_BACKBONE", "gru_attention")
SHOW_PLOTS = False       # True: 运行时弹图；False: 只保存图
VERBOSE = 2              # 2 每个 epoch 一行；1 进度条

XGB_N_ESTIMATORS = 1200
XGB_LEARNING_RATE = 0.03
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = 2.0
XGB_REG_ALPHA = 0.2
XGB_MIN_CHILD_WEIGHT = 4.0
XGB_EARLY_STOPPING_ROUNDS = 60


def configure_console_output() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)

    mode = CONSOLE_OUTPUT_MODE.strip().lower()
    if mode == "auto":
        mode = "predict_report" if RUN_MODE.strip().lower() == "predict_only" else "full"
    if mode == "full":
        return
    if mode == "predict_report":
        original_print = builtins.print

        def predict_report_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            text = sep.join(str(arg) for arg in args)
            stripped = text.strip()
            if (
                stripped.startswith("正在")
                or stripped.startswith("模型导入完成:")
                or stripped.startswith("导入的加工过程 JSON 文件")
                or stripped.startswith("导入文件 -")
                or stripped.startswith("测试集总体结果:")
                or stripped.startswith("预测明细已保存:")
                or stripped.startswith("预测图像已保存:")
                or stripped == "测试阶段完成。"
                or stripped == "逐点预测结果:"
                or stripped.startswith("第")
            ):
                original_print(*args, **kwargs)

        builtins.print = predict_report_print
        return
    if mode == "until_source_summary":
        original_print = builtins.print
        suppress_after_source_summary = {"enabled": False}
        allowed_after_suppression = (
            "Saving comparison figures",
            "Saved figures to:",
            "Prediction test run complete",
        )

        def report_until_source_summary_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            text = sep.join(str(arg) for arg in args)
            if "Hybrid Test Error Buckets" in text:
                suppress_after_source_summary["enabled"] = True
                return
            if suppress_after_source_summary["enabled"] and any(
                fragment in text for fragment in allowed_after_suppression
            ):
                original_print(*args, **kwargs)
                return
            if not suppress_after_source_summary["enabled"]:
                original_print(*args, **kwargs)

        builtins.print = report_until_source_summary_print
        return

    quiet = mode == "quiet"
    if not quiet:
        return

    original_print = builtins.print
    axis_lines_remaining = {"count": 0}
    important_fragments = (
        "Loading data",
        "Found ",
        "RUN_MODE =",
        "Hybrid model path =",
        "Hybrid artifact path =",
        "Loading saved spatial + temporal hybrid model",
        "Loaded trained Hybrid model from",
        "Running model prediction",
        "Evaluating prediction metrics",
        "Saving comparison figures",
        "Building prediction tables",
        "Saving diagnostic CSV files",
        "Prediction test run complete",
        "Frame split ->",
        "Point split ->",
        "Train hybrid samples",
        "Val hybrid samples",
        "Test hybrid samples",
        "Hybrid Train Actual Position Distance",
        "Hybrid Val Actual Position Distance",
        "Hybrid Test Actual Position Distance",
        "Saved predictions to:",
        "Saved position-distance failures to:",
        "Saved source failure summary to:",
        "Saved source diagnostic summary to:",
        "Saved near-boundary diagnostic to:",
        "Saved feature bucket diagnostic to:",
        "Saved line/window diagnostic to:",
        "Saved phase/progress diagnostic to:",
        "Saved position-distance bar plots to:",
        "Saved figures to:",
    )

    def quiet_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        text = sep.join(str(arg) for arg in args)
        stripped = text.strip()
        if stripped in {"Hybrid Train", "Hybrid Val", "Hybrid Test"}:
            axis_lines_remaining["count"] = 3
            original_print(*args, **kwargs)
            return
        if stripped.startswith(("X: RMSE=", "Y: RMSE=", "Z: RMSE=")) and axis_lines_remaining["count"] > 0:
            axis_lines_remaining["count"] -= 1
            original_print(*args, **kwargs)
            return
        if any(fragment in text for fragment in important_fragments):
            original_print(*args, **kwargs)

    builtins.print = quiet_print

TEMP_CHANNELS = [3, 4, 5, 6]
SPINDLE_TEMP_CHANNELS = [3, 4, 6]
AMBIENT_TEMP_CHANNEL = 5
USE_VIB_FEATURES = [
    "ac_rms_g",
    "peak_to_peak_g",
    "band_energy_1000_3000",
]
FRAME_KEYS = ["mapped_file_index", "mapped_nc_line", "mapped_frame_num", "mapped_frame_id"]
TARGET_COLS = ["err_x", "err_y", "err_z"]
ERROR_CLASS_COLS = [f"{col}_class" for col in TARGET_COLS]
POSE_COLS = ["pose_x", "pose_y", "pose_z"]
SOURCE_ONEHOT_PREFIX = "source_id_"
PHYSICAL_CONDITION_FEATURE_CANDIDATES = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_r_xy",
    "pose_r_xyz",
    "feed_speed",
    "spindle_speed",
    "temp_median_mean",
    "temp_median_range",
    "temp_std_mean",
    "temp_ch3_median_c",
    "temp_ch4_median_c",
    "temp_ch5_median_c",
    "temp_ch6_median_c",
    "temp_spindle_mean",
    "temp_spindle_range",
    "temp_spindle_std_mean",
    "temp_ambient_c",
    "temp_spindle_ambient_delta",
    "temp_ch3_ambient_delta",
    "temp_ch4_ambient_delta",
    "temp_ch6_ambient_delta",
    "source_line_progress",
    "source_frame_progress",
    "source_line_progress_centered",
    "source_frame_progress_centered",
    "source_line_frame_progress_diff",
    "progress_temp_interact",
    "progress_motion_acc_interact",
]
MOE_CONDITION_FEATURE_CANDIDATES = PHYSICAL_CONDITION_FEATURE_CANDIDATES


def apply_cli_args() -> None:
    global JSON_DIR, JSON_PATH, OUTPUT_DIR, PREDICT_SOURCE_INDEX, PREDICT_SOURCE_NAME
    global RUN_MODE, HYBRID_MODEL_FILE, HYBRID_ARTIFACT_FILE
    global HYBRID_MODEL_PATH, HYBRID_ARTIFACT_PATH
    global OPEN_PLOTS_AFTER_RUN

    parser = argparse.ArgumentParser(
        description="Train or run single-model prediction for thermal expansion error."
    )
    parser.add_argument("--run-mode", choices=["train", "predict_only"], help="Execution mode.")
    parser.add_argument("--model-path", help="Full path to a saved .keras/.h5 model for prediction.")
    parser.add_argument("--artifact-path", help="Full path to the matching preprocessing .pkl artifact.")
    parser.add_argument("--model-file", help="Model filename under the output directory.")
    parser.add_argument("--artifact-file", help="Artifact filename under the output directory.")
    parser.add_argument("--output-dir", help="Directory used for model/artifact lookup and prediction outputs.")
    parser.add_argument("--data-dir", help="Directory containing the training/prediction JSON files.")
    parser.add_argument("--predict-source-index", type=int, help="Zero-based source index. Use 9 for machining process 10.")
    parser.add_argument("--predict-source-name", help="Exact source JSON filename to predict, for example 模型10.json.")
    parser.add_argument("--open-plots", action="store_true", help="Open prediction plots after saving them.")
    parser.add_argument("--no-open-plots", action="store_true", help="Do not open prediction plots after saving them.")
    args = parser.parse_args()

    if args.data_dir:
        JSON_DIR = os.path.abspath(args.data_dir)
        JSON_PATH = ""
    OUTPUT_DIR = args.output_dir or os.environ.get("TEE_OUTPUT_DIR") or OUTPUT_DIR
    if args.run_mode:
        RUN_MODE = args.run_mode
    if args.model_path:
        HYBRID_MODEL_PATH = args.model_path
        RUN_MODE = "predict_only"
    if args.artifact_path:
        HYBRID_ARTIFACT_PATH = args.artifact_path
    if args.model_file:
        HYBRID_MODEL_FILE = args.model_file
    if args.artifact_file:
        HYBRID_ARTIFACT_FILE = args.artifact_file
    if args.predict_source_index is not None:
        PREDICT_SOURCE_INDEX = args.predict_source_index
        PREDICT_SOURCE_NAME = ""
    if args.predict_source_name:
        PREDICT_SOURCE_NAME = args.predict_source_name
        PREDICT_SOURCE_INDEX = None
    if args.open_plots:
        OPEN_PLOTS_AFTER_RUN = True
    if args.no_open_plots:
        OPEN_PLOTS_AFTER_RUN = False

    # Do not overwrite the bundled best-model artifact when a new training run
    # is launched without explicit output filenames.
    if (
        RUN_MODE.strip().lower() == "train"
        and not args.model_path
        and not args.model_file
        and not os.environ.get("TEE_HYBRID_MODEL_FILE")
    ):
        HYBRID_MODEL_PATH = ""
        HYBRID_MODEL_FILE = "hybrid_spatiotemporal_model.keras"
    if (
        RUN_MODE.strip().lower() == "train"
        and not args.artifact_path
        and not args.artifact_file
        and not os.environ.get("TEE_HYBRID_ARTIFACT_FILE")
    ):
        HYBRID_ARTIFACT_PATH = ""
        HYBRID_ARTIFACT_FILE = "hybrid_spatiotemporal_artifacts.pkl"

# =========================================================
# 1. Reproducibility
# =========================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)



def peak_aware_loss(y_true, y_pred):
    err = y_true - y_pred
    base = tf.keras.losses.huber(y_true, y_pred, delta=0.5)

    abs_true = tf.abs(y_true)
    abs_pred = tf.abs(y_pred)

    # 真实值是峰值，但预测幅度偏小，就额外罚
    peak_underfit = tf.nn.relu(abs_true - abs_pred)

    peak_weight = 1.0 + 2.0 * tf.cast(abs_true > 1.0, tf.float32)
    peak_penalty = peak_weight * tf.square(peak_underfit)

    return tf.reduce_mean(base + 0.3 * peak_penalty)


def elementwise_huber_loss(y_true, y_pred):
    err = y_true - y_pred
    abs_err = tf.abs(err)
    delta = tf.constant(1.0, dtype=y_pred.dtype)
    return tf.where(
        abs_err <= delta,
        0.5 * tf.square(err),
        delta * (abs_err - 0.5 * delta),
    )


def make_position_aware_regression_loss(target_scale: np.ndarray):
    target_scale = np.asarray(target_scale, dtype=np.float32).reshape(1, -1)
    axis_weights = np.asarray([1.0, Y_AXIS_LOSS_WEIGHT, 1.0], dtype=np.float32).reshape(1, -1)

    def position_aware_regression_loss(y_true, y_pred):
        weights = tf.cast(tf.constant(axis_weights), y_pred.dtype)

        scale = tf.cast(tf.constant(target_scale), y_pred.dtype)
        raw_axis_err = (y_true - y_pred) * scale
        raw_abs_axis_err = tf.abs(raw_axis_err)
        raw_distance = tf.sqrt(
            tf.reduce_sum(tf.square(raw_axis_err), axis=-1, keepdims=True) + tf.keras.backend.epsilon()
        )

        distance_delta = tf.cast(POSITION_DISTANCE_THRESHOLD, y_pred.dtype)
        margin_sharpness = tf.cast(DISTANCE_MARGIN_SHARPNESS, y_pred.dtype)
        distance_margin = tf.nn.softplus(margin_sharpness * (raw_distance - distance_delta)) / margin_sharpness

        axis_delta = tf.cast(AXIS_FINE_THRESHOLD, y_pred.dtype)
        axis_huber = tf.where(
            raw_abs_axis_err <= axis_delta,
            0.5 * tf.square(raw_abs_axis_err) / axis_delta,
            raw_abs_axis_err - 0.5 * axis_delta,
        )
        axis_stabilizer = tf.reduce_mean(axis_huber * weights, axis=-1, keepdims=True)
        fine_axis_penalty = tf.reduce_mean(tf.square(tf.nn.relu(raw_abs_axis_err - axis_delta)) * weights, axis=-1, keepdims=True)
        raw_y_abs_err = tf.abs(raw_axis_err[:, 1:2])
        y_over_threshold_penalty = tf.square(tf.nn.relu(raw_y_abs_err - distance_delta))

        if REGRESSION_LOSS_MODE != "distance_primary":
            base_loss = elementwise_huber_loss(y_true, y_pred) * weights
            distance_huber = tf.where(
                raw_distance <= distance_delta,
                0.5 * tf.square(raw_distance),
                distance_delta * (raw_distance - 0.5 * distance_delta),
            )
            over_threshold_penalty = tf.square(tf.nn.relu(raw_distance - distance_delta))
            return (
                base_loss
                + EUCLIDEAN_DISTANCE_LOSS_WEIGHT * distance_huber
                + POSITION_THRESHOLD_LOSS_WEIGHT * over_threshold_penalty
                + AXIS_FINE_THRESHOLD_LOSS_WEIGHT * tf.square(tf.nn.relu(raw_abs_axis_err - axis_delta)) * weights
                + Y_AXIS_THRESHOLD_LOSS_WEIGHT * y_over_threshold_penalty
            )

        sample_loss = (
            EUCLIDEAN_DISTANCE_LOSS_WEIGHT * raw_distance
            + POSITION_THRESHOLD_LOSS_WEIGHT * distance_margin
            + DISTANCE_PRIMARY_AXIS_LOSS_WEIGHT * axis_stabilizer
            + AXIS_FINE_THRESHOLD_LOSS_WEIGHT * fine_axis_penalty
            + Y_AXIS_THRESHOLD_LOSS_WEIGHT * y_over_threshold_penalty
        )
        return tf.repeat(sample_loss, repeats=len(TARGET_COLS), axis=-1)

    position_aware_regression_loss.__name__ = "position_aware_regression_loss"
    return position_aware_regression_loss


KERAS_INCOMPATIBLE_CONFIG_KEYS = {
    "quantization_config",
    "renorm",
    "renorm_clipping",
    "renorm_momentum",
}


def remove_keras_incompatible_config(value):
    if isinstance(value, dict):
        return {
            key: remove_keras_incompatible_config(child)
            for key, child in value.items()
            if key not in KERAS_INCOMPATIBLE_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [remove_keras_incompatible_config(item) for item in value]
    return value


def load_keras_model_compatible(model_path: str, custom_objects: dict = None, compile: bool = False):
    try:
        return load_model(model_path, custom_objects=custom_objects, compile=compile)
    except (ValueError, TypeError) as exc:
        if "quantization_config" not in str(exc):
            raise

        print("Detected incompatible Keras layer config, loading sanitized model copy...")
        with zipfile.ZipFile(model_path, "r") as src:
            config = json.loads(src.read("config.json").decode("utf-8"))
            config = remove_keras_incompatible_config(config)

            fd, sanitized_path = tempfile.mkstemp(suffix=".keras")
            os.close(fd)
            with zipfile.ZipFile(sanitized_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    if info.filename == "config.json":
                        dst.writestr(info, json.dumps(config).encode("utf-8"))
                    else:
                        dst.writestr(info, src.read(info.filename))

        try:
            return load_model(sanitized_path, custom_objects=custom_objects, compile=compile)
        finally:
            try:
                os.remove(sanitized_path)
            except OSError:
                pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_path_name(value) -> str:
    name = os.path.splitext(os.path.basename(str(value)))[0].strip()
    safe_chars = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe_name = "".join(safe_chars).strip("._")
    return safe_name or "unknown_source"


def machining_process_label(index: int) -> str:
    return f"加工过程{index + 1}"


def source_display_label(source_name: str, fallback_index: int = 0) -> str:
    source_index = IMPORTED_SOURCE_INDEX_BY_NAME.get(str(source_name), fallback_index)
    return machining_process_label(source_index)


def save_pickle_artifact(data: dict, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_dir(out_dir)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle_artifact(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Saved preprocessing artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_output_path(explicit_path: str, file_name: str) -> str:
    path = (explicit_path or "").strip()
    if path:
        return path
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(OUTPUT_DIR, file_name)


def infer_artifact_path_from_model(model_path: str) -> str:
    model_dir = os.path.dirname(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    candidate_stems = [
        model_stem.replace("_model_", "_artifacts_", 1),
        model_stem.replace("model", "artifacts", 1),
    ]
    for stem in unique_preserve_order(candidate_stems):
        candidate = os.path.join(model_dir, f"{stem}.pkl")
        if os.path.isfile(candidate):
            return candidate
    return ""


def open_file_with_default_app(path: str) -> None:
    try:
        if os.name == "nt":
            os.startfile(path)
        else:
            import subprocess

            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print(f"Failed to open plot {path}: {exc}")


def open_prediction_plots(out_dir: str, source_names: List[str]) -> None:
    if not OPEN_PLOTS_AFTER_RUN:
        return

    plot_paths: List[str] = []
    for source_name in source_names:
        safe_source = safe_path_name(source_name)
        source_plot_dir = os.path.join(out_dir, "test_by_model", safe_source)
        for axis in ["x", "y", "z"]:
            plot_paths.append(os.path.join(source_plot_dir, f"test_hybrid_v10_{axis}.png"))

        distance_plot_dir = os.path.join(out_dir, "position_distance_bars", safe_source)
        plot_paths.append(
            os.path.join(
                distance_plot_dir,
                f"test_hybrid_v10_{safe_source}_position_distance_bar.png",
            )
        )

    opened = 0
    for plot_path in unique_preserve_order(plot_paths):
        if os.path.isfile(plot_path):
            open_file_with_default_app(plot_path)
            opened += 1

    if opened:
        print(f"Opened {opened} prediction plot(s).")
    else:
        print("No prediction plot files were found to open.")


def validate_required_columns(df: pd.DataFrame, required_cols: List[str], table_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{table_name} is missing required saved feature columns: {missing_cols}")


def assign_float32_columns(df: pd.DataFrame, cols: List[str], values: np.ndarray) -> None:
    df.loc[:, cols] = pd.DataFrame(
        np.asarray(values, dtype=np.float32),
        index=df.index,
        columns=cols,
    )


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def to_float_or_zero(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def flatten_numeric_dict(data, prefix: str) -> dict:
    out = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}_{key}"
            if isinstance(value, dict):
                out.update(flatten_numeric_dict(value, child_prefix))
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    item_prefix = f"{child_prefix}_{idx}"
                    if isinstance(item, dict):
                        out.update(flatten_numeric_dict(item, item_prefix))
                    else:
                        out[item_prefix] = to_float_or_zero(item)
            else:
                out[child_prefix] = to_float_or_zero(value)
    return out


def natural_path_sort_key(path: str):
    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def collect_json_paths(json_dir: str, fallback_json_path: str = "") -> List[str]:
    json_paths: List[str] = []

    if json_dir and os.path.isdir(json_dir):
        json_paths.extend(sorted(glob(os.path.join(json_dir, "*.json")), key=natural_path_sort_key))

    if not json_paths and fallback_json_path and os.path.isfile(fallback_json_path):
        json_paths.append(fallback_json_path)

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found. Checked directory: {json_dir}, fallback file: {fallback_json_path}"
        )

    return json_paths


def json_mapped_file_index_span(json_path: str) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    max_index = -1
    for item in raw.get("point_to_frame_ai_block_avg", []):
        try:
            max_index = max(max_index, int(item.get("mapped_file_index", -1)))
        except (TypeError, ValueError):
            continue
    return max_index + 1


def select_json_paths_for_prediction(json_paths: List[str]) -> List[str]:
    global IMPORTED_SOURCE_INDEX_BY_NAME, PREDICT_SOURCE_INDEX, PREDICT_SOURCE_NAME
    global SELECTED_FILE_INDEX_OFFSET, SELECTED_JSON_PATHS_FOR_OUTPUT

    IMPORTED_SOURCE_INDEX_BY_NAME = {
        os.path.basename(path): idx
        for idx, path in enumerate(json_paths)
    }
    SELECTED_FILE_INDEX_OFFSET = 0

    selected_source_name = PREDICT_SOURCE_NAME.strip() if isinstance(PREDICT_SOURCE_NAME, str) else ""
    if selected_source_name:
        for path in json_paths:
            if os.path.basename(path) == selected_source_name:
                SELECTED_JSON_PATHS_FOR_OUTPUT = [path]
                return json_paths
        raise ValueError(
            f"PREDICT_SOURCE_NAME={selected_source_name!r} was not found. "
            f"Available source names: {[os.path.basename(path) for path in json_paths]}"
        )

    if PREDICT_SOURCE_INDEX is not None:
        if PREDICT_SOURCE_INDEX < 0 or PREDICT_SOURCE_INDEX >= len(json_paths):
            raise ValueError(
                f"PREDICT_SOURCE_INDEX={PREDICT_SOURCE_INDEX} is out of range. "
                f"Available range is 0-{len(json_paths) - 1}."
            )
        selected_path = json_paths[PREDICT_SOURCE_INDEX]
        PREDICT_SOURCE_NAME = os.path.basename(selected_path)
        PREDICT_SOURCE_INDEX = None
        SELECTED_JSON_PATHS_FOR_OUTPUT = [selected_path]
        return json_paths

    SELECTED_JSON_PATHS_FOR_OUTPUT = json_paths
    return json_paths


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_multioutput(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    axis_names = ["X", "Y", "Z"]
    print(f"\n{name}")
    for i, axis in enumerate(axis_names):
        cur_rmse = rmse(y_true[:, i], y_pred[:, i])
        cur_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        within_002_ratio = np.mean(np.abs(y_true[:, i] - y_pred[:, i]) < ERR_THRESHOLD)
        print(
            f"{axis}: RMSE={cur_rmse:.6f}, MAE={cur_mae:.6f}, "
            f"|err|<{ERR_THRESHOLD:.3f} Ratio={within_002_ratio:.6f}"
        )


def actual_position_distance(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    missing_pose_cols = [col for col in POSE_COLS if col not in meta_df.columns]
    if missing_pose_cols:
        raise ValueError(f"meta_df is missing pose columns needed for actual-position distance: {missing_pose_cols}")
    if len(meta_df) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(
            "meta_df, y_true, and y_pred must have the same length for actual-position distance."
        )

    pose_xyz = meta_df[POSE_COLS].to_numpy(dtype=np.float32)
    true_position = pose_xyz + y_true
    pred_position = pose_xyz + y_pred
    return np.linalg.norm(pred_position - true_position, axis=1)


def print_actual_position_distance_summary(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
) -> None:
    if len(meta_df) == 0 or len(meta_df) != len(y_true) or any(col not in meta_df.columns for col in POSE_COLS):
        return

    distances = actual_position_distance(meta_df, y_true, y_pred)
    within_ratio = float(np.mean(distances < POSITION_DISTANCE_THRESHOLD))
    print(
        f"{name} Actual Position Distance: "
        f"mean={np.mean(distances):.6f}, median={np.median(distances):.6f}, "
        f"max={np.max(distances):.6f}, distance<{POSITION_DISTANCE_THRESHOLD:.3f} Ratio={within_ratio:.6f}"
    )


def actual_position_distance_ratio(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(meta_df) == 0 or len(meta_df) != len(y_true) or any(col not in meta_df.columns for col in POSE_COLS):
        return 0.0
    distances = actual_position_distance(meta_df, y_true, y_pred)
    return float(np.mean(distances < POSITION_DISTANCE_THRESHOLD))


class BestPositionDistanceRatioCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        val_inputs: dict,
        val_y_scaled: np.ndarray,
        val_meta_df: pd.DataFrame,
        target_scaler: StandardScaler,
    ):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_y_true = target_scaler.inverse_transform(val_y_scaled)
        self.val_meta_df = val_meta_df
        self.target_scaler = target_scaler
        self.best_ratio = -np.inf
        self.best_epoch = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred_outputs = self.model.predict(self.val_inputs, verbose=0)
        pred_scaled = pred_outputs[0] if isinstance(pred_outputs, (list, tuple)) else pred_outputs
        pred = self.target_scaler.inverse_transform(pred_scaled)
        ratio = actual_position_distance_ratio(self.val_meta_df, self.val_y_true, pred)
        logs["val_position_distance_ratio"] = ratio
        if ratio > self.best_ratio:
            self.best_ratio = ratio
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                f"Restored best val_position_distance_ratio weights: "
                f"epoch={self.best_epoch}, ratio={self.best_ratio:.6f}"
            )


def print_source_ratio_summary(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    if "source_file_name" not in meta_df.columns or len(meta_df) != len(y_true):
        return

    axis_names = ["X", "Y", "Z"]
    print(f"\n{name} Ratio by machining process")
    for source_idx, source_name in enumerate(meta_df["source_file_name"].drop_duplicates()):
        mask = (meta_df["source_file_name"] == source_name).to_numpy()
        if not np.any(mask):
            continue

        source_true = y_true[mask]
        source_pred = y_pred[mask]
        axis_ratios = [
            float(np.mean(np.abs(source_true[:, i] - source_pred[:, i]) < ERR_THRESHOLD))
            for i in range(len(axis_names))
        ]
        overall_ratio = float(np.mean(np.abs(source_true - source_pred) < ERR_THRESHOLD))
        axis_text = ", ".join(
            f"{axis}={ratio:.6f}"
            for axis, ratio in zip(axis_names, axis_ratios)
        )
        display_name = source_display_label(source_name, source_idx)
        print(
            f"  {display_name}: n={int(mask.sum())}, "
            f"overall={overall_ratio:.6f}, {axis_text}"
        )


def print_test_point_prediction_details(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    if len(meta_df) != len(y_true) or len(y_true) != len(y_pred):
        return

    source_labels = {}
    if "source_file_name" in meta_df.columns:
        for source_idx, source_name in enumerate(meta_df["source_file_name"].drop_duplicates()):
            source_labels[source_name] = source_display_label(source_name, source_idx)

    abs_err = np.abs(y_pred - y_true)
    has_pose = all(col in meta_df.columns for col in POSE_COLS)
    if has_pose:
        pose_xyz = meta_df[POSE_COLS].to_numpy(dtype=np.float32)
        true_position = pose_xyz + y_true
        pred_position = pose_xyz + y_pred
        position_distance = np.linalg.norm(pred_position - true_position, axis=1)
    else:
        true_position = None
        pred_position = None
        position_distance = np.linalg.norm(abs_err, axis=1)

    print("逐点预测结果:")
    for idx in range(len(meta_df)):
        row = meta_df.iloc[idx]
        source_name = row.get("source_file_name", "")
        source_label = source_labels.get(source_name, str(source_name) if source_name else "ALL")
        nc_line = row.get("mapped_nc_line", "")
        point_index = row.get("point_index", "")
        line = (
            f"第{idx + 1:04d}点 | {source_label} | NC行={nc_line} | 点号={point_index} | "
            f"三轴绝对误差=({abs_err[idx, 0]:.6f}, {abs_err[idx, 1]:.6f}, {abs_err[idx, 2]:.6f}) | "
            f"实际位置欧氏距离={position_distance[idx]:.6f}"
        )
        if has_pose:
            line += (
                f" | 真实位置=({true_position[idx, 0]:.6f}, {true_position[idx, 1]:.6f}, {true_position[idx, 2]:.6f})"
                f" | 预测位置=({pred_position[idx, 0]:.6f}, {pred_position[idx, 1]:.6f}, {pred_position[idx, 2]:.6f})"
            )
        print(line)


def save_source_diagnostic_summary(
    pred_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    out_path: str,
) -> None:
    if "source_file_name" not in pred_df.columns or "actual_position_distance" not in pred_df.columns:
        return
    if len(pred_df) != len(diag_df):
        return

    rows = []
    axis_cols = ["x", "y", "z"]
    failure_mask = pred_df["actual_position_distance"].to_numpy() >= POSITION_DISTANCE_THRESHOLD
    dominant_axis = pred_df[[f"abs_err_{axis}" for axis in axis_cols]].idxmax(axis=1).str.replace("abs_err_", "", regex=False)

    for source_name, source_indices in pred_df.groupby("source_file_name", sort=False).groups.items():
        idx = np.asarray(list(source_indices), dtype=int)
        source_pred = pred_df.iloc[idx]
        source_diag = diag_df.iloc[idx]
        source_fail_mask = failure_mask[idx]
        total = len(source_pred)
        fail = int(source_fail_mask.sum())
        row = {
            "source_file_name": source_name,
            "total": total,
            "fail": fail,
            "fail_rate": fail / max(total, 1),
            "pass_rate": 1.0 - fail / max(total, 1),
            "distance_mean": float(source_pred["actual_position_distance"].mean()),
            "distance_median": float(source_pred["actual_position_distance"].median()),
            "distance_max": float(source_pred["actual_position_distance"].max()),
        }

        for axis in axis_cols:
            signed_residual = source_pred[f"true_{axis}"] - source_pred[f"hybrid_pred_{axis}"]
            abs_err = source_pred[f"abs_err_{axis}"]
            row[f"signed_residual_{axis}_mean"] = float(signed_residual.mean())
            row[f"signed_residual_{axis}_median"] = float(signed_residual.median())
            row[f"abs_err_{axis}_mean"] = float(abs_err.mean())
            row[f"abs_err_{axis}_median"] = float(abs_err.median())
            row[f"abs_err_{axis}_max"] = float(abs_err.max())
            row[f"fine_pass_{axis}_rate"] = float((abs_err < AXIS_FINE_THRESHOLD).mean())

        source_dominant = dominant_axis.iloc[idx]
        for axis in axis_cols:
            row[f"dominant_{axis}_count"] = int((source_dominant == axis).sum())

        for col in [
            "pose_y",
            "pose_r_xyz",
            "feed_speed",
            "spindle_speed",
            "temp_median_mean",
            "temp_median_range",
            "temp_ambient_c",
            "temp_spindle_ambient_delta",
        ]:
            if col not in source_diag.columns:
                continue
            values = pd.to_numeric(source_diag[col], errors="coerce")
            finite_values = values[np.isfinite(values)]
            row[f"{col}_mean"] = float(finite_values.mean()) if len(finite_values) else np.nan
            row[f"{col}_median"] = float(finite_values.median()) if len(finite_values) else np.nan
            if len(finite_values) >= 4 and finite_values.nunique() > 1:
                low_threshold = float(finite_values.quantile(0.25))
                high_threshold = float(finite_values.quantile(0.75))
                low_mask = values.to_numpy(dtype=np.float32) <= low_threshold
                high_mask = values.to_numpy(dtype=np.float32) >= high_threshold
                row[f"{col}_low_threshold"] = low_threshold
                row[f"{col}_low_fail_rate"] = float(source_fail_mask[low_mask].mean()) if np.any(low_mask) else np.nan
                row[f"{col}_high_threshold"] = high_threshold
                row[f"{col}_high_fail_rate"] = float(source_fail_mask[high_mask].mean()) if np.any(high_mask) else np.nan

        rows.append(row)

    diagnostic_df = pd.DataFrame(rows).sort_values(["fail_rate", "fail"], ascending=[False, False])
    diagnostic_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def add_prediction_diagnostic_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if "actual_position_distance" in out.columns:
        bins = [0.0, 0.02, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08, 0.12, np.inf]
        labels = ["<0.02", "0.02-0.035", "0.035-0.04", "0.04-0.045", "0.045-0.05", "0.05-0.06", "0.06-0.08", "0.08-0.12", ">=0.12"]
        out["distance_bucket"] = pd.cut(out["actual_position_distance"], bins=bins, labels=labels, right=False)
        out["position_fail"] = out["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD
    axis_cols = [f"abs_err_{axis}" for axis in ["x", "y", "z"]]
    if all(col in out.columns for col in axis_cols):
        out["dominant_axis"] = out[axis_cols].idxmax(axis=1).str.replace("abs_err_", "", regex=False)
    for axis in ["x", "y", "z"]:
        true_col = f"true_{axis}"
        pred_col = f"hybrid_pred_{axis}"
        if true_col in out.columns and pred_col in out.columns:
            out[f"signed_residual_{axis}"] = out[pred_col] - out[true_col]
    return out


def candidate_diagnostic_feature_cols(diag_df: pd.DataFrame) -> List[str]:
    preferred = [
        "pose_x", "pose_y", "pose_z", "pose_r_xy", "pose_r_xyz",
        "feed_speed", "spindle_speed",
        "temp_median_mean", "temp_median_range", "temp_std_mean",
        "temp_ch3_median_c", "temp_ch4_median_c", "temp_ch5_median_c", "temp_ch6_median_c",
        "temp_spindle_mean", "temp_spindle_range", "temp_spindle_std_mean",
        "temp_ambient_c", "temp_spindle_ambient_delta",
        "temp_ch3_ambient_delta", "temp_ch4_ambient_delta", "temp_ch6_ambient_delta",
        "source_line_progress", "source_frame_progress",
        "source_line_progress_centered", "source_frame_progress_centered",
        "source_line_frame_progress_diff",
        "progress_temp_interact", "progress_motion_acc_interact",
        "temp_median_mean_rise_from_start", "temp_median_mean_expanding_mean",
        "temp_median_mean_expanding_max", "temp_median_mean_expanding_delta",
        "temp_spindle_ambient_delta_rise_from_start",
        "temp_spindle_ambient_delta_expanding_mean",
        "temp_spindle_ambient_delta_expanding_max",
        "temp_spindle_ambient_delta_expanding_delta",
        "temp_ch3_ambient_delta_rise_from_start",
        "temp_ch4_ambient_delta_rise_from_start",
        "temp_ch6_ambient_delta_rise_from_start",
        "progress_temp_rise_interact", "progress_temp_expanding_mean_interact",
        "progress_spindle_ambient_delta_rise_interact",
        "progress_spindle_ambient_delta_expanding_mean_interact",
        "progress_ch3_ambient_delta_rise_interact",
        "progress_ch4_ambient_delta_rise_interact",
        "progress_ch6_ambient_delta_rise_interact",
    ]
    pattern_cols = [
        col for col in diag_df.columns
        if (
            col.startswith(("motion_", "vib_", "temp_"))
            or col.startswith(("source_line_progress", "source_frame_progress", "progress_"))
            or col.endswith(("_rise_from_start", "_expanding_mean", "_expanding_max", "_expanding_delta"))
            or "_roll" in col
        )
        and pd.api.types.is_numeric_dtype(diag_df[col])
    ]
    return unique_preserve_order([col for col in preferred + sorted(pattern_cols) if col in diag_df.columns])


def save_near_boundary_diagnostic(pred_df: pd.DataFrame, diag_df: pd.DataFrame, out_path: str) -> None:
    if len(pred_df) != len(diag_df) or "actual_position_distance" not in pred_df.columns:
        return
    enriched = add_prediction_diagnostic_columns(pred_df)
    mask = (
        (enriched["actual_position_distance"] >= 0.035)
        & (enriched["actual_position_distance"] < 0.06)
    )
    rows = enriched.loc[mask].copy()
    if rows.empty:
        rows.to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    extra_cols = [
        "source_file_path",
        "original_mapped_file_index",
        "pose_x", "pose_y", "pose_z", "pose_r_xy", "pose_r_xyz",
        "feed_speed", "spindle_speed",
        "temp_median_mean", "temp_median_range", "temp_std_mean",
        "temp_spindle_mean", "temp_spindle_range", "temp_spindle_std_mean",
        "temp_ambient_c", "temp_spindle_ambient_delta",
    ]
    extra_cols.extend(candidate_diagnostic_feature_cols(diag_df))
    for col in unique_preserve_order(extra_cols):
        if col in diag_df.columns and col not in rows.columns:
            rows[col] = diag_df.loc[rows.index, col].to_numpy()

    sort_cols = [col for col in ["position_fail", "actual_position_distance", "source_file_name", "mapped_nc_line"] if col in rows.columns]
    ascending = [False if col == "position_fail" else True for col in sort_cols]
    if "actual_position_distance" in sort_cols:
        ascending[sort_cols.index("actual_position_distance")] = False
    rows = rows.sort_values(sort_cols, ascending=ascending)
    rows.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_failure_feature_bucket_diagnostic(pred_df: pd.DataFrame, diag_df: pd.DataFrame, out_path: str) -> None:
    if len(pred_df) != len(diag_df) or "actual_position_distance" not in pred_df.columns:
        return
    enriched = add_prediction_diagnostic_columns(pred_df)
    feature_cols = candidate_diagnostic_feature_cols(diag_df)
    rows = []
    source_values = ["ALL"]
    if "source_file_name" in enriched.columns:
        source_values.extend(enriched["source_file_name"].dropna().drop_duplicates().tolist())

    for source_name in source_values:
        if source_name == "ALL" or "source_file_name" not in enriched.columns:
            source_mask = np.ones(len(enriched), dtype=bool)
        else:
            source_mask = (enriched["source_file_name"] == source_name).to_numpy()
        if int(source_mask.sum()) < 8:
            continue

        for feature in feature_cols:
            values = pd.to_numeric(diag_df[feature], errors="coerce").to_numpy(dtype=np.float32)
            finite_mask = source_mask & np.isfinite(values)
            if int(finite_mask.sum()) < 8 or len(np.unique(values[finite_mask])) < 3:
                continue
            edges = np.unique(np.quantile(values[finite_mask], [0.0, 0.25, 0.5, 0.75, 1.0]))
            if len(edges) < 3:
                continue
            buckets = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
            for bucket in buckets.categories:
                bucket_mask = finite_mask & (buckets == bucket)
                total = int(bucket_mask.sum())
                if total == 0:
                    continue
                subset = enriched.loc[bucket_mask]
                fail_mask = subset["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD
                near_fail_mask = (
                    (subset["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD)
                    & (subset["actual_position_distance"] < 0.06)
                )
                row = {
                    "source_file_name": source_name,
                    "feature": feature,
                    "bucket": str(bucket),
                    "total": total,
                    "fail": int(fail_mask.sum()),
                    "fail_rate": float(fail_mask.mean()),
                    "near_fail_0.04_0.06": int(near_fail_mask.sum()),
                    "near_fail_rate_0.04_0.06": float(near_fail_mask.mean()),
                    "distance_mean": float(subset["actual_position_distance"].mean()),
                    "distance_median": float(subset["actual_position_distance"].median()),
                    "feature_mean": float(np.mean(values[bucket_mask])),
                }
                if "dominant_axis" in subset.columns:
                    for axis in ["x", "y", "z"]:
                        row[f"dominant_{axis}_fail"] = int(((subset["dominant_axis"] == axis) & fail_mask).sum())
                rows.append(row)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["fail_rate", "fail", "near_fail_rate_0.04_0.06"], ascending=[False, False, False])
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_failure_line_window_diagnostic(pred_df: pd.DataFrame, out_path: str) -> None:
    if "actual_position_distance" not in pred_df.columns:
        return
    enriched = add_prediction_diagnostic_columns(pred_df)
    rows = []
    group_cols = ["source_file_name"] if "source_file_name" in enriched.columns else []
    line_col = "mapped_nc_line" if "mapped_nc_line" in enriched.columns else None
    frame_col = "mapped_frame_num" if "mapped_frame_num" in enriched.columns else None
    if line_col is None:
        enriched.to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    for group_key, group in enriched.groupby(group_cols, sort=False) if group_cols else [("ALL", enriched)]:
        values = pd.to_numeric(group[line_col], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 8 or finite.nunique() < 3:
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 9)))
        buckets = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
        for bucket in buckets.cat.categories:
            subset = group.loc[buckets == bucket]
            if subset.empty:
                continue
            fail_mask = subset["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD
            rows.append({
                "source_file_name": group_key if isinstance(group_key, str) else group_key[0],
                "line_bucket": str(bucket),
                "line_min": float(subset[line_col].min()),
                "line_max": float(subset[line_col].max()),
                "frame_min": float(subset[frame_col].min()) if frame_col in subset.columns else np.nan,
                "frame_max": float(subset[frame_col].max()) if frame_col in subset.columns else np.nan,
                "total": int(len(subset)),
                "fail": int(fail_mask.sum()),
                "fail_rate": float(fail_mask.mean()),
                "near_fail_0.04_0.06": int(((subset["actual_position_distance"] >= 0.04) & (subset["actual_position_distance"] < 0.06)).sum()),
                "distance_mean": float(subset["actual_position_distance"].mean()),
                "distance_max": float(subset["actual_position_distance"].max()),
            })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["fail_rate", "fail"], ascending=[False, False])
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_phase_progress_diagnostic(pred_df: pd.DataFrame, diag_df: pd.DataFrame, out_path: str) -> None:
    if len(pred_df) != len(diag_df) or "actual_position_distance" not in pred_df.columns:
        return
    enriched = add_prediction_diagnostic_columns(pred_df)
    rows = []
    candidate_cols = [
        "source_line_progress",
        "source_frame_progress",
        "temp_median_mean_rise_from_start",
        "temp_median_mean_expanding_delta",
        "progress_temp_rise_interact",
    ]
    source_values = ["ALL"]
    if "source_file_name" in enriched.columns:
        source_values.extend(enriched["source_file_name"].dropna().drop_duplicates().tolist())

    for source_name in source_values:
        if source_name == "ALL" or "source_file_name" not in enriched.columns:
            source_mask = np.ones(len(enriched), dtype=bool)
        else:
            source_mask = (enriched["source_file_name"] == source_name).to_numpy()

        for col in candidate_cols:
            if col in diag_df.columns:
                values = pd.to_numeric(diag_df[col], errors="coerce").to_numpy(dtype=np.float32)
            elif col in enriched.columns:
                values = pd.to_numeric(enriched[col], errors="coerce").to_numpy(dtype=np.float32)
            else:
                continue

            finite_mask = source_mask & np.isfinite(values)
            if int(finite_mask.sum()) < 8 or len(np.unique(values[finite_mask])) < 3:
                continue
            if col.endswith("_progress"):
                edges = np.linspace(0.0, 1.0, 6)
            else:
                edges = np.unique(np.quantile(values[finite_mask], np.linspace(0.0, 1.0, 6)))
            if len(edges) < 3:
                continue
            buckets = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
            for bucket in buckets.categories:
                bucket_mask = finite_mask & (buckets == bucket)
                if not np.any(bucket_mask):
                    continue
                subset = enriched.loc[bucket_mask]
                fail_mask = subset["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD
                near_fail_mask = (
                    (subset["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD)
                    & (subset["actual_position_distance"] < 0.06)
                )
                rows.append({
                    "source_file_name": source_name,
                    "feature": col,
                    "bucket": str(bucket),
                    "total": int(bucket_mask.sum()),
                    "fail": int(fail_mask.sum()),
                    "fail_rate": float(fail_mask.mean()),
                    "near_fail_0.04_0.06": int(near_fail_mask.sum()),
                    "near_fail_rate_0.04_0.06": float(near_fail_mask.mean()),
                    "distance_mean": float(subset["actual_position_distance"].mean()),
                    "distance_median": float(subset["actual_position_distance"].median()),
                    "distance_p90": float(subset["actual_position_distance"].quantile(0.90)),
                    "distance_max": float(subset["actual_position_distance"].max()),
                    "feature_mean": float(np.mean(values[bucket_mask])),
                })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["source_file_name", "feature", "bucket"])
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def fit_error_class_edges(target_scaler: StandardScaler, class_count: int = ERROR_CLASS_COUNT) -> np.ndarray:
    del class_count
    fixed_raw_edges = np.asarray(ERROR_CLASS_EDGES, dtype=np.float32)
    standardized_edges = []
    for axis_idx in range(len(TARGET_COLS)):
        axis_edges = (fixed_raw_edges - target_scaler.mean_[axis_idx]) / target_scaler.scale_[axis_idx]
        standardized_edges.append(axis_edges.astype(np.float32))
    return np.asarray(standardized_edges, dtype=np.float32)


def build_error_class_labels(y_scaled: np.ndarray, class_edges: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(y_scaled, dtype=np.int32)
    for axis_idx in range(y_scaled.shape[1]):
        labels[:, axis_idx] = np.digitize(y_scaled[:, axis_idx], class_edges[axis_idx]).astype(np.int32)
    return labels


def print_error_class_edges(class_edges: np.ndarray) -> None:
    axis_names = ["X", "Y", "Z"]
    print("\nSigned error class edges converted to standardized target space:")
    for axis_idx, axis in enumerate(axis_names):
        edge_text = ", ".join(f"{edge:.6f}" for edge in class_edges[axis_idx])
        print(f"{axis}: {edge_text}")


def build_high_value_regression_weights(class_labels: np.ndarray) -> np.ndarray:
    high_class_start = max(ERROR_CLASS_COUNT - HIGH_VALUE_TOP_CLASSES, 0)
    high_mask = class_labels >= high_class_start
    weights = np.ones(class_labels.shape, dtype=np.float32)
    weights[high_mask] += HIGH_VALUE_REG_LOSS_WEIGHT
    return weights


def print_high_value_weight_summary(class_labels: np.ndarray, weights: np.ndarray, name: str) -> None:
    high_class_start = max(ERROR_CLASS_COUNT - HIGH_VALUE_TOP_CLASSES, 0)
    high_mask = class_labels >= high_class_start
    print(
        f"{name} high-value regression weights: "
        f"high_class_start={high_class_start}, high_axis_targets={int(high_mask.sum())}/{high_mask.size}, "
        f"weight_min={float(np.min(weights)):.3f}, weight_max={float(np.max(weights)):.3f}"
    )


def build_near_threshold_regression_weights(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> np.ndarray:
    distances = np.linalg.norm(np.asarray(y_true, dtype=np.float32) - np.asarray(y_pred, dtype=np.float32), axis=1)
    sample_weights = np.full(distances.shape, NEAR_THRESHOLD_FAR_WEIGHT, dtype=np.float32)

    easy_mask = distances < NEAR_THRESHOLD_EASY_DISTANCE
    stable_mask = (distances >= NEAR_THRESHOLD_EASY_DISTANCE) & (distances < NEAR_THRESHOLD_PROTECT_DISTANCE)
    protect_mask = (distances >= NEAR_THRESHOLD_PROTECT_DISTANCE) & (distances < POSITION_DISTANCE_THRESHOLD)
    rescue_mask = (distances >= POSITION_DISTANCE_THRESHOLD) & (distances < NEAR_THRESHOLD_RESCUE_DISTANCE)
    far_mask = distances >= NEAR_THRESHOLD_RESCUE_DISTANCE

    sample_weights[easy_mask] = NEAR_THRESHOLD_EASY_WEIGHT
    sample_weights[stable_mask] = NEAR_THRESHOLD_STABLE_WEIGHT
    sample_weights[protect_mask] = NEAR_THRESHOLD_PROTECT_WEIGHT
    sample_weights[rescue_mask] = NEAR_THRESHOLD_RESCUE_WEIGHT
    sample_weights[far_mask] = NEAR_THRESHOLD_FAR_WEIGHT

    print(
        f"{name} near-threshold weights: "
        f"easy<{NEAR_THRESHOLD_EASY_DISTANCE:.3f}={int(easy_mask.sum())}, "
        f"stable={int(stable_mask.sum())}, "
        f"protect={int(protect_mask.sum())}, "
        f"rescue={int(rescue_mask.sum())}, "
        f"far>={NEAR_THRESHOLD_RESCUE_DISTANCE:.3f}={int(far_mask.sum())}, "
        f"weight_mean={float(np.mean(sample_weights)):.3f}, "
        f"weight_min={float(np.min(sample_weights)):.3f}, weight_max={float(np.max(sample_weights)):.3f}"
    )
    return np.repeat(sample_weights[:, None], len(TARGET_COLS), axis=1).astype(np.float32)


def build_auxiliary_targets(y_scaled: np.ndarray, target_scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    y_raw = target_scaler.inverse_transform(y_scaled)
    distance = np.linalg.norm(y_raw, axis=1, keepdims=True).astype(np.float32)
    risk = (distance >= POSITION_DISTANCE_THRESHOLD).astype(np.float32)
    return distance, risk


def has_auxiliary_heads() -> bool:
    return DISTANCE_AUX_LOSS_WEIGHT > 0 or RISK_AUX_LOSS_WEIGHT > 0


def make_model_targets(reg_y: np.ndarray, class_y: np.ndarray, distance_y: np.ndarray, risk_y: np.ndarray):
    if has_auxiliary_heads():
        return [reg_y, class_y, distance_y, risk_y]
    return [reg_y, class_y]


def make_model_sample_weights(reg_weights: np.ndarray, class_weights: np.ndarray, distance_y: np.ndarray, risk_y: np.ndarray):
    if has_auxiliary_heads():
        distance_weights = np.ones(distance_y.shape, dtype=np.float32)
        risk_weights = np.ones(risk_y.shape, dtype=np.float32)
        return [reg_weights, class_weights, distance_weights, risk_weights]
    return [reg_weights, class_weights]


def uses_condition_guided_moe() -> bool:
    return USE_MOE_OUTPUT_HEAD and USE_CONDITION_GUIDED_MOE


def make_model_inputs(static_X: np.ndarray, seq_X: np.ndarray, condition_X: np.ndarray = None) -> Dict[str, np.ndarray]:
    inputs = {"static_input": static_X, "seq_input": seq_X}
    if uses_condition_guided_moe():
        if condition_X is None:
            raise ValueError("condition_X is required when condition-guided MoE is enabled.")
        inputs["condition_input"] = condition_X
    return inputs


def split_model_predictions(pred_outputs):
    if not isinstance(pred_outputs, (list, tuple)):
        return pred_outputs, None, None, None
    regression_pred = pred_outputs[0]
    class_pred = pred_outputs[1] if len(pred_outputs) > 1 else None
    distance_pred = pred_outputs[2] if len(pred_outputs) > 2 else None
    risk_pred = pred_outputs[3] if len(pred_outputs) > 3 else None
    return regression_pred, class_pred, distance_pred, risk_pred


def add_error_class_columns(
    df: pd.DataFrame,
    class_edges: np.ndarray,
    target_values: np.ndarray = None,
) -> pd.DataFrame:
    out = df.copy()
    if target_values is None:
        target_values = out[TARGET_COLS].values
    class_values = build_error_class_labels(np.asarray(target_values, dtype=np.float32), class_edges)
    for idx, class_col in enumerate(ERROR_CLASS_COLS):
        out[class_col] = class_values[:, idx]
    return out


def evaluate_error_classification(
    y_true_scaled: np.ndarray,
    class_pred: np.ndarray,
    class_edges: np.ndarray,
    name: str,
) -> None:
    class_true = build_error_class_labels(y_true_scaled, class_edges)
    class_hat = np.argmax(class_pred, axis=-1)
    axis_names = ["X", "Y", "Z"]
    print(f"\n{name} Signed Error-Class Accuracy")
    for i, axis in enumerate(axis_names):
        acc = float(np.mean(class_hat[:, i] == class_true[:, i]))
        near_acc = float(np.mean(np.abs(class_hat[:, i] - class_true[:, i]) <= 1))
        class_mae = float(np.mean(np.abs(class_hat[:, i] - class_true[:, i])))
        print(f"{axis}: class_acc={acc:.6f}, near_class_acc={near_acc:.6f}, class_mae={class_mae:.6f}")


def print_error_bucket_summary(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    bucket_edges = [0.0, 0.01, 0.02, 0.05, np.inf]
    bucket_labels = ["<0.01", "0.01-0.02", "0.02-0.05", ">=0.05"]
    axis_names = ["X", "Y", "Z"]

    print(f"\n{name} Error Buckets")
    for i, axis in enumerate(axis_names):
        abs_err = np.abs(y_true[:, i] - y_pred[:, i])
        counts, _ = np.histogram(abs_err, bins=bucket_edges)
        total = max(len(abs_err), 1)
        summary = ", ".join(
            f"{label}: {count} ({count / total:.3f})"
            for label, count in zip(bucket_labels, counts)
        )
        print(f"{axis}: {summary}")


def print_grouped_ratio_summary(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    value_col: str,
    name: str,
    bins: int = 4,
) -> None:
    if value_col not in meta_df.columns or len(meta_df) == 0:
        return

    values = meta_df[value_col].to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < bins:
        return

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values[finite_mask], quantiles))
    if len(edges) <= 2:
        return

    bucket_ids = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
    axis_names = ["X", "Y", "Z"]
    print(f"\n{name} Ratio by {value_col}")
    for bucket in bucket_ids.categories:
        mask = bucket_ids == bucket
        if int(mask.sum()) == 0:
            continue
        ratios = []
        for i, axis in enumerate(axis_names):
            ratio = np.mean(np.abs(y_true[mask, i] - y_pred[mask, i]) < ERR_THRESHOLD)
            ratios.append(f"{axis}={ratio:.3f}")
        print(f"{bucket}: n={int(mask.sum())}, " + ", ".join(ratios))


def print_y_axis_diagnostic(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    if len(meta_df) == 0 or len(y_true) == 0 or len(y_pred) == 0:
        return

    y_signed_err = y_pred[:, 1] - y_true[:, 1]
    y_abs_err = np.abs(y_true[:, 1] - y_pred[:, 1])
    good_mask = y_abs_err < ERR_THRESHOLD
    bad_mask = ~good_mask

    print(f"\n{name} Y-Axis Diagnostic")
    print(
        f"good(|err_y|<{ERR_THRESHOLD:.3f})={int(good_mask.sum())}, "
        f"bad(|err_y|>={ERR_THRESHOLD:.3f})={int(bad_mask.sum())}"
    )
    print(
        f"signed_err(pred_y-true_y): mean={np.mean(y_signed_err):.6f}, "
        f"median={np.median(y_signed_err):.6f}"
    )
    print(
        f"good_signed_mean={np.mean(y_signed_err[good_mask]):.6f}, "
        f"bad_signed_mean={np.mean(y_signed_err[bad_mask]):.6f}"
    )

    candidate_cols = [
        "feed_speed",
        "spindle_speed",
        "pose_x",
        "pose_y",
        "pose_z",
        "pose_r_xy",
        "pose_r_xyz",
        "temp_median_mean",
        "temp_spindle_mean",
        "temp_ambient_c",
        "temp_spindle_ambient_delta",
    ]

    motion_cols = sorted([col for col in meta_df.columns if col.startswith("motion_")])[:4]
    vib_cols = sorted([col for col in meta_df.columns if col.startswith("vib_")])[:4]
    feature_cols = candidate_cols + motion_cols + vib_cols

    for col in feature_cols:
        if col not in meta_df.columns:
            continue
        values = pd.to_numeric(meta_df[col], errors="coerce").to_numpy(dtype=np.float32)
        good_vals = values[good_mask & np.isfinite(values)]
        bad_vals = values[bad_mask & np.isfinite(values)]
        if len(good_vals) == 0 or len(bad_vals) == 0:
            continue

        print(
            f"{col}: "
            f"good_mean={np.mean(good_vals):.6f}, bad_mean={np.mean(bad_vals):.6f}, "
            f"good_median={np.median(good_vals):.6f}, bad_median={np.median(bad_vals):.6f}"
        )


def fit_global_y_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    pred_y = y_pred[:, 1]
    true_y = y_true[:, 1]
    if len(pred_y) < 2 or np.allclose(pred_y, pred_y[0]):
        return 1.0, float(np.mean(true_y - pred_y)) if len(pred_y) > 0 else 0.0
    slope, intercept = np.polyfit(pred_y, true_y, 1)
    return float(slope), float(intercept)


def apply_global_y_calibrator(y_pred: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    corrected_pred = y_pred.copy()
    corrected_pred[:, 1] = slope * corrected_pred[:, 1] + intercept
    return corrected_pred


def fit_global_axis_calibrators(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    slopes = []
    intercepts = []
    for axis_idx in range(y_true.shape[1]):
        pred_axis = y_pred[:, axis_idx]
        true_axis = y_true[:, axis_idx]
        if len(pred_axis) < 2 or np.allclose(pred_axis, pred_axis[0]):
            slope = 1.0
            intercept = float(np.mean(true_axis - pred_axis)) if len(pred_axis) > 0 else 0.0
        else:
            slope, intercept = np.polyfit(pred_axis, true_axis, 1)
        slopes.append(float(slope))
        intercepts.append(float(intercept))
    return np.asarray(slopes, dtype=np.float32), np.asarray(intercepts, dtype=np.float32)


def apply_global_axis_calibrators(y_pred: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    corrected_pred = y_pred.copy()
    corrected_pred = corrected_pred * slopes.reshape(1, -1) + intercepts.reshape(1, -1)
    return corrected_pred


def within_threshold_ratio(y_true: np.ndarray, y_pred: np.ndarray, axis_idx: int = None) -> float:
    if axis_idx is None:
        return float(np.mean(np.abs(y_true - y_pred) < ERR_THRESHOLD))
    return float(np.mean(np.abs(y_true[:, axis_idx] - y_pred[:, axis_idx]) < ERR_THRESHOLD))


def fit_low_tail_offset_calibrator(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    col: str,
    quantile: float = 0.25,
    max_abs_offset: float = 0.02,
) -> Tuple[float, float]:
    if col not in meta_df.columns or len(meta_df) == 0:
        return 0.0, 0.0

    values = pd.to_numeric(meta_df[col], errors="coerce").to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) < 4:
        return 0.0, 0.0

    threshold = float(np.quantile(values[finite_mask], quantile))
    low_mask = finite_mask & (values <= threshold)
    if int(low_mask.sum()) == 0:
        return threshold, 0.0

    residual_y = y_true[:, 1] - y_pred[:, 1]
    offset = float(np.mean(residual_y[low_mask]))
    offset = float(np.clip(offset, -max_abs_offset, max_abs_offset))
    return threshold, offset


def apply_low_tail_offset_calibrator(
    meta_df: pd.DataFrame,
    y_pred: np.ndarray,
    col: str,
    threshold: float,
    offset: float,
) -> np.ndarray:
    corrected_pred = y_pred.copy()
    if col not in meta_df.columns or offset == 0.0:
        return corrected_pred

    values = pd.to_numeric(meta_df[col], errors="coerce").to_numpy(dtype=np.float32)
    low_mask = np.isfinite(values) & (values <= threshold)
    corrected_pred[low_mask, 1] = corrected_pred[low_mask, 1] + offset
    return corrected_pred


def fit_tail_axis_offset_calibrator(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    col: str,
    quantile: float,
    tail: str,
    max_abs_offset: float = 0.02,
) -> Tuple[float, np.ndarray]:
    if col not in meta_df.columns or len(meta_df) == 0:
        return 0.0, np.zeros(y_pred.shape[1], dtype=np.float32)

    values = pd.to_numeric(meta_df[col], errors="coerce").to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) < 4:
        return 0.0, np.zeros(y_pred.shape[1], dtype=np.float32)

    threshold = float(np.quantile(values[finite_mask], quantile))
    if tail == "high":
        mask = finite_mask & (values >= threshold)
    elif tail == "low":
        mask = finite_mask & (values <= threshold)
    else:
        raise ValueError(f"Unsupported tail: {tail}")

    if int(mask.sum()) == 0:
        return threshold, np.zeros(y_pred.shape[1], dtype=np.float32)

    offsets = np.mean(y_true[mask] - y_pred[mask], axis=0)
    offsets = np.clip(offsets, -max_abs_offset, max_abs_offset)
    return threshold, offsets.astype(np.float32)


def apply_tail_axis_offset_calibrator(
    meta_df: pd.DataFrame,
    y_pred: np.ndarray,
    col: str,
    threshold: float,
    tail: str,
    offsets: np.ndarray,
) -> np.ndarray:
    corrected_pred = y_pred.copy()
    if col not in meta_df.columns or np.allclose(offsets, 0.0):
        return corrected_pred

    values = pd.to_numeric(meta_df[col], errors="coerce").to_numpy(dtype=np.float32)
    if tail == "high":
        mask = np.isfinite(values) & (values >= threshold)
    elif tail == "low":
        mask = np.isfinite(values) & (values <= threshold)
    else:
        raise ValueError(f"Unsupported tail: {tail}")

    corrected_pred[mask] = corrected_pred[mask] + offsets.reshape(1, -1)
    return corrected_pred


def fit_group_axis_offset_calibrator(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    min_group_samples: int = 8,
    max_abs_offset: float = 0.03,
    shrinkage: float = 16.0,
    axis_idx: int = None,
) -> Dict[str, np.ndarray]:
    if group_col not in meta_df.columns or len(meta_df) != len(y_true):
        return {}

    offsets = {}
    residual = y_true - y_pred
    groups = meta_df[group_col].astype(str).to_numpy()
    for group_name in pd.unique(groups):
        mask = groups == group_name
        group_n = int(mask.sum())
        if group_n < min_group_samples:
            continue

        offset = np.mean(residual[mask], axis=0)
        offset = offset * (group_n / (group_n + shrinkage))
        if axis_idx is not None:
            axis_offset = np.zeros(y_pred.shape[1], dtype=np.float32)
            axis_offset[axis_idx] = offset[axis_idx]
            offset = axis_offset
        offset = np.clip(offset, -max_abs_offset, max_abs_offset)
        offsets[str(group_name)] = offset.astype(np.float32)
    return offsets


def apply_group_axis_offset_calibrator(
    meta_df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str,
    offsets: Dict[str, np.ndarray],
) -> np.ndarray:
    corrected_pred = y_pred.copy()
    if group_col not in meta_df.columns or not offsets:
        return corrected_pred

    groups = meta_df[group_col].astype(str).to_numpy()
    for group_name, offset in offsets.items():
        mask = groups == group_name
        if np.any(mask):
            corrected_pred[mask] = corrected_pred[mask] + offset.reshape(1, -1)
    return corrected_pred


def fit_group_region_axis_offset_calibrator(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    value_col: str,
    quantile: float,
    tail: str,
    min_region_samples: int = 5,
    max_abs_offset: float = 0.025,
    shrinkage: float = 12.0,
    axis_idx: int = None,
) -> Tuple[float, Dict[str, np.ndarray]]:
    if group_col not in meta_df.columns or value_col not in meta_df.columns or len(meta_df) != len(y_true):
        return 0.0, {}

    values = pd.to_numeric(meta_df[value_col], errors="coerce").to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) < min_region_samples:
        return 0.0, {}

    threshold = float(np.quantile(values[finite_mask], quantile))
    if tail == "high":
        region_mask = finite_mask & (values >= threshold)
    elif tail == "low":
        region_mask = finite_mask & (values <= threshold)
    else:
        raise ValueError(f"Unsupported tail: {tail}")

    residual = y_true - y_pred
    groups = meta_df[group_col].astype(str).to_numpy()
    offsets = {}
    for group_name in pd.unique(groups):
        mask = region_mask & (groups == group_name)
        group_n = int(mask.sum())
        if group_n < min_region_samples:
            continue

        offset = np.mean(residual[mask], axis=0)
        offset = offset * (group_n / (group_n + shrinkage))
        if axis_idx is not None:
            axis_offset = np.zeros(y_pred.shape[1], dtype=np.float32)
            axis_offset[axis_idx] = offset[axis_idx]
            offset = axis_offset
        offset = np.clip(offset, -max_abs_offset, max_abs_offset)
        offsets[str(group_name)] = offset.astype(np.float32)

    return threshold, offsets


def apply_group_region_axis_offset_calibrator(
    meta_df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str,
    value_col: str,
    threshold: float,
    tail: str,
    offsets: Dict[str, np.ndarray],
) -> np.ndarray:
    corrected_pred = y_pred.copy()
    if group_col not in meta_df.columns or value_col not in meta_df.columns or not offsets:
        return corrected_pred

    values = pd.to_numeric(meta_df[value_col], errors="coerce").to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(values)
    if tail == "high":
        region_mask = finite_mask & (values >= threshold)
    elif tail == "low":
        region_mask = finite_mask & (values <= threshold)
    else:
        raise ValueError(f"Unsupported tail: {tail}")

    groups = meta_df[group_col].astype(str).to_numpy()
    for group_name, offset in offsets.items():
        mask = region_mask & (groups == group_name)
        if np.any(mask):
            corrected_pred[mask] = corrected_pred[mask] + offset.reshape(1, -1)
    return corrected_pred


def build_sequences(X: np.ndarray, y: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(time_steps, len(X)):
        xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def build_grouped_sequences(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str],
                            group_cols: List[str], time_steps: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, meta_parts = [], [], []
    for _, group in df.groupby(group_cols, sort=False):
        group = group.reset_index(drop=True)
        if len(group) <= time_steps:
            continue
        X_group = group[feature_cols].values
        y_group = group[target_cols].values
        X_seq, y_seq = build_sequences(X_group, y_group, time_steps)
        xs.append(X_seq)
        ys.append(y_seq)
        meta_parts.append(group.iloc[time_steps:].copy())
    if not xs:
        return (
            np.empty((0, time_steps, len(feature_cols))),
            np.empty((0, len(target_cols))),
            df.iloc[0:0].copy(),
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), pd.concat(meta_parts, ignore_index=True)


def build_grouped_feature_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_cols: List[str],
    time_steps: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    xs, meta_parts = [], []
    for _, group in df.groupby(group_cols, sort=False):
        group = group.reset_index(drop=True)
        if len(group) < time_steps:
            continue
        X_group = group[feature_cols].values
        for i in range(time_steps - 1, len(group)):
            xs.append(X_group[i - time_steps + 1:i + 1])
        meta_parts.append(group.iloc[time_steps - 1:].copy())
    if not xs:
        return np.empty((0, time_steps, len(feature_cols))), df.iloc[0:0].copy()
    return np.asarray(xs), pd.concat(meta_parts, ignore_index=True)


def build_hybrid_samples(
    frame_seq_df: pd.DataFrame,
    frame_seq_X: np.ndarray,
    point_df: pd.DataFrame,
    static_cols: List[str],
    target_cols: List[str],
    class_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    meta_cols = FRAME_KEYS + ["point_index"]
    if "source_file_name" in point_df.columns:
        meta_cols.append("source_file_name")

    seq_map: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    for idx, row in frame_seq_df.reset_index(drop=True).iterrows():
        frame_key = tuple(int(row[key]) for key in FRAME_KEYS)
        seq_map[frame_key] = frame_seq_X[idx]

    static_parts, seq_parts, y_parts, class_parts, meta_parts = [], [], [], [], []
    for _, row in point_df.iterrows():
        frame_key = tuple(int(row[key]) for key in FRAME_KEYS)
        seq_value = seq_map.get(frame_key)
        if seq_value is None:
            continue
        static_parts.append(row[static_cols].to_numpy(dtype=np.float32))
        seq_parts.append(seq_value)
        y_parts.append(row[target_cols].to_numpy(dtype=np.float32))
        if class_cols is not None:
            class_parts.append(row[class_cols].to_numpy(dtype=np.int32))
        meta_parts.append(row[meta_cols].to_dict())

    if not static_parts:
        return (
            np.empty((0, len(static_cols)), dtype=np.float32),
            np.empty((0, frame_seq_X.shape[1], frame_seq_X.shape[2]), dtype=np.float32),
            np.empty((0, len(target_cols)), dtype=np.float32),
            np.empty((0, len(class_cols) if class_cols is not None else 0), dtype=np.int32),
            point_df.iloc[0:0][meta_cols].copy(),
        )

    return (
        np.asarray(static_parts, dtype=np.float32),
        np.asarray(seq_parts, dtype=np.float32),
        np.asarray(y_parts, dtype=np.float32),
        np.asarray(class_parts, dtype=np.int32) if class_cols is not None else np.empty((0, 0), dtype=np.int32),
        pd.DataFrame(meta_parts),
    )


def select_points_by_frames(point_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    return point_df.merge(frame_df[FRAME_KEYS].drop_duplicates(), on=FRAME_KEYS, how="inner")


def add_group_diff_features(df: pd.DataFrame, cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"d_{col}"] = out.groupby(group_cols)[col].diff().fillna(0.0)
    return out


def train_axis_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> List[GradientBoostingRegressor]:
    models = []
    for axis_idx in range(y_train.shape[1]):
        model = GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
        )
        if sample_weight is None:
            model.fit(X_train, y_train[:, axis_idx])
        else:
            model.fit(X_train, y_train[:, axis_idx], sample_weight=sample_weight)
        models.append(model)
    return models


def predict_axis_models(models: List[GradientBoostingRegressor], X: np.ndarray) -> np.ndarray:
    preds = [model.predict(X) for model in models]
    return np.column_stack(preds)


def build_residual_feature_matrix(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    feature_cols: List[str],
) -> np.ndarray:
    feature_df = df.reindex(columns=feature_cols, fill_value=0.0).copy()
    for col in feature_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)

    X_base = feature_df.to_numpy(dtype=np.float32)
    pred = np.asarray(base_pred, dtype=np.float32)
    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True).astype(np.float32)
    return np.hstack([X_base, pred, pred_norm]).astype(np.float32)


def fit_residual_corrected_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train_true: np.ndarray,
    y_val_true: np.ndarray,
    train_base_pred: np.ndarray,
    val_base_pred: np.ndarray,
    test_base_pred: np.ndarray,
    feature_cols: List[str],
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    residual_y = y_train_true - train_base_pred
    train_X = build_residual_feature_matrix(train_df, train_base_pred, feature_cols)
    val_X = build_residual_feature_matrix(val_df, val_base_pred, feature_cols)
    test_X = build_residual_feature_matrix(test_df, test_base_pred, feature_cols)

    residual_models = train_axis_models(train_X, residual_y)
    train_residual = np.clip(
        predict_axis_models(residual_models, train_X),
        -RESIDUAL_MAX_ABS_CORRECTION,
        RESIDUAL_MAX_ABS_CORRECTION,
    )
    val_residual = np.clip(
        predict_axis_models(residual_models, val_X),
        -RESIDUAL_MAX_ABS_CORRECTION,
        RESIDUAL_MAX_ABS_CORRECTION,
    )
    test_residual = np.clip(
        predict_axis_models(residual_models, test_X),
        -RESIDUAL_MAX_ABS_CORRECTION,
        RESIDUAL_MAX_ABS_CORRECTION,
    )

    best_scale = max(
        RESIDUAL_SCALE_CANDIDATES,
        key=lambda scale: actual_position_distance_ratio(
            val_df,
            y_val_true,
            val_base_pred + scale * val_residual,
        ),
    )
    print(f"{name} residual correction: best_scale={best_scale:.2f}")
    return (
        train_base_pred + best_scale * train_residual,
        val_base_pred + best_scale * val_residual,
        test_base_pred + best_scale * test_residual,
        float(best_scale),
    )


def get_progress_values(df: pd.DataFrame) -> np.ndarray:
    for col in ["source_line_progress", "source_frame_progress"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return np.zeros(len(df), dtype=np.float32)


def compute_late_stage_residual_gate(
    df: pd.DataFrame,
    apply_low: float = LATE_STAGE_RESIDUAL_APPLY_LOW,
    apply_full: float = LATE_STAGE_RESIDUAL_APPLY_FULL,
) -> np.ndarray:
    progress = get_progress_values(df)
    if apply_full <= apply_low:
        gate = (progress >= apply_low).astype(np.float32)
    else:
        gate = np.clip((progress - apply_low) / (apply_full - apply_low), 0.0, 1.0).astype(np.float32)
    return gate.reshape(-1, 1)


def print_late_stage_candidate_summary(
    df: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
    name: str,
) -> None:
    progress = get_progress_values(df)
    mask = progress >= LATE_STAGE_RESIDUAL_APPLY_LOW
    if int(mask.sum()) == 0:
        return
    distances = np.linalg.norm(np.asarray(y_true)[mask] - np.asarray(pred)[mask], axis=1)
    ratio = float(np.mean(distances < POSITION_DISTANCE_THRESHOLD))
    print(
        f"{name} late-stage progress>={LATE_STAGE_RESIDUAL_APPLY_LOW:.2f}: "
        f"count={int(mask.sum())}, ratio={ratio:.6f}, mean={float(np.mean(distances)):.6f}, "
        f"p90={float(np.quantile(distances, 0.90)):.6f}, max={float(np.max(distances)):.6f}"
    )


def fit_late_stage_residual_corrected_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train_true: np.ndarray,
    y_val_true: np.ndarray,
    train_base_pred: np.ndarray,
    val_base_pred: np.ndarray,
    test_base_pred: np.ndarray,
    feature_cols: List[str],
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    train_progress = get_progress_values(train_df)
    train_mask = train_progress >= LATE_STAGE_RESIDUAL_TRAIN_LOW
    train_count = int(train_mask.sum())
    if train_count < LATE_STAGE_RESIDUAL_MIN_SAMPLES:
        print(
            f"{name} late-stage residual correction skipped: "
            f"train_count={train_count} < {LATE_STAGE_RESIDUAL_MIN_SAMPLES}"
        )
        return train_base_pred, val_base_pred, test_base_pred, 0.0, train_count

    residual_y = y_train_true - train_base_pred
    train_X_all = build_residual_feature_matrix(train_df, train_base_pred, feature_cols)
    val_X = build_residual_feature_matrix(val_df, val_base_pred, feature_cols)
    test_X = build_residual_feature_matrix(test_df, test_base_pred, feature_cols)

    late_progress = np.clip(
        (train_progress[train_mask] - LATE_STAGE_RESIDUAL_TRAIN_LOW) / max(1.0 - LATE_STAGE_RESIDUAL_TRAIN_LOW, 1e-6),
        0.0,
        1.0,
    )
    sample_weight = (1.0 + 2.0 * late_progress).astype(np.float32)
    residual_models = train_axis_models(
        train_X_all[train_mask],
        residual_y[train_mask],
        sample_weight=sample_weight,
    )

    train_residual = np.clip(
        predict_axis_models(residual_models, train_X_all),
        -LATE_STAGE_RESIDUAL_MAX_ABS,
        LATE_STAGE_RESIDUAL_MAX_ABS,
    )
    val_residual = np.clip(
        predict_axis_models(residual_models, val_X),
        -LATE_STAGE_RESIDUAL_MAX_ABS,
        LATE_STAGE_RESIDUAL_MAX_ABS,
    )
    test_residual = np.clip(
        predict_axis_models(residual_models, test_X),
        -LATE_STAGE_RESIDUAL_MAX_ABS,
        LATE_STAGE_RESIDUAL_MAX_ABS,
    )

    train_gate = compute_late_stage_residual_gate(train_df)
    val_gate = compute_late_stage_residual_gate(val_df)
    test_gate = compute_late_stage_residual_gate(test_df)

    best_scale = max(
        LATE_STAGE_RESIDUAL_SCALE_CANDIDATES,
        key=lambda scale: actual_position_distance_ratio(
            val_df,
            y_val_true,
            val_base_pred + scale * val_gate * val_residual,
        ),
    )
    print(
        f"{name} late-stage residual correction: "
        f"train_count={train_count}, best_scale={best_scale:.2f}, "
        f"train_low={LATE_STAGE_RESIDUAL_TRAIN_LOW:.2f}, apply={LATE_STAGE_RESIDUAL_APPLY_LOW:.2f}->{LATE_STAGE_RESIDUAL_APPLY_FULL:.2f}, "
        f"max_abs={LATE_STAGE_RESIDUAL_MAX_ABS:.3f}"
    )
    return (
        train_base_pred + best_scale * train_gate * train_residual,
        val_base_pred + best_scale * val_gate * val_residual,
        test_base_pred + best_scale * test_gate * test_residual,
        float(best_scale),
        train_count,
    )


def compute_near_boundary_residual_gate(
    base_distance: np.ndarray,
    apply_low: float = NEAR_BOUNDARY_RESIDUAL_APPLY_LOW,
    apply_high: float = NEAR_BOUNDARY_RESIDUAL_APPLY_HIGH,
    hard_high: float = NEAR_BOUNDARY_RESIDUAL_HARD_HIGH,
) -> np.ndarray:
    distance = np.asarray(base_distance, dtype=np.float32).reshape(-1)
    gate = np.zeros_like(distance, dtype=np.float32)
    if apply_high <= apply_low:
        return gate.reshape(-1, 1)

    ramp_up_low = max(0.0, apply_low - 0.005)
    ramp_up_width = max(apply_low - ramp_up_low, 1e-6)
    ramp_up_mask = (distance >= ramp_up_low) & (distance < apply_low)
    gate[ramp_up_mask] = (distance[ramp_up_mask] - ramp_up_low) / ramp_up_width

    near_mask = (distance >= apply_low) & (distance <= apply_high)
    gate[near_mask] = 1.0

    tail_width = max(hard_high - apply_high, 1e-6)
    tail_mask = (distance > apply_high) & (distance < hard_high)
    gate[tail_mask] = np.maximum(0.0, 1.0 - (distance[tail_mask] - apply_high) / tail_width)
    return gate.reshape(-1, 1).astype(np.float32)


def fit_near_boundary_residual_corrected_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train_true: np.ndarray,
    y_val_true: np.ndarray,
    train_base_pred: np.ndarray,
    val_base_pred: np.ndarray,
    test_base_pred: np.ndarray,
    feature_cols: List[str],
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    train_distance = actual_position_distance(train_df, y_train_true, train_base_pred)
    val_distance = actual_position_distance(val_df, y_val_true, val_base_pred)

    train_mask = (
        (train_distance >= NEAR_BOUNDARY_RESIDUAL_TRAIN_LOW)
        & (train_distance <= NEAR_BOUNDARY_RESIDUAL_TRAIN_HIGH)
    )
    train_count = int(train_mask.sum())
    if train_count < 24:
        print(f"{name} near-boundary residual correction: skipped, train_near_count={train_count}")
        return train_base_pred.copy(), val_base_pred.copy(), test_base_pred.copy(), 0.0, train_count

    residual_y = y_train_true[train_mask] - train_base_pred[train_mask]
    train_X_all = build_residual_feature_matrix(train_df, train_base_pred, feature_cols)
    val_X = build_residual_feature_matrix(val_df, val_base_pred, feature_cols)
    test_X = build_residual_feature_matrix(test_df, test_base_pred, feature_cols)

    distance_model = None
    train_gate_distance = np.linalg.norm(train_base_pred, axis=1)
    val_gate_distance = np.linalg.norm(val_base_pred, axis=1)
    test_gate_distance = np.linalg.norm(test_base_pred, axis=1)
    if USE_PREDICTED_DISTANCE_RESIDUAL_GATE:
        distance_model = GradientBoostingRegressor(
            random_state=RANDOM_SEED + 17,
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
        )
        distance_weights = np.ones(train_distance.shape, dtype=np.float32)
        distance_weights[
            (train_distance >= NEAR_BOUNDARY_RESIDUAL_TRAIN_LOW)
            & (train_distance <= NEAR_BOUNDARY_RESIDUAL_TRAIN_HIGH)
        ] = 2.0
        distance_weights[
            (train_distance >= POSITION_DISTANCE_THRESHOLD)
            & (train_distance <= NEAR_BOUNDARY_RESIDUAL_APPLY_HIGH)
        ] = 3.0
        distance_model.fit(train_X_all, train_distance, sample_weight=distance_weights)
        train_gate_distance = np.clip(distance_model.predict(train_X_all), 0.0, None)
        val_gate_distance = np.clip(distance_model.predict(val_X), 0.0, None)
        test_gate_distance = np.clip(distance_model.predict(test_X), 0.0, None)

    residual_models = train_axis_models(train_X_all[train_mask], residual_y)
    train_residual = np.clip(
        predict_axis_models(residual_models, train_X_all),
        -NEAR_BOUNDARY_RESIDUAL_MAX_ABS,
        NEAR_BOUNDARY_RESIDUAL_MAX_ABS,
    )
    val_residual = np.clip(
        predict_axis_models(residual_models, val_X),
        -NEAR_BOUNDARY_RESIDUAL_MAX_ABS,
        NEAR_BOUNDARY_RESIDUAL_MAX_ABS,
    )
    test_residual = np.clip(
        predict_axis_models(residual_models, test_X),
        -NEAR_BOUNDARY_RESIDUAL_MAX_ABS,
        NEAR_BOUNDARY_RESIDUAL_MAX_ABS,
    )

    # Use true distance only to select training samples. For validation/test application,
    # use the same visible predicted-distance gate available at inference.
    train_gate = compute_near_boundary_residual_gate(train_gate_distance)
    val_gate = compute_near_boundary_residual_gate(val_gate_distance)
    test_gate = compute_near_boundary_residual_gate(test_gate_distance)
    val_near_true = (
        (val_distance >= NEAR_BOUNDARY_RESIDUAL_APPLY_LOW)
        & (val_distance <= NEAR_BOUNDARY_RESIDUAL_APPLY_HIGH)
    )
    val_near_pred = val_gate.reshape(-1) > 0.5
    val_gate_precision = float(np.mean(val_near_true[val_near_pred])) if np.any(val_near_pred) else 0.0
    val_gate_recall = float(np.mean(val_near_pred[val_near_true])) if np.any(val_near_true) else 0.0

    best_scale = max(
        NEAR_BOUNDARY_RESIDUAL_SCALE_CANDIDATES,
        key=lambda scale: actual_position_distance_ratio(
            val_df,
            y_val_true,
            val_base_pred + scale * val_gate * val_residual,
        ),
    )
    print(
        f"{name} near-boundary residual correction: train_near_count={train_count}, "
        f"gate={'predicted_distance' if distance_model is not None else 'base_pred_norm'}, "
        f"val_gate_mean={float(np.mean(val_gate)):.3f}, test_gate_mean={float(np.mean(test_gate)):.3f}, "
        f"val_gate_precision={val_gate_precision:.3f}, val_gate_recall={val_gate_recall:.3f}, "
        f"best_scale={best_scale:.2f}"
    )
    return (
        train_base_pred + best_scale * train_gate * train_residual,
        val_base_pred + best_scale * val_gate * val_residual,
        test_base_pred + best_scale * test_gate * test_residual,
        float(best_scale),
        train_count,
    )


def fit_per_source_residual_corrected_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train_true: np.ndarray,
    y_val_true: np.ndarray,
    train_base_pred: np.ndarray,
    val_base_pred: np.ndarray,
    test_base_pred: np.ndarray,
    feature_cols: List[str],
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    if "source_file_name" not in train_df.columns:
        return train_base_pred.copy(), val_base_pred.copy(), test_base_pred.copy(), 0.0, 0

    train_residual = np.zeros_like(train_base_pred, dtype=np.float32)
    val_residual = np.zeros_like(val_base_pred, dtype=np.float32)
    test_residual = np.zeros_like(test_base_pred, dtype=np.float32)
    trained_groups = 0

    train_groups = train_df["source_file_name"].astype(str).to_numpy()
    val_groups = val_df["source_file_name"].astype(str).to_numpy()
    test_groups = test_df["source_file_name"].astype(str).to_numpy()

    for source_name in pd.unique(train_groups):
        train_mask = train_groups == source_name
        if int(train_mask.sum()) < PER_SOURCE_RESIDUAL_MIN_SAMPLES:
            continue

        source_residual_y = y_train_true[train_mask] - train_base_pred[train_mask]
        source_train_X = build_residual_feature_matrix(
            train_df.iloc[np.where(train_mask)[0]],
            train_base_pred[train_mask],
            feature_cols,
        )
        source_models = train_axis_models(source_train_X, source_residual_y)
        trained_groups += 1

        for cur_df, cur_base_pred, cur_groups, out_residual in [
            (train_df, train_base_pred, train_groups, train_residual),
            (val_df, val_base_pred, val_groups, val_residual),
            (test_df, test_base_pred, test_groups, test_residual),
        ]:
            cur_mask = cur_groups == source_name
            if not np.any(cur_mask):
                continue
            cur_X = build_residual_feature_matrix(
                cur_df.iloc[np.where(cur_mask)[0]],
                cur_base_pred[cur_mask],
                feature_cols,
            )
            out_residual[cur_mask] = np.clip(
                predict_axis_models(source_models, cur_X),
                -RESIDUAL_MAX_ABS_CORRECTION,
                RESIDUAL_MAX_ABS_CORRECTION,
            )

    best_scale = max(
        RESIDUAL_SCALE_CANDIDATES,
        key=lambda scale: actual_position_distance_ratio(
            val_df,
            y_val_true,
            val_base_pred + scale * val_residual,
        ),
    )
    print(f"{name} per-source residual correction: groups={trained_groups}, best_scale={best_scale:.2f}")
    return (
        train_base_pred + best_scale * train_residual,
        val_base_pred + best_scale * val_residual,
        test_base_pred + best_scale * test_residual,
        float(best_scale),
        trained_groups,
    )


def select_tabular_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude_cols = set(
        FRAME_KEYS
        + TARGET_COLS
        + ERROR_CLASS_COLS
        + [
            "point_index",
            "source_file_name",
            "source_file_path",
            "original_mapped_file_index",
        ]
    )
    return sorted(
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    )


def save_or_show(fig, out_path: str, show_plots: bool):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)


def save_history_plot(history, title: str, out_path: str, show_plots: bool) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_or_show(fig, out_path, show_plots)


def save_compare_plot(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, axis_idx: int,
                      title: str, label_a: str, label_b: str, out_path: str, show_plots: bool) -> None:
    axis_name = ["X", "Y", "Z"][axis_idx]
    fig = plt.figure(figsize=(12, 4))
    plt.plot(y_true[:, axis_idx], label="True")
    plt.plot(y_pred_a[:, axis_idx], label=label_a)
    plt.plot(y_pred_b[:, axis_idx], label=label_b)
    plt.title(f"{title} - {axis_name}")
    plt.xlabel("Sample")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    save_or_show(fig, out_path, show_plots)


def save_single_compare_plot(y_true: np.ndarray, y_pred: np.ndarray, axis_idx: int,
                             title: str, label_pred: str, out_path: str, show_plots: bool) -> None:
    axis_name = ["X", "Y", "Z"][axis_idx]
    fig = plt.figure(figsize=(12, 4))
    plt.plot(y_true[:, axis_idx], label="True")
    plt.plot(y_pred[:, axis_idx], label=label_pred)
    plt.title(f"{title} - {axis_name}")
    plt.xlabel("Sample")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    save_or_show(fig, out_path, show_plots)


def save_position_distance_bar_plot(
    distances: np.ndarray,
    title: str,
    out_path: str,
    show_plots: bool,
    threshold: float = POSITION_DISTANCE_THRESHOLD,
) -> None:
    distances = np.asarray(distances, dtype=np.float32)
    if distances.size == 0:
        return

    x = np.arange(distances.size)
    fail_mask = distances >= threshold
    colors = np.where(fail_mask, "#d95f02", "#1b9e77")

    width = max(12.0, min(32.0, distances.size * 0.11))
    fig, ax = plt.subplots(figsize=(width, 4.8))
    ax.bar(x, distances, color=colors, width=0.85, linewidth=0)
    ax.axhline(threshold, color="#cc0000", linestyle="--", linewidth=1.4, label=f"threshold={threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Test point index")
    ax.set_ylabel("Actual position distance")
    ax.set_ylim(0, max(float(np.nanmax(distances)) * 1.08, threshold * 1.4))
    ax.grid(axis="y", alpha=0.25)

    pass_count = int((~fail_mask).sum())
    fail_count = int(fail_mask.sum())
    ratio = pass_count / max(len(distances), 1)
    ax.text(
        0.01,
        0.96,
        f"pass={pass_count}/{len(distances)} ({ratio:.3f}), fail={fail_count}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.legend(loc="upper right")
    save_or_show(fig, out_path, show_plots)


def save_position_distance_bar_plots(
    pred_df: pd.DataFrame,
    out_dir: str,
    file_prefix: str,
    show_plots: bool,
) -> None:
    if "actual_position_distance" not in pred_df.columns:
        return

    distance_dir = os.path.join(out_dir, "position_distance_bars")
    distances = pd.to_numeric(pred_df["actual_position_distance"], errors="coerce").to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(distances)
    if not np.any(finite_mask):
        return

    save_position_distance_bar_plot(
        distances[finite_mask],
        "Test Actual Position Distance - All Sources",
        os.path.join(distance_dir, f"{file_prefix}_all_position_distance_bar.png"),
        show_plots,
    )

    if "source_file_name" not in pred_df.columns:
        return
    for source_name, group in pred_df.groupby("source_file_name", sort=False):
        source_distances = pd.to_numeric(group["actual_position_distance"], errors="coerce").to_numpy(dtype=np.float32)
        source_distances = source_distances[np.isfinite(source_distances)]
        if source_distances.size == 0:
            continue
        save_position_distance_bar_plot(
            source_distances,
            f"Test Actual Position Distance - {source_name}",
            os.path.join(
                distance_dir,
                safe_path_name(str(source_name)),
                f"{file_prefix}_{safe_path_name(str(source_name))}_position_distance_bar.png",
            ),
            show_plots,
        )


def save_source_compare_plots(
    split_name: str,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_label: str,
    out_dir: str,
    file_prefix: str,
    show_plots: bool,
) -> None:
    if "source_file_name" not in meta_df.columns or len(meta_df) != len(y_true):
        return

    source_plot_dir = os.path.join(out_dir, f"{split_name}_by_model")
    for source_name in meta_df["source_file_name"].drop_duplicates():
        mask = (meta_df["source_file_name"] == source_name).to_numpy()
        if not np.any(mask):
            continue

        source_dir = os.path.join(source_plot_dir, safe_path_name(source_name))
        for i, axis in enumerate(["x", "y", "z"]):
            save_single_compare_plot(
                y_true[mask],
                y_pred[mask],
                i,
                f"{split_name.capitalize()} {model_label} - {source_name}",
                f"{model_label} Pred",
                os.path.join(source_dir, f"{split_name}_{file_prefix}_{axis}.png"),
                show_plots,
            )


def save_split_plots(split_name: str,
                     points_df: pd.DataFrame,
                     y_true: np.ndarray,
                     trend_pred: np.ndarray,
                     final_pred: np.ndarray,
                     residual_pred: np.ndarray,
                     out_dir: str,
                     show_plots: bool) -> None:
    thermal_pred = points_df[["thermal_pred_x", "thermal_pred_y", "thermal_pred_z"]].values
    split_title = split_name.capitalize()
    for i, axis in enumerate(["x", "y", "z"]):
        save_single_compare_plot(
            y_true, thermal_pred, i,
            f"{split_title} Thermal", "Thermal Pred",
            os.path.join(out_dir, f"{split_name}_thermal_{axis}_v7.png"),
            show_plots,
        )
        save_single_compare_plot(
            y_true, trend_pred, i,
            f"{split_title} Trend", "Trend Pred",
            os.path.join(out_dir, f"{split_name}_trend_{axis}_v7.png"),
            show_plots,
        )
        save_compare_plot(
            y_true, trend_pred, final_pred, i,
            f"{split_title} True vs Trend vs Final", "Trend Pred", "Final Pred",
            os.path.join(out_dir, f"{split_name}_final_{axis}_v7.png"),
            show_plots,
        )
        save_single_compare_plot(
            y_true - trend_pred, residual_pred, i,
            f"{split_title} Residual", "Residual Pred",
            os.path.join(out_dir, f"{split_name}_residual_{axis}_v7.png"),
            show_plots,
        )


# =========================================================
# 2. Load JSON
# =========================================================
def load_dataset(json_path: str, file_index_offset: int = 0) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw["point_to_frame_ai_block_avg"]:
        row = {}
        row["source_file_name"] = os.path.basename(json_path)
        row["source_file_path"] = json_path
        row["point_index"] = item.get("point_index", -1)
        original_file_index = int(item.get("mapped_file_index", -1))
        row["mapped_file_index"] = original_file_index + file_index_offset
        row["original_mapped_file_index"] = original_file_index
        row["mapped_frame_id"] = item.get("mapped_frame_id", -1)
        row["mapped_frame_num"] = item.get("mapped_frame_num", -1)
        row["mapped_nc_line"] = item.get("mapped_nc_line", -1)

        err = item.get("point_error_xyz", [0.0, 0.0, 0.0])
        row["err_x"] = to_float_or_zero(err[0] if len(err) > 0 else 0.0)
        row["err_y"] = to_float_or_zero(err[1] if len(err) > 1 else 0.0)
        row["err_z"] = to_float_or_zero(err[2] if len(err) > 2 else 0.0)

        pose = item.get("mapped_frame_pose", {})
        row["pose_x"] = to_float_or_zero(pose.get("x", 0.0))
        row["pose_y"] = to_float_or_zero(pose.get("y", 0.0))
        row["pose_z"] = to_float_or_zero(pose.get("z", 0.0))
        row["pose_a"] = to_float_or_zero(pose.get("a", 0.0))
        row["pose_c"] = to_float_or_zero(pose.get("c", 0.0))

        ms = item.get("mapped_machine_state", {})
        row["feed_speed"] = to_float_or_zero(ms.get("feed_speed", 0.0))
        row["spindle_speed"] = to_float_or_zero(ms.get("spindle_speed", 0.0))

        motion = item.get("mapped_motion", {})
        row.update(flatten_numeric_dict(motion, "motion"))

        ai = item.get("ai_block_avg", {})
        for ch in TEMP_CHANNELS:
            row[f"temp_ch{ch}_median_c"] = to_float_or_zero(ai.get(f"temperature.ch{ch}.median_c", 0.0))
            row[f"temp_ch{ch}_std_c"] = to_float_or_zero(ai.get(f"temperature.ch{ch}.std_c", 0.0))

        for vib_ch in [0, 1, 2]:
            for feat in USE_VIB_FEATURES:
                key = f"vibration.ch{vib_ch}.{feat}"
                row[f"vib_ch{vib_ch}_{feat}"] = to_float_or_zero(ai.get(key, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=FRAME_KEYS + ["point_index"], kind="mergesort").reset_index(drop=True)
    return df


def load_and_merge_datasets(json_paths: List[str]) -> pd.DataFrame:
    merged_parts: List[pd.DataFrame] = []
    file_index_offset = SELECTED_FILE_INDEX_OFFSET

    for json_path in json_paths:
        df = load_dataset(json_path, file_index_offset=file_index_offset)
        print(f"Loaded {len(df)} rows from: {json_path}")

        if df.empty:
            continue

        merged_parts.append(df)
        file_index_offset = int(df["mapped_file_index"].max()) + 1

    if not merged_parts:
        raise ValueError("All JSON datasets are empty. Nothing to train on.")

    merged_df = pd.concat(merged_parts, axis=0, ignore_index=True, sort=False)
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df.loc[:, numeric_cols] = merged_df[numeric_cols].fillna(0.0)
    merged_df = merged_df.sort_values(
        by=FRAME_KEYS + ["point_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"Merged {len(merged_parts)} JSON files into one dataset: {merged_df.shape}")
    return merged_df


def compute_group_split_sizes(n: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0

    val_n = int(n * val_ratio)
    test_n = int(n * test_ratio)

    if n >= 3 and val_ratio > 0:
        val_n = max(val_n, 1)
    if n >= 3 and test_ratio > 0:
        test_n = max(test_n, 1)

    while val_n + test_n >= n:
        if test_n >= val_n and test_n > 0:
            test_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            break

    train_n = n - val_n - test_n
    return train_n, val_n, test_n


def split_frame_dataset_by_source(
    frame_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    group_col: str = "source_file_name",
    phase_blocks: int = PHASE_SPLIT_BLOCKS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    print("\nFrame split by machining process phase blocks:")
    for source_idx, (source_name, group) in enumerate(frame_df.groupby(group_col, sort=False)):
        group = group.sort_values(by=FRAME_KEYS, kind="mergesort").reset_index(drop=True)

        source_train_parts: List[pd.DataFrame] = []
        source_val_parts: List[pd.DataFrame] = []
        source_test_parts: List[pd.DataFrame] = []
        block_count = max(1, min(phase_blocks, len(group)))

        for block_indices in np.array_split(np.arange(len(group)), block_count):
            block = group.iloc[block_indices].reset_index(drop=True)
            train_n, val_n, test_n = compute_group_split_sizes(len(block), val_ratio, test_ratio)
            train_part = block.iloc[:train_n].copy()
            val_part = block.iloc[train_n:train_n + val_n].copy()
            test_part = block.iloc[train_n + val_n:train_n + val_n + test_n].copy()

            if not train_part.empty:
                source_train_parts.append(train_part)
            if not val_part.empty:
                source_val_parts.append(val_part)
            if not test_part.empty:
                source_test_parts.append(test_part)

        train_part = pd.concat(source_train_parts, ignore_index=True) if source_train_parts else group.iloc[0:0].copy()
        val_part = pd.concat(source_val_parts, ignore_index=True) if source_val_parts else group.iloc[0:0].copy()
        test_part = pd.concat(source_test_parts, ignore_index=True) if source_test_parts else group.iloc[0:0].copy()

        print(
            f"  {source_display_label(source_name, source_idx)}: total_frames={len(group)}, phase_blocks={block_count}, "
            f"train_frames={len(train_part)}, val_frames={len(val_part)}, test_frames={len(test_part)}"
        )

        if not train_part.empty:
            train_parts.append(train_part)
        if not val_part.empty:
            val_parts.append(val_part)
        if not test_part.empty:
            test_parts.append(test_part)

    empty_frame = frame_df.iloc[0:0].copy()
    train_frame = pd.concat(train_parts, ignore_index=True) if train_parts else empty_frame.copy()
    val_frame = pd.concat(val_parts, ignore_index=True) if val_parts else empty_frame.copy()
    test_frame = pd.concat(test_parts, ignore_index=True) if test_parts else empty_frame.copy()
    return train_frame, val_frame, test_frame


def split_frame_dataset_random_frames(
    frame_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    group_col: str = "source_file_name",
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)

    print("\nFrame split by random frames within each machining process:")
    for source_idx, (source_name, group) in enumerate(frame_df.groupby(group_col, sort=False)):
        group = group.sort_values(by=FRAME_KEYS, kind="mergesort").reset_index(drop=True)
        train_n, val_n, test_n = compute_group_split_sizes(len(group), val_ratio, test_ratio)
        indices = np.arange(len(group))
        rng.shuffle(indices)

        train_idx = np.sort(indices[:train_n])
        val_idx = np.sort(indices[train_n:train_n + val_n])
        test_idx = np.sort(indices[train_n + val_n:train_n + val_n + test_n])

        train_part = group.iloc[train_idx].copy()
        val_part = group.iloc[val_idx].copy()
        test_part = group.iloc[test_idx].copy()

        print(
            f"  {source_display_label(source_name, source_idx)}: total_frames={len(group)}, "
            f"train_frames={len(train_part)}, val_frames={len(val_part)}, test_frames={len(test_part)}"
        )

        if not train_part.empty:
            train_parts.append(train_part)
        if not val_part.empty:
            val_parts.append(val_part)
        if not test_part.empty:
            test_parts.append(test_part)

    empty_frame = frame_df.iloc[0:0].copy()
    train_frame = pd.concat(train_parts, ignore_index=True) if train_parts else empty_frame.copy()
    val_frame = pd.concat(val_parts, ignore_index=True) if val_parts else empty_frame.copy()
    test_frame = pd.concat(test_parts, ignore_index=True) if test_parts else empty_frame.copy()
    return train_frame, val_frame, test_frame


def compute_chunk_split_counts(chunk_count: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if chunk_count <= 0:
        return 0, 0, 0
    if chunk_count == 1:
        return 1, 0, 0

    val_n = int(round(chunk_count * val_ratio))
    test_n = int(round(chunk_count * test_ratio))
    if chunk_count >= 3 and val_ratio > 0:
        val_n = max(val_n, 1)
    if chunk_count >= 3 and test_ratio > 0:
        test_n = max(test_n, 1)

    while val_n + test_n >= chunk_count:
        if test_n >= val_n and test_n > 0:
            test_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            break

    train_n = chunk_count - val_n - test_n
    return train_n, val_n, test_n


def split_frame_dataset_random_phase_chunks(
    frame_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    group_col: str = "source_file_name",
    chunk_count: int = RANDOM_CHUNK_SPLIT_COUNT,
    phase_blocks: int = PHASE_SPLIT_BLOCKS,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)

    print("\nFrame split by phase blocks, with randomized chunks inside each phase block:")
    for source_idx, (source_name, group) in enumerate(frame_df.groupby(group_col, sort=False)):
        group = group.sort_values(by=FRAME_KEYS, kind="mergesort").reset_index(drop=True)
        block_count = max(1, min(phase_blocks, len(group)))
        chunks_per_block = max(1, int(round(max(chunk_count, block_count) / block_count)))

        source_train_parts: List[pd.DataFrame] = []
        source_val_parts: List[pd.DataFrame] = []
        source_test_parts: List[pd.DataFrame] = []
        total_inner_chunks = 0

        for block_indices in np.array_split(np.arange(len(group)), block_count):
            block = group.iloc[block_indices].reset_index(drop=True)
            cur_inner_count = max(1, min(chunks_per_block, len(block)))
            chunks = [
                block.iloc[idx].copy()
                for idx in np.array_split(np.arange(len(block)), cur_inner_count)
                if len(idx) > 0
            ]
            total_inner_chunks += len(chunks)

            order = np.arange(len(chunks))
            rng.shuffle(order)
            shuffled_block = pd.concat([chunks[idx] for idx in order], ignore_index=True)
            train_n, val_n, test_n = compute_group_split_sizes(len(shuffled_block), val_ratio, test_ratio)

            train_part = shuffled_block.iloc[:train_n].copy()
            val_part = shuffled_block.iloc[train_n:train_n + val_n].copy()
            test_part = shuffled_block.iloc[train_n + val_n:train_n + val_n + test_n].copy()

            if not train_part.empty:
                source_train_parts.append(train_part)
            if not val_part.empty:
                source_val_parts.append(val_part)
            if not test_part.empty:
                source_test_parts.append(test_part)

        train_part = pd.concat(source_train_parts, ignore_index=True) if source_train_parts else group.iloc[0:0].copy()
        val_part = pd.concat(source_val_parts, ignore_index=True) if source_val_parts else group.iloc[0:0].copy()
        test_part = pd.concat(source_test_parts, ignore_index=True) if source_test_parts else group.iloc[0:0].copy()

        train_part = train_part.sort_values(by=FRAME_KEYS, kind="mergesort").reset_index(drop=True)
        val_part = val_part.sort_values(by=FRAME_KEYS, kind="mergesort").reset_index(drop=True)
        test_part = test_part.sort_values(by=FRAME_KEYS, kind="mergesort").reset_index(drop=True)

        print(
            f"  {source_display_label(source_name, source_idx)}: total_frames={len(group)}, phase_blocks={block_count}, "
            f"chunks_per_block={chunks_per_block}, total_inner_chunks={total_inner_chunks}, "
            f"train_frames={len(train_part)}, val_frames={len(val_part)}, test_frames={len(test_part)}"
        )

        if not train_part.empty:
            train_parts.append(train_part)
        if not val_part.empty:
            val_parts.append(val_part)
        if not test_part.empty:
            test_parts.append(test_part)

    empty_frame = frame_df.iloc[0:0].copy()
    train_frame = pd.concat(train_parts, ignore_index=True) if train_parts else empty_frame.copy()
    val_frame = pd.concat(val_parts, ignore_index=True) if val_parts else empty_frame.copy()
    test_frame = pd.concat(test_parts, ignore_index=True) if test_parts else empty_frame.copy()
    return train_frame, val_frame, test_frame


def split_frame_dataset_leave_one_json_out(
    frame_df: pd.DataFrame,
    test_json_index: int,
    val_ratio: float,
    group_col: str = "source_file_name",
    phase_blocks: int = PHASE_SPLIT_BLOCKS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_names = list(frame_df[group_col].drop_duplicates())
    if len(source_names) < 2:
        raise ValueError("Need at least 2 JSON files to use one complete JSON as the test set.")
    if test_json_index < 0 or test_json_index >= len(source_names):
        raise ValueError(
            f"TEST_JSON_INDEX={test_json_index} is out of range. "
            f"Available range is 0-{len(source_names) - 1}."
        )

    test_source_name = source_names[test_json_index]
    print("\nLeave-one-machining-process-out frame split:")
    for idx, source_name in enumerate(source_names):
        role = "test" if idx == test_json_index else "train/val"
        print(f"  [{idx}] {source_display_label(source_name, idx)}: {role}")

    test_frame = (
        frame_df[frame_df[group_col] == test_source_name]
        .sort_values(by=FRAME_KEYS, kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    train_val_frame = (
        frame_df[frame_df[group_col] != test_source_name]
        .sort_values(by=[group_col] + FRAME_KEYS, kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )

    train_frame, val_frame, _ = split_frame_dataset_by_source(
        train_val_frame,
        val_ratio=val_ratio,
        test_ratio=0.0,
        group_col=group_col,
        phase_blocks=phase_blocks,
    )
    return train_frame, val_frame, test_frame


def filter_test_frame_by_source(
    test_frame: pd.DataFrame,
    all_frame: pd.DataFrame,
    source_name: str = "",
    source_index=None,
    group_col: str = "source_file_name",
) -> pd.DataFrame:
    source_names = list(all_frame[group_col].drop_duplicates())

    print("\nAvailable machining process indices:")
    for idx, cur_source_name in enumerate(source_names):
        print(f"  [{idx}] {source_display_label(cur_source_name, idx)}")

    selected_source_name = source_name.strip() if isinstance(source_name, str) else ""
    if selected_source_name:
        if selected_source_name not in source_names:
            raise ValueError(
                f"PREDICT_SOURCE_NAME={selected_source_name!r} was not found. "
                f"Available source names: {source_names}"
            )
    elif source_index is not None:
        if source_index < 0 or source_index >= len(source_names):
            raise ValueError(
                f"PREDICT_SOURCE_INDEX={source_index} is out of range. "
                f"Available range is 0-{len(source_names) - 1}."
            )
        selected_source_name = source_names[source_index]

    if not selected_source_name:
        print("Prediction source selection: all test machining processes")
        return test_frame

    filtered_test_frame = (
        test_frame[test_frame[group_col] == selected_source_name]
        .reset_index(drop=True)
        .copy()
    )
    if filtered_test_frame.empty:
        raise ValueError(
            f"Selected prediction source {selected_source_name!r} has no frames in the current test split. "
            "Use SPLIT_MODE='phase_blocks', or set TEST_JSON_INDEX/PREDICT_SOURCE_INDEX to the same source "
            "when using SPLIT_MODE='leave_one_json_out'."
        )

    selected_idx = source_names.index(selected_source_name)
    print(
        f"Prediction source selection: {source_display_label(selected_source_name, selected_idx)} "
        f"({len(filtered_test_frame)}/{len(test_frame)} test frames kept)"
    )
    return filtered_test_frame


def print_point_split_by_source(train_points: pd.DataFrame, val_points: pd.DataFrame, test_points: pd.DataFrame) -> None:
    source_names = sorted(
        set(train_points.get("source_file_name", pd.Series(dtype=str)).tolist()) |
        set(val_points.get("source_file_name", pd.Series(dtype=str)).tolist()) |
        set(test_points.get("source_file_name", pd.Series(dtype=str)).tolist())
    )
    if not source_names:
        return

    print("\nPoint split by machining process:")
    for source_idx, source_name in enumerate(source_names):
        train_n = int((train_points["source_file_name"] == source_name).sum()) if "source_file_name" in train_points.columns else 0
        val_n = int((val_points["source_file_name"] == source_name).sum()) if "source_file_name" in val_points.columns else 0
        test_n = int((test_points["source_file_name"] == source_name).sum()) if "source_file_name" in test_points.columns else 0
        total_n = train_n + val_n + test_n
        print(f"  {source_display_label(source_name, source_idx)}: total={total_n}, train={train_n}, val={val_n}, test={test_n}")


def print_split_frame_overlap(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    train_keys = set(map(tuple, train_frame[FRAME_KEYS].drop_duplicates().to_numpy().tolist()))
    val_keys = set(map(tuple, val_frame[FRAME_KEYS].drop_duplicates().to_numpy().tolist()))
    test_keys = set(map(tuple, test_frame[FRAME_KEYS].drop_duplicates().to_numpy().tolist()))

    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    print("\nSplit frame-key overlap check:")
    print(
        f"  unique frames -> train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}"
    )
    print(
        f"  overlaps -> train/val={len(train_val_overlap)}, "
        f"train/test={len(train_test_overlap)}, val/test={len(val_test_overlap)}"
    )
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  WARNING: Some exact frame keys appear in more than one split.")




# =========================================================
# 3. Feature engineering
# =========================================================
def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    med_cols = [f"temp_ch{ch}_median_c" for ch in TEMP_CHANNELS]
    std_cols = [f"temp_ch{ch}_std_c" for ch in TEMP_CHANNELS]
    spindle_med_cols = [f"temp_ch{ch}_median_c" for ch in SPINDLE_TEMP_CHANNELS]
    spindle_std_cols = [f"temp_ch{ch}_std_c" for ch in SPINDLE_TEMP_CHANNELS]
    ambient_col = f"temp_ch{AMBIENT_TEMP_CHANNEL}_median_c"

    out["temp_ambient_c"] = out[ambient_col]
    out["temp_spindle_mean"] = out[spindle_med_cols].mean(axis=1)
    out["temp_spindle_max"] = out[spindle_med_cols].max(axis=1)
    out["temp_spindle_min"] = out[spindle_med_cols].min(axis=1)
    out["temp_spindle_range"] = out["temp_spindle_max"] - out["temp_spindle_min"]
    out["temp_spindle_std_mean"] = out[spindle_std_cols].mean(axis=1)
    out["temp_spindle_ambient_delta"] = out["temp_spindle_mean"] - out["temp_ambient_c"]

    # Backward-compatible names now represent spindle temperature only; ch5 is ambient.
    out["temp_median_mean"] = out["temp_spindle_mean"]
    out["temp_median_max"] = out["temp_spindle_max"]
    out["temp_median_min"] = out["temp_spindle_min"]
    out["temp_median_range"] = out["temp_median_max"] - out["temp_median_min"]
    out["temp_std_mean"] = out["temp_spindle_std_mean"]
    out["temp_all_sensor_mean"] = out[med_cols].mean(axis=1)
    out["temp_all_sensor_std_mean"] = out[std_cols].mean(axis=1)

    for i, ch1 in enumerate(TEMP_CHANNELS):
        for ch2 in TEMP_CHANNELS[i + 1:]:
            out[f"temp_diff_{ch1}_{ch2}"] = out[f"temp_ch{ch1}_median_c"] - out[f"temp_ch{ch2}_median_c"]
    for ch in SPINDLE_TEMP_CHANNELS:
        out[f"temp_ch{ch}_ambient_delta"] = out[f"temp_ch{ch}_median_c"] - out["temp_ambient_c"]

    # 关键修正：热项的差分在 frame 级别做，而不是点级别乱 diff
    return out


def add_targeted_features(df: pd.DataFrame, group_col: str = "mapped_file_index") -> pd.DataFrame:
    out = df.copy()

    # Spatial nonlinear terms for X-axis difficulty
    out["pose_x_sq"] = out["pose_x"] ** 2
    out["pose_y_sq"] = out["pose_y"] ** 2
    out["pose_z_sq"] = out["pose_z"] ** 2
    out["pose_xy"] = out["pose_x"] * out["pose_y"]
    out["pose_xz"] = out["pose_x"] * out["pose_z"]
    out["pose_yz"] = out["pose_y"] * out["pose_z"]

    grouped = out.groupby(group_col, sort=False)

    # Motion-state features for Y-axis difficulty
    out["d_feed_speed"] = grouped["feed_speed"].diff().fillna(0.0)
    out["d_spindle_speed"] = grouped["spindle_speed"].diff().fillna(0.0)
    out["feed_speed_roll3_mean"] = grouped["feed_speed"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out["spindle_speed_roll3_mean"] = grouped["spindle_speed"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out["feed_pose_y_interact"] = out["feed_speed"] * out["pose_y"]
    out["feed_pose_rxyz_interact"] = out["feed_speed"] * out["pose_r_xyz"]
    out["spindle_pose_y_interact"] = out["spindle_speed"] * out["pose_y"]

    return out


def add_y_axis_targeted_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pose_x_abs"] = np.abs(out["pose_x"])
    out["pose_y_abs"] = np.abs(out["pose_y"])
    out["pose_z_abs"] = np.abs(out["pose_z"])
    out["pose_x_sign"] = np.sign(out["pose_x"])
    out["pose_y_sign"] = np.sign(out["pose_y"])
    out["pose_z_sign"] = np.sign(out["pose_z"])

    acc_x_col = "motion_acceleration_a_X"
    acc_y_col = "motion_acceleration_a_Y"
    if acc_x_col in out.columns and acc_y_col in out.columns:
        out["motion_acc_xy_norm"] = np.sqrt(out[acc_x_col] ** 2 + out[acc_y_col] ** 2)
        out["motion_acc_x_sign"] = np.sign(out[acc_x_col])
        out["motion_acc_y_sign"] = np.sign(out[acc_y_col])
        out["pose_y_acc_y_interact"] = out["pose_y"] * out[acc_y_col]
        out["pose_x_acc_y_interact"] = out["pose_x"] * out[acc_y_col]
        out["pose_z_acc_y_interact"] = out["pose_z"] * out[acc_y_col]
        out["pose_y_acc_x_interact"] = out["pose_y"] * out[acc_x_col]

    return out


def add_source_onehot_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source_file_name" not in out.columns:
        return out

    source_names = sorted(out["source_file_name"].dropna().astype(str).unique().tolist())
    for source_name in source_names:
        out[f"{SOURCE_ONEHOT_PREFIX}{safe_path_name(source_name)}"] = (
            out["source_file_name"].astype(str) == source_name
        ).astype(np.float32)
    return out


def add_source_progress_features(df: pd.DataFrame, group_col: str = "source_file_name") -> pd.DataFrame:
    out = df.copy()
    if group_col not in out.columns:
        group_col = "mapped_file_index"
    if group_col not in out.columns:
        return out

    grouped = out.groupby(group_col, sort=False)
    for col, name in [
        ("mapped_nc_line", "source_line_progress"),
        ("mapped_frame_num", "source_frame_progress"),
    ]:
        if col not in out.columns:
            continue
        min_v = grouped[col].transform("min")
        max_v = grouped[col].transform("max")
        denom = (max_v - min_v).replace(0, np.nan)
        progress = ((out[col] - min_v) / denom).fillna(0.0)
        out[name] = progress.astype(np.float32)
        out[f"{name}_centered"] = (out[name] - 0.5).astype(np.float32)
        out[f"{name}_sq"] = (out[name] ** 2).astype(np.float32)

    if "source_line_progress" in out.columns and "source_frame_progress" in out.columns:
        out["source_line_frame_progress_diff"] = (
            out["source_line_progress"] - out["source_frame_progress"]
        ).astype(np.float32)
    return out


def add_long_rolling_frame_features(frame_df: pd.DataFrame, group_col: str = "source_file_name") -> pd.DataFrame:
    out = frame_df.copy()
    if group_col not in out.columns:
        group_col = "mapped_file_index"
    if group_col not in out.columns:
        return out

    sort_cols = [group_col] + FRAME_KEYS
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    grouped = out.groupby(group_col, sort=False)
    base_cols = [
        "temp_median_mean",
        "temp_median_range",
        "temp_std_mean",
        "motion_linear_acc_xyz",
        "motion_linear_speed_xyz",
        "motion_dt",
        "temp_spindle_ambient_delta",
    ]
    base_cols.extend([f"temp_ch{ch}_median_c" for ch in TEMP_CHANNELS])
    base_cols.extend([f"temp_ch{ch}_ambient_delta" for ch in SPINDLE_TEMP_CHANNELS])
    base_cols = unique_preserve_order([
        col for col in base_cols
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col])
    ])

    for col in base_cols:
        for window in LONG_ROLLING_WINDOWS:
            roll = grouped[col].transform(lambda s, w=window: s.rolling(window=w, min_periods=1).mean())
            out[f"{col}_roll{window}_mean"] = roll.astype(np.float32)
            if window >= 20:
                roll_std = grouped[col].transform(lambda s, w=window: s.rolling(window=w, min_periods=2).std().fillna(0.0))
                out[f"{col}_roll{window}_std"] = roll_std.astype(np.float32)
            out[f"{col}_roll{window}_delta"] = (out[col] - roll).astype(np.float32)

    if "source_line_progress" in out.columns and "temp_median_mean" in out.columns:
        out["progress_temp_interact"] = (out["source_line_progress"] * out["temp_median_mean"]).astype(np.float32)
    if "source_line_progress" in out.columns and "motion_linear_acc_xyz" in out.columns:
        out["progress_motion_acc_interact"] = (out["source_line_progress"] * out["motion_linear_acc_xyz"]).astype(np.float32)

    return out.sort_values(FRAME_KEYS, kind="mergesort").reset_index(drop=True)


def add_thermal_accumulation_features(frame_df: pd.DataFrame, group_col: str = "source_file_name") -> pd.DataFrame:
    out = frame_df.copy()
    if group_col not in out.columns:
        group_col = "mapped_file_index"
    if group_col not in out.columns:
        return out

    out = out.sort_values([group_col] + FRAME_KEYS, kind="mergesort").reset_index(drop=True)
    grouped = out.groupby(group_col, sort=False)
    mode = THERMAL_ACCUMULATION_MODE.strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return out.sort_values(FRAME_KEYS, kind="mergesort").reset_index(drop=True)

    temp_cols = [
        "temp_median_mean",
        "temp_median_range",
        "temp_std_mean",
        "temp_spindle_ambient_delta",
    ]
    temp_cols.extend([f"temp_ch{ch}_ambient_delta" for ch in SPINDLE_TEMP_CHANNELS])
    temp_cols.extend([f"temp_ch{ch}_median_c" for ch in TEMP_CHANNELS])
    if mode in {"minimal", "lite", "small"}:
        temp_cols = [
            "temp_median_mean",
            "temp_spindle_ambient_delta",
            *[f"temp_ch{ch}_ambient_delta" for ch in SPINDLE_TEMP_CHANNELS],
        ]
    elif mode in {"summary", "compact"}:
        temp_cols = [
            "temp_median_mean",
            "temp_median_range",
            "temp_spindle_ambient_delta",
            *[f"temp_ch{ch}_ambient_delta" for ch in SPINDLE_TEMP_CHANNELS],
        ]
    temp_cols = unique_preserve_order([
        col for col in temp_cols
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col])
    ])

    for col in temp_cols:
        start_v = grouped[col].transform("first")
        out[f"{col}_rise_from_start"] = (out[col] - start_v).astype(np.float32)
        if mode not in {"minimal", "lite", "small"}:
            expanding_mean = grouped[col].transform(lambda s: s.expanding(min_periods=1).mean())
            expanding_max = grouped[col].transform(lambda s: s.expanding(min_periods=1).max())
            out[f"{col}_expanding_mean"] = expanding_mean.astype(np.float32)
            out[f"{col}_expanding_max"] = expanding_max.astype(np.float32)
            out[f"{col}_expanding_delta"] = (out[col] - expanding_mean).astype(np.float32)

    if "source_line_progress" in out.columns:
        progress = out["source_line_progress"].astype(np.float32)
    elif "source_frame_progress" in out.columns:
        progress = out["source_frame_progress"].astype(np.float32)
    else:
        progress = None

    if progress is not None:
        if "temp_median_mean_rise_from_start" in out.columns:
            out["progress_temp_rise_interact"] = (
                progress * out["temp_median_mean_rise_from_start"]
            ).astype(np.float32)
        if "temp_spindle_ambient_delta_rise_from_start" in out.columns:
            out["progress_spindle_ambient_delta_rise_interact"] = (
                progress * out["temp_spindle_ambient_delta_rise_from_start"]
            ).astype(np.float32)
        for ch in SPINDLE_TEMP_CHANNELS:
            rise_col = f"temp_ch{ch}_ambient_delta_rise_from_start"
            if rise_col in out.columns:
                out[f"progress_ch{ch}_ambient_delta_rise_interact"] = (
                    progress * out[rise_col]
                ).astype(np.float32)
        if mode not in {"minimal", "lite", "small"} and "temp_median_mean_expanding_mean" in out.columns:
            out["progress_temp_expanding_mean_interact"] = (
                progress * out["temp_median_mean_expanding_mean"]
            ).astype(np.float32)
        if mode not in {"minimal", "lite", "small"} and "temp_spindle_ambient_delta_expanding_mean" in out.columns:
            out["progress_spindle_ambient_delta_expanding_mean_interact"] = (
                progress * out["temp_spindle_ambient_delta_expanding_mean"]
            ).astype(np.float32)
        if mode not in {"minimal", "lite", "small"} and "temp_median_range_rise_from_start" in out.columns:
            out["progress_temp_range_rise_interact"] = (
                progress * out["temp_median_range_rise_from_start"]
            ).astype(np.float32)

    return out.sort_values(FRAME_KEYS, kind="mergesort").reset_index(drop=True)


def merge_frame_context_to_points(point_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    context_prefixes = (
        "source_line_progress",
        "source_frame_progress",
        "source_line_frame_progress_diff",
        "progress_",
    )
    context_suffixes = (
        "_rise_from_start",
        "_expanding_mean",
        "_expanding_max",
        "_expanding_delta",
    )
    context_cols = [
        col for col in frame_df.columns
        if col not in point_df.columns
        and (
            col.startswith(context_prefixes)
            or col.endswith(context_suffixes)
            or "_roll" in col
        )
        and pd.api.types.is_numeric_dtype(frame_df[col])
    ]
    if not context_cols:
        return point_df
    return point_df.merge(frame_df[FRAME_KEYS + context_cols], on=FRAME_KEYS, how="left")


def is_progress_context_feature(col: str) -> bool:
    return (
        col.startswith("source_line_progress")
        or col.startswith("source_frame_progress")
        or col.startswith("source_line_frame_progress_diff")
    )


def is_thermal_accumulation_feature(col: str) -> bool:
    return (
        col.endswith((
            "_rise_from_start",
            "_expanding_mean",
            "_expanding_max",
            "_expanding_delta",
        ))
        or col.startswith("progress_temp")
        or col.startswith("progress_spindle_ambient_delta")
        or (col.startswith("progress_ch") and "ambient_delta" in col)
    )


def is_long_rolling_context_feature(col: str) -> bool:
    return "_roll" in col or col == "progress_motion_acc_interact"


def is_thermal_history_feature(col: str) -> bool:
    return (
        col.startswith("temp_")
        or col.startswith("progress_temp")
        or col.startswith("progress_spindle_ambient_delta")
        or (col.startswith("progress_ch") and "ambient_delta" in col)
    )


def is_context_feature(col: str) -> bool:
    return (
        is_progress_context_feature(col)
        or is_thermal_accumulation_feature(col)
        or is_long_rolling_context_feature(col)
    )


def is_minimal_context_feature(col: str) -> bool:
    return col in {
        "source_line_progress",
        "source_frame_progress",
        "source_line_progress_sq",
        "source_frame_progress_sq",
        "temp_median_mean_rise_from_start",
        "temp_spindle_ambient_delta_rise_from_start",
        "temp_ch3_ambient_delta_rise_from_start",
        "temp_ch4_ambient_delta_rise_from_start",
        "temp_ch6_ambient_delta_rise_from_start",
        "progress_temp_rise_interact",
        "progress_spindle_ambient_delta_rise_interact",
        "progress_ch3_ambient_delta_rise_interact",
        "progress_ch4_ambient_delta_rise_interact",
        "progress_ch6_ambient_delta_rise_interact",
    }


def keep_feature_for_mode(col: str, mode: str = None) -> bool:
    mode = (mode or FEATURE_SET_MODE).strip().lower()
    if mode in {"current", "all", "full"}:
        return True
    if mode in {"minimal", "minimal_context", "compact"}:
        return not is_context_feature(col) or is_minimal_context_feature(col)
    if mode in {"thermal_history", "thermal_multiscale"}:
        return (not is_context_feature(col)) or is_minimal_context_feature(col) or is_thermal_history_feature(col)
    if mode in {"no_context", "base"}:
        return not is_context_feature(col)
    if mode in {"no_thermal", "progress_only"}:
        return not is_thermal_accumulation_feature(col) and not is_long_rolling_context_feature(col)
    if mode in {"thermal_only", "no_progress"}:
        return not is_progress_context_feature(col) and not is_long_rolling_context_feature(col)
    if mode in {"no_rolling"}:
        return not is_long_rolling_context_feature(col)
    return True


def filter_feature_cols_for_mode(cols: List[str], mode: str = None) -> List[str]:
    return [col for col in cols if keep_feature_for_mode(col, mode)]


def make_frame_level_table(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {}
    for col in df.columns:
        if col in FRAME_KEYS:
            continue
        if col in ["source_file_name", "source_file_path", "original_mapped_file_index"]:
            agg_dict[col] = "first"
        if col.startswith("temp_") or col.startswith("feed_") or col.startswith("spindle_") or col.startswith("vib_") or col.startswith("motion_"):
            agg_dict[col] = "first"
        elif col in TARGET_COLS:
            agg_dict[col] = "mean"  # frame 级热趋势先学平均误差
    frame_df = df.groupby(FRAME_KEYS, as_index=False).agg(agg_dict)

    # 现在在 frame 级别做温度差分
    for ch in TEMP_CHANNELS:
        frame_df[f"d_temp_ch{ch}_median_c"] = frame_df.groupby("mapped_file_index")[f"temp_ch{ch}_median_c"].diff().fillna(0.0)
        frame_df[f"d_temp_ch{ch}_std_c"] = frame_df.groupby("mapped_file_index")[f"temp_ch{ch}_std_c"].diff().fillna(0.0)
    return frame_df


# =========================================================
# 4. Models
# =========================================================
def build_sequence_lstm(input_dim: int, time_steps: int, output_dim: int,
                        lstm_units: Tuple[int, int] = (64, 32), dense_units: int = 32,
                        dropout_rate: float = 0.2) -> tf.keras.Model:
    model = Sequential([
        LSTM(lstm_units[0], input_shape=(time_steps, input_dim), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units[1], return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(output_dim),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_geo_lstm(input_dim: int, time_steps: int) -> tf.keras.Model:
    return build_sequence_lstm(input_dim, time_steps, output_dim=3, lstm_units=(64, 32), dense_units=32, dropout_rate=0.2)


def build_thermal_lstm(input_dim: int, time_steps: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, input_shape=(time_steps, input_dim), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(64, activation="relu"),
        Dense(3),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=tf.keras.losses.Huber())
    return model


def build_res_lstm(input_dim: int, time_steps: int) -> tf.keras.Model:
    return build_sequence_lstm(input_dim, time_steps, output_dim=3, lstm_units=(64, 32), dense_units=32, dropout_rate=0.2)


def build_residual_mlp(input_dim: int) -> tf.keras.Model:
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(3),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber())
    return model


def build_all_in_one_mlp(input_dim: int) -> tf.keras.Model:
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(3),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(8e-4), loss=tf.keras.losses.Huber())
    return model


def build_spatiotemporal_model(
    static_dim: int,
    seq_dim: int,
    time_steps: int,
    target_scale: np.ndarray,
    condition_dim: int = 0,
) -> tf.keras.Model:
    static_input = Input(shape=(static_dim,), name="static_input")
    reg = regularizers.l2(L2_REG)

    static_branch = static_input
    for layer_idx, units in enumerate(STATIC_DENSE_UNITS):
        static_branch = Dense(int(units), activation="relu", kernel_regularizer=reg)(static_branch)
        if layer_idx < len(STATIC_DROPOUT_RATES):
            static_branch = BatchNormalization()(static_branch)
            static_branch = Dropout(float(STATIC_DROPOUT_RATES[layer_idx]))(static_branch)

    seq_input = Input(shape=(time_steps, seq_dim), name="seq_input")

    def build_recurrent_branch(backbone_name: str):
        temporal_layer = GRU if backbone_name.startswith("gru") else LSTM
        first_units = int(TEMPORAL_UNITS[0]) if len(TEMPORAL_UNITS) > 0 else 48
        second_units = int(TEMPORAL_UNITS[1]) if len(TEMPORAL_UNITS) > 1 else 24
        branch = temporal_layer(
            first_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
        )(seq_input)
        branch = Dropout(SEQ_DROPOUT_RATE)(branch)
        branch = temporal_layer(
            second_units,
            return_sequences=(backbone_name in {"gru_attention", "lstm_attention", "gru_tcn_attention", "lstm_tcn_attention"}),
            dropout=0.2,
            recurrent_dropout=0.1,
        )(branch)
        if backbone_name in {"gru_attention", "lstm_attention", "gru_tcn_attention", "lstm_tcn_attention"}:
            attention_branch = Attention(name=f"{backbone_name}_self_attention")([branch, branch])
            branch = Concatenate(name=f"{backbone_name}_attention_pool")([
                GlobalAveragePooling1D()(branch),
                GlobalAveragePooling1D()(attention_branch),
            ])
        return branch

    def build_tcn_branch(name_prefix: str = "tcn"):
        branch = Conv1D(32, kernel_size=1, padding="same", activation="relu", kernel_regularizer=reg, name=f"{name_prefix}_input_proj")(seq_input)
        for dilation_rate in [1, 2, 4]:
            residual = branch
            branch = Conv1D(
                32,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation_rate,
                activation="relu",
                kernel_regularizer=reg,
                name=f"{name_prefix}_d{dilation_rate}_conv1",
            )(branch)
            branch = Dropout(0.15, name=f"{name_prefix}_d{dilation_rate}_dropout")(branch)
            branch = Conv1D(
                32,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation_rate,
                activation="relu",
                kernel_regularizer=reg,
                name=f"{name_prefix}_d{dilation_rate}_conv2",
            )(branch)
            branch = Add(name=f"{name_prefix}_d{dilation_rate}_residual")([residual, branch])
        return GlobalAveragePooling1D(name=f"{name_prefix}_pool")(branch)

    def build_transformer_branch(name_prefix: str = "transformer"):
        model_dim = 48
        num_heads = 2
        ff_dim = 96
        branch = Dense(model_dim, activation="linear", kernel_regularizer=reg, name=f"{name_prefix}_input_proj")(seq_input)
        position = np.arange(time_steps, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, model_dim, 2, dtype=np.float32) * (-np.log(10000.0) / model_dim))
        pos_encoding = np.zeros((1, time_steps, model_dim), dtype=np.float32)
        pos_encoding[0, :, 0::2] = np.sin(position * div_term)
        pos_encoding[0, :, 1::2] = np.cos(position * div_term[:pos_encoding[0, :, 1::2].shape[1]])
        branch = Add(name=f"{name_prefix}_positional_add")([branch, tf.constant(pos_encoding)])

        for block_idx in range(2):
            attn_input = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_b{block_idx}_attn_norm")(branch)
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=model_dim // num_heads,
                dropout=0.10,
                name=f"{name_prefix}_b{block_idx}_mha",
            )(attn_input, attn_input)
            branch = Add(name=f"{name_prefix}_b{block_idx}_attn_residual")([branch, attn_output])
            ff_input = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_b{block_idx}_ff_norm")(branch)
            ff_output = Dense(ff_dim, activation="relu", kernel_regularizer=reg, name=f"{name_prefix}_b{block_idx}_ff1")(ff_input)
            ff_output = Dropout(0.10, name=f"{name_prefix}_b{block_idx}_ff_dropout")(ff_output)
            ff_output = Dense(model_dim, activation="linear", kernel_regularizer=reg, name=f"{name_prefix}_b{block_idx}_ff2")(ff_output)
            branch = Add(name=f"{name_prefix}_b{block_idx}_ff_residual")([branch, ff_output])

        branch = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_final_norm")(branch)
        return GlobalAveragePooling1D(name=f"{name_prefix}_pool")(branch)

    if TEMPORAL_BACKBONE in {"gru", "gru_attention", "lstm", "lstm_attention"}:
        seq_branch = build_recurrent_branch(TEMPORAL_BACKBONE)
    elif TEMPORAL_BACKBONE == "tcn":
        seq_branch = build_tcn_branch("tcn")
    elif TEMPORAL_BACKBONE == "transformer":
        seq_branch = build_transformer_branch("transformer")
    elif TEMPORAL_BACKBONE in {"gru_tcn_attention", "lstm_tcn_attention"}:
        recurrent_branch = build_recurrent_branch(TEMPORAL_BACKBONE)
        tcn_branch = build_tcn_branch("parallel_tcn")
        seq_branch = Concatenate(name="gru_tcn_temporal_fusion")([recurrent_branch, tcn_branch])
    else:
        raise ValueError(f"Unsupported TEMPORAL_BACKBONE: {TEMPORAL_BACKBONE}")
    seq_branch = Dense(SEQ_DENSE_UNITS, activation="relu", kernel_regularizer=reg)(seq_branch)

    fused = Concatenate()([static_branch, seq_branch])
    for layer_idx, units in enumerate(FUSED_DENSE_UNITS):
        fused = Dense(int(units), activation="relu", kernel_regularizer=reg)(fused)
        if layer_idx == 0 and FUSED_DROPOUT_RATE > 0:
            fused = Dropout(FUSED_DROPOUT_RATE)(fused)

    def build_axis_head(axis_name: str):
        axis_branch = Dense(AXIS_HEAD_UNITS, activation="relu", kernel_regularizer=reg, name=f"{axis_name}_head_dense")(fused)
        axis_branch = Dropout(0.05, name=f"{axis_name}_head_dropout")(axis_branch)
        return Dense(1, name=f"err_{axis_name}_output")(axis_branch)

    if USE_MOE_OUTPUT_HEAD:
        gate_source = fused
        condition_input = None
        if uses_condition_guided_moe():
            if condition_dim <= 0:
                raise ValueError("condition_dim must be positive when condition-guided MoE is enabled.")
            condition_input = Input(shape=(condition_dim,), name="condition_input")
            gate_source = Concatenate(name="moe_gate_condition_fusion")([condition_input, fused])

        expert_outputs = []
        for expert_idx in range(MOE_EXPERT_COUNT):
            expert_branch = fused
            for layer_idx, units in enumerate(MOE_EXPERT_HIDDEN_UNITS):
                expert_branch = Dense(
                    int(units),
                    activation="relu",
                    kernel_regularizer=reg,
                    name=f"moe_expert_{expert_idx}_dense_{layer_idx}",
                )(expert_branch)
                expert_branch = Dropout(0.05, name=f"moe_expert_{expert_idx}_dropout_{layer_idx}")(expert_branch)
            expert_outputs.append(Dense(len(TARGET_COLS), name=f"moe_expert_{expert_idx}_output")(expert_branch))
        gate_branch = gate_source
        for layer_idx, units in enumerate(MOE_GATE_HIDDEN_UNITS):
            gate_branch = Dense(
                int(units),
                activation="relu",
                kernel_regularizer=reg,
                name=f"moe_gate_dense_{layer_idx}",
            )(gate_branch)
        gate_output = Dense(MOE_EXPERT_COUNT, activation="softmax", name="moe_gate_output")(gate_branch)

        def weighted_expert_sum(inputs):
            gate = inputs[0]
            experts = tf.stack(inputs[1:], axis=1)
            return tf.reduce_sum(experts * tf.expand_dims(gate, axis=-1), axis=1)

        regression_output = Lambda(
            weighted_expert_sum,
            name="tool_error_output",
        )([gate_output] + expert_outputs)
    else:
        x_output = build_axis_head("x")
        y_output = build_axis_head("y")
        z_output = build_axis_head("z")
        regression_output = Concatenate(name="tool_error_output")([x_output, y_output, z_output])
    class_branch = Dense(24, activation="relu", kernel_regularizer=reg, name="error_class_dense")(fused)
    class_output = Dense(
        len(TARGET_COLS) * ERROR_CLASS_COUNT,
        name="error_class_logits",
    )(class_branch)
    class_output = Reshape((len(TARGET_COLS), ERROR_CLASS_COUNT), name="error_class_reshape")(class_output)
    class_output = Softmax(axis=-1, name="error_class_output")(class_output)

    outputs = [regression_output, class_output]
    losses = {
        "tool_error_output": make_position_aware_regression_loss(target_scale),
        "error_class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
    }
    loss_weights = {
        "tool_error_output": 1.0,
        "error_class_output": ERROR_CLASS_LOSS_WEIGHT,
    }
    metrics = {
        "error_class_output": tf.keras.metrics.SparseCategoricalAccuracy(name="class_acc"),
    }

    if has_auxiliary_heads():
        distance_branch = Dense(24, activation="relu", kernel_regularizer=reg, name="distance_aux_dense")(fused)
        distance_output = Dense(1, activation="softplus", name="distance_aux_output")(distance_branch)
        risk_branch = Dense(24, activation="relu", kernel_regularizer=reg, name="risk_aux_dense")(fused)
        risk_output = Dense(1, activation="sigmoid", name="risk_aux_output")(risk_branch)
        outputs.extend([distance_output, risk_output])
        losses["distance_aux_output"] = tf.keras.losses.Huber(delta=0.02)
        losses["risk_aux_output"] = tf.keras.losses.BinaryCrossentropy()
        loss_weights["distance_aux_output"] = DISTANCE_AUX_LOSS_WEIGHT
        loss_weights["risk_aux_output"] = RISK_AUX_LOSS_WEIGHT
        metrics["risk_aux_output"] = tf.keras.metrics.BinaryAccuracy(name="risk_acc")

    model_inputs = [static_input, seq_input]
    if USE_MOE_OUTPUT_HEAD and uses_condition_guided_moe():
        model_inputs.append(condition_input)
    model = Model(inputs=model_inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


# =========================================================
# 5. Main
# =========================================================
def main() -> None:
    configure_console_output()
    apply_cli_args()

    ensure_dir(OUTPUT_DIR)
    run_mode = RUN_MODE.strip().lower()
    if run_mode not in {"train", "predict_only"}:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE!r}. Use 'train' or 'predict_only'.")
    model_path = resolve_output_path(HYBRID_MODEL_PATH, HYBRID_MODEL_FILE)
    artifact_path = resolve_output_path(HYBRID_ARTIFACT_PATH, HYBRID_ARTIFACT_FILE)
    if run_mode == "predict_only" and HYBRID_MODEL_PATH and not HYBRID_ARTIFACT_PATH:
        inferred_artifact_path = infer_artifact_path_from_model(model_path)
        if inferred_artifact_path:
            artifact_path = inferred_artifact_path

    print("Loading data...")
    all_json_paths = collect_json_paths(JSON_DIR, JSON_PATH)
    json_paths = select_json_paths_for_prediction(all_json_paths)
    print(f"Found {len(json_paths)} JSON file(s) in: {JSON_DIR}")
    print("JSON files to merge:")
    for json_path in json_paths:
        print(f"  - {json_path}")
    print(f"USE_Y_AXIS_TARGETED_FEATURES = {USE_Y_AXIS_TARGETED_FEATURES}")
    print(f"USE_SOURCE_ONEHOT_FEATURES = {USE_SOURCE_ONEHOT_FEATURES}")
    print(f"USE_PHYSICAL_CONDITION_STATIC_FEATURES = {USE_PHYSICAL_CONDITION_STATIC_FEATURES}")
    print(f"USE_SOURCE_PROGRESS_FEATURES = {USE_SOURCE_PROGRESS_FEATURES}")
    print(f"USE_THERMAL_ACCUMULATION_FEATURES = {USE_THERMAL_ACCUMULATION_FEATURES}")
    print(f"FEATURE_SET_MODE = {FEATURE_SET_MODE}")
    print(f"THERMAL_ACCUMULATION_MODE = {THERMAL_ACCUMULATION_MODE}")
    print(f"USE_LONG_ROLLING_FEATURES = {USE_LONG_ROLLING_FEATURES}")
    print(f"LONG_ROLLING_WINDOWS = {LONG_ROLLING_WINDOWS}")
    print(f"USE_NEAR_BOUNDARY_RESIDUAL = {USE_NEAR_BOUNDARY_RESIDUAL}")
    print(
        f"USE_LATE_STAGE_RESIDUAL = {USE_LATE_STAGE_RESIDUAL}, "
        f"train_low={LATE_STAGE_RESIDUAL_TRAIN_LOW}, "
        f"apply={LATE_STAGE_RESIDUAL_APPLY_LOW}->{LATE_STAGE_RESIDUAL_APPLY_FULL}, "
        f"max_abs={LATE_STAGE_RESIDUAL_MAX_ABS}, "
        f"scales={LATE_STAGE_RESIDUAL_SCALE_CANDIDATES}"
    )
    print(f"TEMPORAL_BACKBONE = {TEMPORAL_BACKBONE}")
    print(f"SPLIT_MODE = {SPLIT_MODE}")
    print(f"PHASE_SPLIT_BLOCKS = {PHASE_SPLIT_BLOCKS}")
    print(f"RANDOM_CHUNK_SPLIT_COUNT = {RANDOM_CHUNK_SPLIT_COUNT}")
    print(f"RUN_MODE = {run_mode}")
    print(f"PREDICT_SOURCE_INDEX = {PREDICT_SOURCE_INDEX}, PREDICT_SOURCE_NAME = {PREDICT_SOURCE_NAME!r}")
    print(f"Hybrid model path = {model_path}")
    print(f"Hybrid artifact path = {artifact_path}")

    point_df = load_and_merge_datasets(json_paths)
    point_df = add_temperature_features(point_df)
    point_df["sin_pose_a"] = np.sin(np.deg2rad(point_df["pose_a"]))
    point_df["cos_pose_a"] = np.cos(np.deg2rad(point_df["pose_a"]))
    point_df["sin_pose_c"] = np.sin(np.deg2rad(point_df["pose_c"]))
    point_df["cos_pose_c"] = np.cos(np.deg2rad(point_df["pose_c"]))
    point_df["pose_r_xy"] = np.sqrt(point_df["pose_x"] ** 2 + point_df["pose_y"] ** 2)
    point_df["pose_r_xyz"] = np.sqrt(point_df["pose_x"] ** 2 + point_df["pose_y"] ** 2 + point_df["pose_z"] ** 2)
    point_df = add_group_diff_features(point_df, ["pose_x", "pose_y", "pose_z", "pose_a", "pose_c"], ["mapped_file_index"])
    point_df = add_group_diff_features(
        point_df,
        [f"temp_ch{ch}_median_c" for ch in TEMP_CHANNELS] + [f"temp_ch{ch}_std_c" for ch in TEMP_CHANNELS],
        ["mapped_file_index"],
    )
    point_df["temp_feed_interact"] = point_df["temp_median_mean"] * point_df["feed_speed"]
    point_df["temp_spindle_interact"] = point_df["temp_median_mean"] * point_df["spindle_speed"]
    point_df["pose_feed_interact"] = point_df["pose_r_xyz"] * point_df["feed_speed"]
    point_df["pose_spindle_interact"] = point_df["pose_r_xyz"] * point_df["spindle_speed"]
    if USE_SOURCE_PROGRESS_FEATURES:
        point_df = add_source_progress_features(point_df)
    if USE_Y_AXIS_TARGETED_FEATURES:
        point_df = add_y_axis_targeted_features(point_df)
    if USE_SOURCE_ONEHOT_FEATURES:
        point_df = add_source_onehot_features(point_df)
    frame_df = make_frame_level_table(point_df)
    if USE_SOURCE_PROGRESS_FEATURES:
        frame_df = add_source_progress_features(frame_df)
    if USE_THERMAL_ACCUMULATION_FEATURES:
        frame_df = add_thermal_accumulation_features(frame_df)
    if USE_LONG_ROLLING_FEATURES:
        frame_df = add_long_rolling_frame_features(frame_df)
    if USE_SOURCE_PROGRESS_FEATURES or USE_THERMAL_ACCUMULATION_FEATURES:
        point_df = merge_frame_context_to_points(point_df, frame_df)

    print(f"Point-level shape: {point_df.shape}")
    print(f"Frame-level shape: {frame_df.shape}")

    # ---------------- split data ----------------
    if SPLIT_MODE == "leave_one_json_out":
        train_frame, val_frame, test_frame = split_frame_dataset_leave_one_json_out(
            frame_df,
            test_json_index=TEST_JSON_INDEX,
            val_ratio=VAL_RATIO,
            group_col="source_file_name",
        )
    elif SPLIT_MODE == "phase_blocks":
        train_frame, val_frame, test_frame = split_frame_dataset_by_source(
            frame_df,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            group_col="source_file_name",
        )
    elif SPLIT_MODE == "random_phase_chunks":
        train_frame, val_frame, test_frame = split_frame_dataset_random_phase_chunks(
            frame_df,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            group_col="source_file_name",
            chunk_count=RANDOM_CHUNK_SPLIT_COUNT,
            phase_blocks=PHASE_SPLIT_BLOCKS,
            seed=RANDOM_SEED,
        )
    elif SPLIT_MODE == "random_frames":
        train_frame, val_frame, test_frame = split_frame_dataset_random_frames(
            frame_df,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            group_col="source_file_name",
            seed=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unsupported SPLIT_MODE: {SPLIT_MODE}")

    test_frame = filter_test_frame_by_source(
        test_frame,
        frame_df,
        source_name=PREDICT_SOURCE_NAME,
        source_index=PREDICT_SOURCE_INDEX,
        group_col="source_file_name",
    )

    print(f"Frame split -> Train: {train_frame.shape}, Val: {val_frame.shape}, Test: {test_frame.shape}")
    print_split_frame_overlap(train_frame, val_frame, test_frame)

    train_points_raw = select_points_by_frames(point_df, train_frame)
    val_points_raw = select_points_by_frames(point_df, val_frame)
    test_points_raw = select_points_by_frames(point_df, test_frame)
    print(f"Point split -> Train: {train_points_raw.shape}, Val: {val_points_raw.shape}, Test: {test_points_raw.shape}")
    print_point_split_by_source(train_points_raw, val_points_raw, test_points_raw)

    static_feature_cols = [
        "pose_x",
        "pose_y",
        "pose_z",
        "sin_pose_a",
        "cos_pose_a",
        "sin_pose_c",
        "cos_pose_c",
        "pose_r_xy",
        "pose_r_xyz",
        "d_pose_x",
        "d_pose_y",
        "d_pose_z",
        "d_pose_a",
        "d_pose_c",
    ]
    static_feature_cols.extend([
        col for col in point_df.columns
        if (
            col.startswith("source_line_progress")
            or col.startswith("source_frame_progress")
            or col.startswith("source_line_frame_progress_diff")
            or col.startswith("progress_")
            or "_roll" in col
        )
        and pd.api.types.is_numeric_dtype(point_df[col])
    ])
    if USE_PHYSICAL_CONDITION_STATIC_FEATURES or uses_condition_guided_moe():
        static_feature_cols.extend([
            col for col in PHYSICAL_CONDITION_FEATURE_CANDIDATES
            if col in point_df.columns and pd.api.types.is_numeric_dtype(point_df[col])
        ])
    if USE_Y_AXIS_TARGETED_FEATURES:
        static_feature_cols.extend([
            "pose_x_abs",
            "pose_y_abs",
            "pose_z_abs",
            "pose_x_sign",
            "pose_y_sign",
            "pose_z_sign",
            "motion_acc_xy_norm",
            "motion_acc_x_sign",
            "motion_acc_y_sign",
            "pose_y_acc_y_interact",
            "pose_x_acc_y_interact",
            "pose_z_acc_y_interact",
            "pose_y_acc_x_interact",
        ])
    if USE_SOURCE_ONEHOT_FEATURES:
        static_feature_cols.extend(sorted([col for col in point_df.columns if col.startswith(SOURCE_ONEHOT_PREFIX)]))
    static_feature_cols = unique_preserve_order([col for col in static_feature_cols if col in point_df.columns])
    static_feature_cols = filter_feature_cols_for_mode(static_feature_cols)

    state_exclude_cols = set(FRAME_KEYS + TARGET_COLS + ["source_file_name", "source_file_path", "original_mapped_file_index"])
    if USE_SOURCE_ONEHOT_FEATURES:
        state_exclude_cols.update([col for col in frame_df.columns if col.startswith(SOURCE_ONEHOT_PREFIX)])
    state_feature_cols = [
        col for col in frame_df.columns
        if col not in state_exclude_cols and pd.api.types.is_numeric_dtype(frame_df[col])
    ]
    state_feature_cols = sorted(filter_feature_cols_for_mode(state_feature_cols))

    saved_artifacts = None
    if run_mode == "predict_only":
        saved_artifacts = load_pickle_artifact(artifact_path)
        saved_time_steps = int(saved_artifacts.get("time_steps", TIME_STEPS))
        saved_backbone = saved_artifacts.get("temporal_backbone", TEMPORAL_BACKBONE)
        if saved_time_steps != TIME_STEPS:
            raise ValueError(
                f"Saved model was trained with TIME_STEPS={saved_time_steps}, "
                f"but current TIME_STEPS={TIME_STEPS}."
            )
        if saved_backbone != TEMPORAL_BACKBONE:
            raise ValueError(
                f"Saved model was trained with TEMPORAL_BACKBONE={saved_backbone!r}, "
                f"but current TEMPORAL_BACKBONE={TEMPORAL_BACKBONE!r}."
            )
        static_feature_cols = saved_artifacts["static_feature_cols"]
        state_feature_cols = saved_artifacts["state_feature_cols"]
        validate_required_columns(point_df, static_feature_cols, "point_df")
        validate_required_columns(frame_df, state_feature_cols, "frame_df")

    if saved_artifacts is not None and "condition_feature_cols" in saved_artifacts:
        condition_feature_cols = saved_artifacts["condition_feature_cols"]
    else:
        condition_feature_cols = [
            col for col in MOE_CONDITION_FEATURE_CANDIDATES
            if col in static_feature_cols and not col.startswith(SOURCE_ONEHOT_PREFIX)
        ]
    if uses_condition_guided_moe() and not condition_feature_cols:
        raise ValueError("Condition-guided MoE is enabled, but no condition features were found.")

    print(f"\nStatic feature count: {len(static_feature_cols)}")
    print(f"State sequence feature count: {len(state_feature_cols)}")
    if uses_condition_guided_moe():
        print(f"MoE condition feature count: {len(condition_feature_cols)} -> {condition_feature_cols}")

    frame_scaler = saved_artifacts["frame_scaler"] if saved_artifacts is not None else StandardScaler()
    train_frame_s = (
        frame_scaler.transform(train_frame[state_feature_cols].values)
        if saved_artifacts is not None
        else frame_scaler.fit_transform(train_frame[state_feature_cols].values)
    )
    val_frame_s = frame_scaler.transform(val_frame[state_feature_cols].values)
    test_frame_s = frame_scaler.transform(test_frame[state_feature_cols].values)

    static_scaler = saved_artifacts["static_scaler"] if saved_artifacts is not None else StandardScaler()
    train_static_s = (
        static_scaler.transform(train_points_raw[static_feature_cols].values)
        if saved_artifacts is not None
        else static_scaler.fit_transform(train_points_raw[static_feature_cols].values)
    )
    val_static_s = static_scaler.transform(val_points_raw[static_feature_cols].values)
    test_static_s = static_scaler.transform(test_points_raw[static_feature_cols].values)

    target_scaler = saved_artifacts["target_scaler"] if saved_artifacts is not None else StandardScaler()
    train_y_s = (
        target_scaler.transform(train_points_raw[TARGET_COLS].values)
        if saved_artifacts is not None
        else target_scaler.fit_transform(train_points_raw[TARGET_COLS].values)
    )
    val_y_s = target_scaler.transform(val_points_raw[TARGET_COLS].values)
    test_y_s = target_scaler.transform(test_points_raw[TARGET_COLS].values)

    error_class_edges = (
        saved_artifacts["error_class_edges"]
        if saved_artifacts is not None
        else fit_error_class_edges(target_scaler)
    )
    print_error_class_edges(error_class_edges)
    train_points_raw = add_error_class_columns(train_points_raw, error_class_edges, train_y_s)
    val_points_raw = add_error_class_columns(val_points_raw, error_class_edges, val_y_s)
    test_points_raw = add_error_class_columns(test_points_raw, error_class_edges, test_y_s)

    train_frame_seq = train_frame.copy()
    val_frame_seq = val_frame.copy()
    test_frame_seq = test_frame.copy()
    train_frame_seq.loc[:, state_feature_cols] = train_frame_seq[state_feature_cols].astype(np.float32)
    val_frame_seq.loc[:, state_feature_cols] = val_frame_seq[state_feature_cols].astype(np.float32)
    test_frame_seq.loc[:, state_feature_cols] = test_frame_seq[state_feature_cols].astype(np.float32)
    assign_float32_columns(train_frame_seq, state_feature_cols, train_frame_s)
    assign_float32_columns(val_frame_seq, state_feature_cols, val_frame_s)
    assign_float32_columns(test_frame_seq, state_feature_cols, test_frame_s)

    train_points_seq = train_points_raw.copy()
    val_points_seq = val_points_raw.copy()
    test_points_seq = test_points_raw.copy()
    train_points_seq.loc[:, static_feature_cols] = train_points_seq[static_feature_cols].astype(np.float32)
    val_points_seq.loc[:, static_feature_cols] = val_points_seq[static_feature_cols].astype(np.float32)
    test_points_seq.loc[:, static_feature_cols] = test_points_seq[static_feature_cols].astype(np.float32)
    train_points_seq.loc[:, TARGET_COLS] = train_points_seq[TARGET_COLS].astype(np.float32)
    val_points_seq.loc[:, TARGET_COLS] = val_points_seq[TARGET_COLS].astype(np.float32)
    test_points_seq.loc[:, TARGET_COLS] = test_points_seq[TARGET_COLS].astype(np.float32)
    assign_float32_columns(train_points_seq, static_feature_cols, train_static_s)
    assign_float32_columns(val_points_seq, static_feature_cols, val_static_s)
    assign_float32_columns(test_points_seq, static_feature_cols, test_static_s)
    assign_float32_columns(train_points_seq, TARGET_COLS, train_y_s)
    assign_float32_columns(val_points_seq, TARGET_COLS, val_y_s)
    assign_float32_columns(test_points_seq, TARGET_COLS, test_y_s)

    frame_group_cols = ["source_file_name"]
    train_frame_X_seq, train_frame_meta = build_grouped_feature_sequences(
        train_frame_seq, state_feature_cols, frame_group_cols, TIME_STEPS
    )
    val_frame_X_seq, val_frame_meta = build_grouped_feature_sequences(
        val_frame_seq, state_feature_cols, frame_group_cols, TIME_STEPS
    )
    test_frame_X_seq, test_frame_meta = build_grouped_feature_sequences(
        test_frame_seq, state_feature_cols, frame_group_cols, TIME_STEPS
    )

    print(f"Train frame sequences: {train_frame_X_seq.shape}")
    print(f"Val frame sequences:   {val_frame_X_seq.shape}")
    print(f"Test frame sequences:  {test_frame_X_seq.shape}")

    train_static_X, train_seq_X, train_y, train_class_y, train_meta = build_hybrid_samples(
        train_frame_meta, train_frame_X_seq, train_points_seq, static_feature_cols, TARGET_COLS, ERROR_CLASS_COLS
    )
    val_static_X, val_seq_X, val_y, val_class_y, val_meta = build_hybrid_samples(
        val_frame_meta, val_frame_X_seq, val_points_seq, static_feature_cols, TARGET_COLS, ERROR_CLASS_COLS
    )
    test_static_X, test_seq_X, test_y, test_class_y, test_meta = build_hybrid_samples(
        test_frame_meta, test_frame_X_seq, test_points_seq, static_feature_cols, TARGET_COLS, ERROR_CLASS_COLS
    )
    condition_feature_indices = [static_feature_cols.index(col) for col in condition_feature_cols]
    train_condition_X = train_static_X[:, condition_feature_indices].astype(np.float32) if condition_feature_indices else np.empty((len(train_static_X), 0), dtype=np.float32)
    val_condition_X = val_static_X[:, condition_feature_indices].astype(np.float32) if condition_feature_indices else np.empty((len(val_static_X), 0), dtype=np.float32)
    test_condition_X = test_static_X[:, condition_feature_indices].astype(np.float32) if condition_feature_indices else np.empty((len(test_static_X), 0), dtype=np.float32)

    print(f"Train hybrid samples -> static: {train_static_X.shape}, seq: {train_seq_X.shape}, y: {train_y.shape}")
    print(f"Val hybrid samples   -> static: {val_static_X.shape}, seq: {val_seq_X.shape}, y: {val_y.shape}")
    print(f"Test hybrid samples  -> static: {test_static_X.shape}, seq: {test_seq_X.shape}, y: {test_y.shape}")
    if uses_condition_guided_moe():
        print(f"MoE condition samples -> train: {train_condition_X.shape}, val: {val_condition_X.shape}, test: {test_condition_X.shape}")
    print(f"Error-class samples  -> train: {train_class_y.shape}, val: {val_class_y.shape}, test: {test_class_y.shape}")

    train_distance_y, train_risk_y = build_auxiliary_targets(train_y, target_scaler)
    val_distance_y, val_risk_y = build_auxiliary_targets(val_y, target_scaler)
    test_distance_y, test_risk_y = build_auxiliary_targets(test_y, target_scaler)
    if has_auxiliary_heads():
        print(
            "Auxiliary targets -> "
            f"train_distance_mean={float(np.mean(train_distance_y)):.6f}, "
            f"val_distance_mean={float(np.mean(val_distance_y)):.6f}, "
            f"test_distance_mean={float(np.mean(test_distance_y)):.6f}, "
            f"train_risk_rate={float(np.mean(train_risk_y)):.6f}, "
            f"val_risk_rate={float(np.mean(val_risk_y)):.6f}"
        )

    train_reg_weights = build_high_value_regression_weights(train_class_y)
    val_reg_weights = build_high_value_regression_weights(val_class_y)
    train_class_weights = np.ones(train_class_y.shape, dtype=np.float32)
    val_class_weights = np.ones(val_class_y.shape, dtype=np.float32)
    train_sample_weights = make_model_sample_weights(train_reg_weights, train_class_weights, train_distance_y, train_risk_y)
    val_sample_weights = make_model_sample_weights(val_reg_weights, val_class_weights, val_distance_y, val_risk_y)
    print_high_value_weight_summary(train_class_y, train_reg_weights, "Train")
    print_high_value_weight_summary(val_class_y, val_reg_weights, "Val")

    if len(train_static_X) == 0 or len(val_static_X) == 0 or len(test_static_X) == 0:
        raise ValueError("Not enough samples to build hybrid spatiotemporal data. Reduce TIME_STEPS or check frame continuity.")

    train_diag_df = train_meta.merge(
        train_points_raw,
        on=FRAME_KEYS + ["point_index"],
        how="left",
        suffixes=("_meta", ""),
    )
    val_diag_df = val_meta.merge(
        val_points_raw,
        on=FRAME_KEYS + ["point_index"],
        how="left",
        suffixes=("_meta", ""),
    )
    test_diag_df = test_meta.merge(
        test_points_raw,
        on=FRAME_KEYS + ["point_index"],
        how="left",
        suffixes=("_meta", ""),
    )

    if run_mode == "train":
        save_pickle_artifact(
            {
                "static_feature_cols": static_feature_cols,
                "condition_feature_cols": condition_feature_cols,
                "state_feature_cols": state_feature_cols,
                "frame_scaler": frame_scaler,
                "static_scaler": static_scaler,
                "target_scaler": target_scaler,
                "error_class_edges": error_class_edges,
                "time_steps": TIME_STEPS,
                "temporal_backbone": TEMPORAL_BACKBONE,
                "feature_set_mode": FEATURE_SET_MODE,
                "thermal_accumulation_mode": THERMAL_ACCUMULATION_MODE,
            },
            artifact_path,
        )
        print(f"\nSaved preprocessing artifact to: {artifact_path}")

        print("\n================ Training spatial + temporal hybrid model ================")
        model = build_spatiotemporal_model(
            static_dim=train_static_X.shape[1],
            seq_dim=train_seq_X.shape[2],
            time_steps=TIME_STEPS,
            target_scale=target_scaler.scale_,
            condition_dim=train_condition_X.shape[1],
        )
        es = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        )
        position_ratio_cb = BestPositionDistanceRatioCallback(
            val_inputs=make_model_inputs(val_static_X, val_seq_X, val_condition_X),
            val_y_scaled=val_y,
            val_meta_df=val_diag_df,
            target_scaler=target_scaler,
        )
        hist = model.fit(
            make_model_inputs(train_static_X, train_seq_X, train_condition_X),
            make_model_targets(train_y, train_class_y, train_distance_y, train_risk_y),
            validation_data=(
                make_model_inputs(val_static_X, val_seq_X, val_condition_X),
                make_model_targets(val_y, val_class_y, val_distance_y, val_risk_y),
                val_sample_weights,
            ),
            sample_weight=train_sample_weights,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            callbacks=[reduce_lr, es, position_ratio_cb],
        )
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        best_val_loss = float(np.min(hist.history["val_loss"]))
        final_train_loss = float(hist.history["loss"][best_epoch - 1])
        print(
            f"Best validation epoch: {best_epoch}, "
            f"train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}"
        )
        print(
            f"Best validation position-distance ratio epoch: {position_ratio_cb.best_epoch}, "
            f"val_position_distance_ratio={position_ratio_cb.best_ratio:.6f}"
        )
        save_history_plot(hist, "Spatial + Temporal Hybrid Loss", os.path.join(OUTPUT_DIR, "hybrid_spatiotemporal_loss_v10.png"), SHOW_PLOTS)

        if USE_NEAR_THRESHOLD_REWEIGHTING and NEAR_THRESHOLD_FINE_TUNE_EPOCHS > 0:
            print("\n================ Near-threshold weighted fine-tuning ================")
            if NEAR_THRESHOLD_FINE_TUNE_LR > 0:
                try:
                    model.optimizer.learning_rate.assign(NEAR_THRESHOLD_FINE_TUNE_LR)
                except AttributeError:
                    tf.keras.backend.set_value(model.optimizer.learning_rate, NEAR_THRESHOLD_FINE_TUNE_LR)
                print(f"Near-threshold fine-tune learning rate reset to: {NEAR_THRESHOLD_FINE_TUNE_LR:g}")

            train_pred_scaled_ft, _, _, _ = split_model_predictions(model.predict(
                make_model_inputs(train_static_X, train_seq_X, train_condition_X),
                verbose=0,
            ))
            val_pred_scaled_ft, _, _, _ = split_model_predictions(model.predict(
                make_model_inputs(val_static_X, val_seq_X, val_condition_X),
                verbose=0,
            ))
            train_pred_ft = target_scaler.inverse_transform(train_pred_scaled_ft)
            val_pred_ft = target_scaler.inverse_transform(val_pred_scaled_ft)
            y_train_true_ft = target_scaler.inverse_transform(train_y)
            y_val_true_ft = target_scaler.inverse_transform(val_y)

            train_near_weights = build_near_threshold_regression_weights(
                y_train_true_ft,
                train_pred_ft,
                "Train",
            )
            val_near_weights = build_near_threshold_regression_weights(
                y_val_true_ft,
                val_pred_ft,
                "Val",
            )
            train_reg_weights_ft = (train_reg_weights * train_near_weights).astype(np.float32)
            val_reg_weights_ft = (val_reg_weights * val_near_weights).astype(np.float32)
            train_sample_weights_ft = make_model_sample_weights(train_reg_weights_ft, train_class_weights, train_distance_y, train_risk_y)
            val_sample_weights_ft = make_model_sample_weights(val_reg_weights_ft, val_class_weights, val_distance_y, val_risk_y)

            fine_es = EarlyStopping(
                monitor="val_loss",
                patience=NEAR_THRESHOLD_FINE_TUNE_PATIENCE,
                min_delta=EARLY_STOP_MIN_DELTA,
                restore_best_weights=True,
            )
            fine_reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=max(4, REDUCE_LR_PATIENCE // 2),
                min_lr=MIN_LR,
                verbose=1,
            )
            fine_position_ratio_cb = BestPositionDistanceRatioCallback(
                val_inputs=make_model_inputs(val_static_X, val_seq_X, val_condition_X),
                val_y_scaled=val_y,
                val_meta_df=val_diag_df,
                target_scaler=target_scaler,
            )
            fine_hist = model.fit(
                make_model_inputs(train_static_X, train_seq_X, train_condition_X),
                make_model_targets(train_y, train_class_y, train_distance_y, train_risk_y),
                validation_data=(
                    make_model_inputs(val_static_X, val_seq_X, val_condition_X),
                    make_model_targets(val_y, val_class_y, val_distance_y, val_risk_y),
                    val_sample_weights_ft,
                ),
                sample_weight=train_sample_weights_ft,
                epochs=NEAR_THRESHOLD_FINE_TUNE_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=VERBOSE,
                callbacks=[fine_reduce_lr, fine_es, fine_position_ratio_cb],
            )
            fine_best_epoch = int(np.argmin(fine_hist.history["val_loss"])) + 1
            fine_best_val_loss = float(np.min(fine_hist.history["val_loss"]))
            print(
                f"Near-threshold fine-tune best validation epoch: {fine_best_epoch}, "
                f"val_loss={fine_best_val_loss:.6f}"
            )
            print(
                f"Near-threshold fine-tune best position-distance ratio epoch: "
                f"{fine_position_ratio_cb.best_epoch}, "
                f"val_position_distance_ratio={fine_position_ratio_cb.best_ratio:.6f}"
            )
            save_history_plot(
                fine_hist,
                "Near-threshold Weighted Fine-tune Loss",
                os.path.join(OUTPUT_DIR, "hybrid_spatiotemporal_near_threshold_finetune_loss_v10.png"),
                SHOW_PLOTS,
            )

        model.save(model_path)
        print(f"Saved trained Hybrid model to: {model_path}")
    else:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Saved Hybrid model not found: {model_path}")
        print("正在导入已训练模型...")
        model = load_keras_model_compatible(
            model_path,
            custom_objects={"elementwise_huber_loss": elementwise_huber_loss},
            compile=False,
        )
        print(f"模型导入完成: {model_path}")
        print("导入的加工过程 JSON 文件:")
        for json_idx, json_path in enumerate(SELECTED_JSON_PATHS_FOR_OUTPUT):
            print(f"导入文件 - {source_display_label(os.path.basename(json_path), json_idx)}: {os.path.basename(json_path)}")

    print("正在执行测试阶段预测...")
    train_pred_scaled, train_class_pred, train_distance_pred, train_risk_pred = split_model_predictions(
        model.predict(make_model_inputs(train_static_X, train_seq_X, train_condition_X), verbose=0)
    )
    val_pred_scaled, val_class_pred, val_distance_pred, val_risk_pred = split_model_predictions(
        model.predict(make_model_inputs(val_static_X, val_seq_X, val_condition_X), verbose=0)
    )
    test_pred_scaled, test_class_pred, test_distance_pred, test_risk_pred = split_model_predictions(
        model.predict(make_model_inputs(test_static_X, test_seq_X, test_condition_X), verbose=0)
    )

    train_pred = target_scaler.inverse_transform(
        train_pred_scaled
    )
    val_pred = target_scaler.inverse_transform(
        val_pred_scaled
    )
    test_pred = target_scaler.inverse_transform(
        test_pred_scaled
    )
    test_moe_gate_pred = None
    if USE_MOE_OUTPUT_HEAD and any(layer.name == "moe_gate_output" for layer in model.layers):
        gate_probe_model = Model(inputs=model.inputs, outputs=model.get_layer("moe_gate_output").output)
        test_moe_gate_pred = gate_probe_model.predict(
            make_model_inputs(test_static_X, test_seq_X, test_condition_X),
            verbose=0,
        )

    y_train_true = target_scaler.inverse_transform(train_y)
    y_val_true = target_scaler.inverse_transform(val_y)
    y_test_true = target_scaler.inverse_transform(test_y)

    print("正在统计测试结果...")
    evaluate_error_classification(train_y, train_class_pred, error_class_edges, "Hybrid Train")
    evaluate_error_classification(val_y, val_class_pred, error_class_edges, "Hybrid Val")
    evaluate_error_classification(test_y, test_class_pred, error_class_edges, "Hybrid Test")

    train_diag_df = train_meta.merge(
        train_points_raw,
        on=FRAME_KEYS + ["point_index"],
        how="left",
        suffixes=("_meta", ""),
    )
    val_diag_df = val_meta.merge(
        val_points_raw,
        on=FRAME_KEYS + ["point_index"],
        how="left",
        suffixes=("_meta", ""),
    )
    test_diag_df = test_meta.merge(
        test_points_raw,
        on=FRAME_KEYS + ["point_index"],
        how="left",
        suffixes=("_meta", ""),
    )

    pre_cal_train_pred = train_pred.copy()
    pre_cal_val_pred = val_pred.copy()
    pre_cal_test_pred = test_pred.copy()

    if len(train_diag_df) == len(train_meta) and len(val_diag_df) == len(val_meta) and len(test_diag_df) == len(test_meta):
        print("\n================ Validation-driven XYZ calibration ================")
        evaluate_multioutput(y_train_true, pre_cal_train_pred, "Hybrid Train (Before Calibration)")
        evaluate_multioutput(y_val_true, pre_cal_val_pred, "Hybrid Val (Before Calibration)")
        evaluate_multioutput(y_test_true, pre_cal_test_pred, "Hybrid Test (Before Calibration)")
        print_actual_position_distance_summary(train_diag_df, y_train_true, pre_cal_train_pred, "Hybrid Train (Before Calibration)")
        print_actual_position_distance_summary(val_diag_df, y_val_true, pre_cal_val_pred, "Hybrid Val (Before Calibration)")
        print_actual_position_distance_summary(test_diag_df, y_test_true, pre_cal_test_pred, "Hybrid Test (Before Calibration)")
        print_error_bucket_summary(y_test_true, pre_cal_test_pred, "Hybrid Test (Before Calibration)")

        axis_slopes, axis_intercepts = fit_global_axis_calibrators(y_val_true, pre_cal_val_pred)
        global_train_pred = apply_global_axis_calibrators(pre_cal_train_pred, axis_slopes, axis_intercepts)
        global_val_pred = apply_global_axis_calibrators(pre_cal_val_pred, axis_slopes, axis_intercepts)
        global_test_pred = apply_global_axis_calibrators(pre_cal_test_pred, axis_slopes, axis_intercepts)
        print(
            "Global XYZ calibration: "
            f"X={axis_slopes[0]:.6f}*pred+{axis_intercepts[0]:.6f}, "
            f"Y={axis_slopes[1]:.6f}*pred+{axis_intercepts[1]:.6f}, "
            f"Z={axis_slopes[2]:.6f}*pred+{axis_intercepts[2]:.6f}"
        )
        evaluate_multioutput(y_val_true, global_val_pred, "Hybrid Val (Global XYZ Calibration)")
        evaluate_multioutput(y_test_true, global_test_pred, "Hybrid Test (Global XYZ Calibration)")
        print_actual_position_distance_summary(val_diag_df, y_val_true, global_val_pred, "Hybrid Val (Global XYZ Calibration)")
        print_actual_position_distance_summary(test_diag_df, y_test_true, global_test_pred, "Hybrid Test (Global XYZ Calibration)")
        print_error_bucket_summary(y_test_true, global_test_pred, "Hybrid Test (Global XYZ Calibration)")

        calibration_candidates = [
            ("none", pre_cal_train_pred, pre_cal_val_pred, pre_cal_test_pred),
            ("global_xyz", global_train_pred, global_val_pred, global_test_pred),
        ]

        y_slope, y_intercept = fit_global_y_calibrator(y_val_true, pre_cal_val_pred)
        global_y_train_pred = apply_global_y_calibrator(pre_cal_train_pred, y_slope, y_intercept)
        global_y_val_pred = apply_global_y_calibrator(pre_cal_val_pred, y_slope, y_intercept)
        global_y_test_pred = apply_global_y_calibrator(pre_cal_test_pred, y_slope, y_intercept)
        print(f"Global Y-only calibration: Y={y_slope:.6f}*pred+{y_intercept:.6f}")
        evaluate_multioutput(y_val_true, global_y_val_pred, "Hybrid Val (Global Y Calibration)")
        evaluate_multioutput(y_test_true, global_y_test_pred, "Hybrid Test (Global Y Calibration)")
        print_actual_position_distance_summary(val_diag_df, y_val_true, global_y_val_pred, "Hybrid Val (Global Y Calibration)")
        print_actual_position_distance_summary(test_diag_df, y_test_true, global_y_test_pred, "Hybrid Test (Global Y Calibration)")
        print_error_bucket_summary(y_test_true, global_y_test_pred, "Hybrid Test (Global Y Calibration)")
        calibration_candidates.append(("global_y", global_y_train_pred, global_y_val_pred, global_y_test_pred))

        source_axis_train_pred, source_axis_val_pred, source_axis_test_pred = (
            global_train_pred,
            global_val_pred,
            global_test_pred,
        )
        source_y_train_pred, source_y_val_pred, source_y_test_pred = (
            global_y_train_pred,
            global_y_val_pred,
            global_y_test_pred,
        )
        if USE_SOURCE_OFFSET_CALIBRATION:
            source_axis_offsets = fit_group_axis_offset_calibrator(
                val_diag_df,
                y_val_true,
                global_val_pred,
                group_col="source_file_name",
                min_group_samples=8,
                max_abs_offset=0.03,
                shrinkage=16.0,
            )
            source_axis_train_pred = apply_group_axis_offset_calibrator(
                train_diag_df, global_train_pred, "source_file_name", source_axis_offsets
            )
            source_axis_val_pred = apply_group_axis_offset_calibrator(
                val_diag_df, global_val_pred, "source_file_name", source_axis_offsets
            )
            source_axis_test_pred = apply_group_axis_offset_calibrator(
                test_diag_df, global_test_pred, "source_file_name", source_axis_offsets
            )
            print(f"Source XYZ offset calibration: groups={len(source_axis_offsets)}")
            evaluate_multioutput(y_val_true, source_axis_val_pred, "Hybrid Val (Source XYZ Offset)")
            evaluate_multioutput(y_test_true, source_axis_test_pred, "Hybrid Test (Source XYZ Offset)")
            print_actual_position_distance_summary(val_diag_df, y_val_true, source_axis_val_pred, "Hybrid Val (Source XYZ Offset)")
            print_actual_position_distance_summary(test_diag_df, y_test_true, source_axis_test_pred, "Hybrid Test (Source XYZ Offset)")
            print_error_bucket_summary(y_test_true, source_axis_test_pred, "Hybrid Test (Source XYZ Offset)")
            calibration_candidates.append(("source_xyz_offset", source_axis_train_pred, source_axis_val_pred, source_axis_test_pred))

            source_y_offsets = fit_group_axis_offset_calibrator(
                val_diag_df,
                y_val_true,
                global_y_val_pred,
                group_col="source_file_name",
                min_group_samples=8,
                max_abs_offset=0.03,
                shrinkage=16.0,
                axis_idx=1,
            )
            source_y_train_pred = apply_group_axis_offset_calibrator(
                train_diag_df, global_y_train_pred, "source_file_name", source_y_offsets
            )
            source_y_val_pred = apply_group_axis_offset_calibrator(
                val_diag_df, global_y_val_pred, "source_file_name", source_y_offsets
            )
            source_y_test_pred = apply_group_axis_offset_calibrator(
                test_diag_df, global_y_test_pred, "source_file_name", source_y_offsets
            )
            print(f"Source Y-only offset calibration: groups={len(source_y_offsets)}")
            evaluate_multioutput(y_val_true, source_y_val_pred, "Hybrid Val (Source Y Offset)")
            evaluate_multioutput(y_test_true, source_y_test_pred, "Hybrid Test (Source Y Offset)")
            print_actual_position_distance_summary(val_diag_df, y_val_true, source_y_val_pred, "Hybrid Val (Source Y Offset)")
            print_actual_position_distance_summary(test_diag_df, y_test_true, source_y_test_pred, "Hybrid Test (Source Y Offset)")
            print_error_bucket_summary(y_test_true, source_y_test_pred, "Hybrid Test (Source Y Offset)")
            calibration_candidates.append(("source_y_offset", source_y_train_pred, source_y_val_pred, source_y_test_pred))

        source_region_test_preds = {}
        source_region_specs = []
        if USE_SOURCE_REGION_CALIBRATION:
            source_region_specs = [
                ("source_region_pose_y_low_xyz", "pose_y", 0.25, "low", None),
                ("source_region_pose_y_low_y", "pose_y", 0.25, "low", 1),
                ("source_region_pose_r_low_xyz", "pose_r_xyz", 0.25, "low", None),
                ("source_region_pose_r_high_xyz", "pose_r_xyz", 0.75, "high", None),
                ("source_region_pose_r_high_y", "pose_r_xyz", 0.75, "high", 1),
                ("source_region_pose_r_high_z", "pose_r_xyz", 0.75, "high", 2),
                ("source_region_temp_high_xyz", "temp_median_mean", 0.75, "high", None),
            ]
        for mode_name, col, quantile, tail, axis_idx in source_region_specs:
            threshold, offsets = fit_group_region_axis_offset_calibrator(
                val_diag_df,
                y_val_true,
                global_val_pred,
                group_col="source_file_name",
                value_col=col,
                quantile=quantile,
                tail=tail,
                min_region_samples=5,
                max_abs_offset=0.025,
                shrinkage=12.0,
                axis_idx=axis_idx,
            )
            cur_train_pred = apply_group_region_axis_offset_calibrator(
                train_diag_df, global_train_pred, "source_file_name", col, threshold, tail, offsets
            )
            cur_val_pred = apply_group_region_axis_offset_calibrator(
                val_diag_df, global_val_pred, "source_file_name", col, threshold, tail, offsets
            )
            cur_test_pred = apply_group_region_axis_offset_calibrator(
                test_diag_df, global_test_pred, "source_file_name", col, threshold, tail, offsets
            )
            axis_text = "XYZ" if axis_idx is None else ["X", "Y", "Z"][axis_idx]
            print(
                f"{mode_name}: col={col}, tail={tail}, threshold={threshold:.6f}, "
                f"axis={axis_text}, groups={len(offsets)}"
            )
            evaluate_multioutput(y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
            evaluate_multioutput(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            print_actual_position_distance_summary(val_diag_df, y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
            print_actual_position_distance_summary(test_diag_df, y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            print_error_bucket_summary(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            calibration_candidates.append((mode_name, cur_train_pred, cur_val_pred, cur_test_pred))
            source_region_test_preds[mode_name] = cur_test_pred

        tail_calibrated_test_preds = {}
        tail_calibration_specs = [
            ("global_xyz_pose_r_high", "pose_r_xyz", 0.75, "high"),
            ("global_xyz_feed_high", "feed_speed", 0.75, "high"),
            ("global_xyz_feed_low", "feed_speed", 0.25, "low"),
        ]
        for mode_name, col, quantile, tail in tail_calibration_specs:
            threshold, offsets = fit_tail_axis_offset_calibrator(
                val_diag_df,
                y_val_true,
                global_val_pred,
                col=col,
                quantile=quantile,
                tail=tail,
                max_abs_offset=0.02,
            )
            cur_train_pred = apply_tail_axis_offset_calibrator(
                train_diag_df, global_train_pred, col, threshold, tail, offsets
            )
            cur_val_pred = apply_tail_axis_offset_calibrator(
                val_diag_df, global_val_pred, col, threshold, tail, offsets
            )
            cur_test_pred = apply_tail_axis_offset_calibrator(
                test_diag_df, global_test_pred, col, threshold, tail, offsets
            )
            print(
                f"{mode_name}: col={col}, tail={tail}, threshold={threshold:.6f}, "
                f"offsets=[X={offsets[0]:.6f}, Y={offsets[1]:.6f}, Z={offsets[2]:.6f}]"
            )
            evaluate_multioutput(y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
            evaluate_multioutput(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            print_actual_position_distance_summary(val_diag_df, y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
            print_actual_position_distance_summary(test_diag_df, y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            print_error_bucket_summary(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            calibration_candidates.append((mode_name, cur_train_pred, cur_val_pred, cur_test_pred))
            tail_calibrated_test_preds[mode_name] = cur_test_pred

        y_tail_calibration_specs = [
            ("global_y_pose_y_low", "pose_y", 0.25),
            ("global_y_pose_y_low_mid", "pose_y", 0.50),
            ("global_y_pose_y_high", "pose_y", 0.75),
            ("global_y_feed_low", "feed_speed", 0.25),
            ("global_y_feed_high", "feed_speed", 0.75),
        ]
        for mode_name, col, quantile in y_tail_calibration_specs:
            threshold, offset = fit_low_tail_offset_calibrator(
                val_diag_df,
                y_val_true,
                global_y_val_pred,
                col=col,
                quantile=quantile,
                max_abs_offset=0.02,
            )
            if "high" in mode_name:
                threshold, offsets = fit_tail_axis_offset_calibrator(
                    val_diag_df,
                    y_val_true,
                    global_y_val_pred,
                    col=col,
                    quantile=quantile,
                    tail="high",
                    max_abs_offset=0.02,
                )
                offset = float(offsets[1])
                cur_train_pred = apply_tail_axis_offset_calibrator(
                    train_diag_df, global_y_train_pred, col, threshold, "high", np.asarray([0.0, offset, 0.0], dtype=np.float32)
                )
                cur_val_pred = apply_tail_axis_offset_calibrator(
                    val_diag_df, global_y_val_pred, col, threshold, "high", np.asarray([0.0, offset, 0.0], dtype=np.float32)
                )
                cur_test_pred = apply_tail_axis_offset_calibrator(
                    test_diag_df, global_y_test_pred, col, threshold, "high", np.asarray([0.0, offset, 0.0], dtype=np.float32)
                )
            else:
                cur_train_pred = apply_low_tail_offset_calibrator(
                    train_diag_df, global_y_train_pred, col, threshold, offset
                )
                cur_val_pred = apply_low_tail_offset_calibrator(
                    val_diag_df, global_y_val_pred, col, threshold, offset
                )
                cur_test_pred = apply_low_tail_offset_calibrator(
                    test_diag_df, global_y_test_pred, col, threshold, offset
                )

            print(f"{mode_name}: col={col}, threshold={threshold:.6f}, y_offset={offset:.6f}")
            evaluate_multioutput(y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
            evaluate_multioutput(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            print_actual_position_distance_summary(val_diag_df, y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
            print_actual_position_distance_summary(test_diag_df, y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            print_error_bucket_summary(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
            calibration_candidates.append((mode_name, cur_train_pred, cur_val_pred, cur_test_pred))
            tail_calibrated_test_preds[mode_name] = cur_test_pred

        residual_test_preds = {}
        if USE_RESIDUAL_CORRECTION:
            print("\n================ Residual correction candidates ================")
            residual_feature_cols = select_tabular_feature_cols(train_diag_df)
            residual_base_candidates = [
                ("residual_pre_cal", pre_cal_train_pred, pre_cal_val_pred, pre_cal_test_pred),
                ("residual_global_xyz", global_train_pred, global_val_pred, global_test_pred),
                ("residual_global_y", global_y_train_pred, global_y_val_pred, global_y_test_pred),
            ]
            if USE_SOURCE_OFFSET_CALIBRATION:
                residual_base_candidates.extend([
                    ("residual_source_xyz", source_axis_train_pred, source_axis_val_pred, source_axis_test_pred),
                    ("residual_source_y", source_y_train_pred, source_y_val_pred, source_y_test_pred),
                ])
            print(f"Residual feature count: {len(residual_feature_cols)}")
            for mode_name, base_train_pred, base_val_pred, base_test_pred in residual_base_candidates:
                cur_train_pred, cur_val_pred, cur_test_pred, residual_scale = fit_residual_corrected_predictions(
                    train_diag_df,
                    val_diag_df,
                    test_diag_df,
                    y_train_true,
                    y_val_true,
                    base_train_pred,
                    base_val_pred,
                    base_test_pred,
                    residual_feature_cols,
                    mode_name,
                )
                if residual_scale > 0.0:
                    evaluate_multioutput(y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
                    evaluate_multioutput(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
                    print_actual_position_distance_summary(val_diag_df, y_val_true, cur_val_pred, f"Hybrid Val ({mode_name})")
                    print_actual_position_distance_summary(test_diag_df, y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
                    print_error_bucket_summary(y_test_true, cur_test_pred, f"Hybrid Test ({mode_name})")
                    calibration_candidates.append((mode_name, cur_train_pred, cur_val_pred, cur_test_pred))
                    residual_test_preds[mode_name] = cur_test_pred

                if USE_LATE_STAGE_RESIDUAL:
                    late_mode_name = f"late_stage_{mode_name}"
                    (
                        late_train_pred,
                        late_val_pred,
                        late_test_pred,
                        late_scale,
                        late_train_count,
                    ) = fit_late_stage_residual_corrected_predictions(
                        train_diag_df,
                        val_diag_df,
                        test_diag_df,
                        y_train_true,
                        y_val_true,
                        base_train_pred,
                        base_val_pred,
                        base_test_pred,
                        residual_feature_cols,
                        late_mode_name,
                    )
                    if late_scale > 0.0 and late_train_count > 0:
                        evaluate_multioutput(y_val_true, late_val_pred, f"Hybrid Val ({late_mode_name})")
                        evaluate_multioutput(y_test_true, late_test_pred, f"Hybrid Test ({late_mode_name})")
                        print_actual_position_distance_summary(val_diag_df, y_val_true, late_val_pred, f"Hybrid Val ({late_mode_name})")
                        print_actual_position_distance_summary(test_diag_df, y_test_true, late_test_pred, f"Hybrid Test ({late_mode_name})")
                        print_late_stage_candidate_summary(val_diag_df, y_val_true, late_val_pred, f"Hybrid Val ({late_mode_name})")
                        print_late_stage_candidate_summary(test_diag_df, y_test_true, late_test_pred, f"Hybrid Test ({late_mode_name})")
                        print_error_bucket_summary(y_test_true, late_test_pred, f"Hybrid Test ({late_mode_name})")
                        calibration_candidates.append((late_mode_name, late_train_pred, late_val_pred, late_test_pred))
                        residual_test_preds[late_mode_name] = late_test_pred

                if USE_NEAR_BOUNDARY_RESIDUAL:
                    near_mode_name = f"near_boundary_{mode_name}"
                    (
                        nb_train_pred,
                        nb_val_pred,
                        nb_test_pred,
                        nb_scale,
                        nb_train_count,
                    ) = fit_near_boundary_residual_corrected_predictions(
                        train_diag_df,
                        val_diag_df,
                        test_diag_df,
                        y_train_true,
                        y_val_true,
                        base_train_pred,
                        base_val_pred,
                        base_test_pred,
                        residual_feature_cols,
                        near_mode_name,
                    )
                    if nb_scale > 0.0 and nb_train_count > 0:
                        evaluate_multioutput(y_val_true, nb_val_pred, f"Hybrid Val ({near_mode_name})")
                        evaluate_multioutput(y_test_true, nb_test_pred, f"Hybrid Test ({near_mode_name})")
                        print_actual_position_distance_summary(val_diag_df, y_val_true, nb_val_pred, f"Hybrid Val ({near_mode_name})")
                        print_actual_position_distance_summary(test_diag_df, y_test_true, nb_test_pred, f"Hybrid Test ({near_mode_name})")
                        print_error_bucket_summary(y_test_true, nb_test_pred, f"Hybrid Test ({near_mode_name})")
                        calibration_candidates.append((near_mode_name, nb_train_pred, nb_val_pred, nb_test_pred))
                        residual_test_preds[near_mode_name] = nb_test_pred

                if not USE_PER_SOURCE_RESIDUAL:
                    continue

                per_source_mode_name = f"per_source_{mode_name}"
                (
                    ps_train_pred,
                    ps_val_pred,
                    ps_test_pred,
                    ps_scale,
                    ps_groups,
                ) = fit_per_source_residual_corrected_predictions(
                    train_diag_df,
                    val_diag_df,
                    test_diag_df,
                    y_train_true,
                    y_val_true,
                    base_train_pred,
                    base_val_pred,
                    base_test_pred,
                    residual_feature_cols,
                    per_source_mode_name,
                )
                if ps_scale <= 0.0 or ps_groups == 0:
                    continue
                evaluate_multioutput(y_val_true, ps_val_pred, f"Hybrid Val ({per_source_mode_name})")
                evaluate_multioutput(y_test_true, ps_test_pred, f"Hybrid Test ({per_source_mode_name})")
                print_actual_position_distance_summary(val_diag_df, y_val_true, ps_val_pred, f"Hybrid Val ({per_source_mode_name})")
                print_actual_position_distance_summary(test_diag_df, y_test_true, ps_test_pred, f"Hybrid Test ({per_source_mode_name})")
                print_error_bucket_summary(y_test_true, ps_test_pred, f"Hybrid Test ({per_source_mode_name})")
                calibration_candidates.append((per_source_mode_name, ps_train_pred, ps_val_pred, ps_test_pred))
                residual_test_preds[per_source_mode_name] = ps_test_pred

        selected_mode, train_pred, val_pred, test_pred = max(
            calibration_candidates,
            key=lambda item: actual_position_distance_ratio(val_diag_df, y_val_true, item[2]),
        )
        print(
            f"Selected XYZ calibration mode: {selected_mode}, "
            f"val_overall_ratio={within_threshold_ratio(y_val_true, val_pred):.6f}, "
            f"test_overall_ratio={within_threshold_ratio(y_test_true, test_pred):.6f}, "
            f"val_position_distance_ratio={actual_position_distance_ratio(val_diag_df, y_val_true, val_pred):.6f}, "
            f"test_position_distance_ratio={actual_position_distance_ratio(test_diag_df, y_test_true, test_pred):.6f}, "
            f"val_y_ratio={within_threshold_ratio(y_val_true, val_pred, axis_idx=1):.6f}, "
            f"test_y_ratio={within_threshold_ratio(y_test_true, test_pred, axis_idx=1):.6f}"
        )

    evaluate_multioutput(y_train_true, train_pred, "Hybrid Train")
    evaluate_multioutput(y_val_true, val_pred, "Hybrid Val")
    evaluate_multioutput(y_test_true, test_pred, "Hybrid Test")
    print_actual_position_distance_summary(train_diag_df, y_train_true, train_pred, "Hybrid Train")
    print_actual_position_distance_summary(val_diag_df, y_val_true, val_pred, "Hybrid Val")
    print_actual_position_distance_summary(test_diag_df, y_test_true, test_pred, "Hybrid Test")
    if len(test_diag_df) == len(y_test_true) and all(col in test_diag_df.columns for col in POSE_COLS):
        test_distances = actual_position_distance(test_diag_df, y_test_true, test_pred)
        test_pass_ratio = float(np.mean(test_distances < POSITION_DISTANCE_THRESHOLD))
        print(
            f"测试集总体结果: 样本数={len(test_distances)}, "
            f"平均欧氏距离={np.mean(test_distances):.6f}, "
            f"最大欧氏距离={np.max(test_distances):.6f}, "
            f"距离<{POSITION_DISTANCE_THRESHOLD:.3f}占比={test_pass_ratio:.6f}"
        )
    print_test_point_prediction_details(test_diag_df, y_test_true, test_pred, "Hybrid Test")

    if STOP_AFTER_SOURCE_SUMMARY:
        return

    print_error_bucket_summary(y_test_true, test_pred, "Hybrid Test")

    if len(test_diag_df) == len(test_meta):
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "temp_median_mean", "Hybrid Test")
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "temp_ambient_c", "Hybrid Test")
        for ch in TEMP_CHANNELS:
            print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, f"temp_ch{ch}_median_c", "Hybrid Test")
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "temp_spindle_ambient_delta", "Hybrid Test")
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "temp_spindle_ambient_delta_rise_from_start", "Hybrid Test")
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "progress_spindle_ambient_delta_rise_interact", "Hybrid Test")
        for ch in SPINDLE_TEMP_CHANNELS:
            print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, f"temp_ch{ch}_ambient_delta", "Hybrid Test")
            print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, f"temp_ch{ch}_ambient_delta_rise_from_start", "Hybrid Test")
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "pose_r_xyz", "Hybrid Test")
        print_grouped_ratio_summary(test_diag_df, y_test_true, test_pred, "feed_speed", "Hybrid Test")
        print_y_axis_diagnostic(test_diag_df, y_test_true, test_pred, "Hybrid Test")

    print("正在保存预测图像...")
    for split_name, y_true, y_pred in [
        ("train", y_train_true, train_pred),
        ("test", y_test_true, test_pred),
    ]:
        for i, axis in enumerate(["x", "y", "z"]):
            save_single_compare_plot(
                y_true,
                y_pred,
                i,
                f"{split_name.capitalize()} Hybrid",
                "Hybrid Pred",
                os.path.join(OUTPUT_DIR, f"{split_name}_hybrid_{axis}_v10.png"),
                SHOW_PLOTS,
            )

    save_source_compare_plots(
        "train",
        train_meta,
        y_train_true,
        train_pred,
        "Hybrid",
        OUTPUT_DIR,
        "hybrid_v10",
        SHOW_PLOTS,
    )
    save_source_compare_plots(
        "test",
        test_meta,
        y_test_true,
        test_pred,
        "Hybrid",
        OUTPUT_DIR,
        "hybrid_v10",
        SHOW_PLOTS,
    )

    print("正在整理预测结果表...")
    pred_meta_cols = FRAME_KEYS + ["point_index"]
    if "source_file_name" in test_meta.columns:
        pred_meta_cols.append("source_file_name")
    pred_df = test_meta[pred_meta_cols].copy()
    test_class_true = build_error_class_labels(test_y, error_class_edges)
    test_class_hat = np.argmax(test_class_pred, axis=-1)
    for j, axis in enumerate(["x", "y", "z"]):
        pred_df[f"true_{axis}"] = y_test_true[:, j]
        pred_df[f"hybrid_pred_{axis}"] = test_pred[:, j]
        pred_df[f"hybrid_base_pred_{axis}"] = pre_cal_test_pred[:, j]
        pred_df[f"true_{axis}_class"] = test_class_true[:, j]
        pred_df[f"pred_{axis}_class"] = test_class_hat[:, j]
    if len(test_diag_df) == len(test_meta) and all(col in test_diag_df.columns for col in POSE_COLS):
        pose_xyz = test_diag_df[POSE_COLS].to_numpy(dtype=np.float32)
        true_position = pose_xyz + y_test_true
        pred_position = pose_xyz + test_pred
        base_pred_position = pose_xyz + pre_cal_test_pred
        pred_df["actual_position_distance"] = np.linalg.norm(pred_position - true_position, axis=1)
        pred_df["base_actual_position_distance"] = np.linalg.norm(base_pred_position - true_position, axis=1)
        for j, axis in enumerate(["x", "y", "z"]):
            pred_df[f"pose_{axis}"] = pose_xyz[:, j]
            pred_df[f"true_actual_{axis}"] = true_position[:, j]
            pred_df[f"hybrid_pred_actual_{axis}"] = pred_position[:, j]
            pred_df[f"hybrid_base_pred_actual_{axis}"] = base_pred_position[:, j]
            pred_df[f"abs_err_{axis}"] = np.abs(pred_df[f"hybrid_pred_{axis}"] - pred_df[f"true_{axis}"])
            pred_df[f"fine_pass_{axis}"] = pred_df[f"abs_err_{axis}"] < AXIS_FINE_THRESHOLD
    if 'global_test_pred' in locals():
        for j, axis in enumerate(["x", "y", "z"]):
            pred_df[f"hybrid_global_xyz_cal_pred_{axis}"] = global_test_pred[:, j]
    if 'tail_calibrated_test_preds' in locals():
        for mode_name, mode_pred in tail_calibrated_test_preds.items():
            for j, axis in enumerate(["x", "y", "z"]):
                pred_df[f"hybrid_{mode_name}_pred_{axis}"] = mode_pred[:, j]
    if 'source_region_test_preds' in locals():
        for mode_name, mode_pred in source_region_test_preds.items():
            for j, axis in enumerate(["x", "y", "z"]):
                pred_df[f"hybrid_{mode_name}_pred_{axis}"] = mode_pred[:, j]
    if 'residual_test_preds' in locals():
        for mode_name, mode_pred in residual_test_preds.items():
            for j, axis in enumerate(["x", "y", "z"]):
                pred_df[f"hybrid_{mode_name}_pred_{axis}"] = mode_pred[:, j]

    distance_bar_dir = os.path.join(OUTPUT_DIR, "position_distance_bars")
    if "actual_position_distance" in pred_df.columns:
        save_position_distance_bar_plots(
            pred_df,
            OUTPUT_DIR,
            "test_hybrid_v10",
            SHOW_PLOTS,
        )

    print("正在保存预测结果文件...")
    moe_gate_path = os.path.join(OUTPUT_DIR, "tool_error_test_moe_gate_diagnostic_v10.csv")
    if test_moe_gate_pred is not None and len(test_moe_gate_pred) == len(pred_df):
        for expert_idx in range(test_moe_gate_pred.shape[1]):
            pred_df[f"moe_gate_{expert_idx}"] = test_moe_gate_pred[:, expert_idx]
        gate_diag_cols = pred_meta_cols.copy()
        gate_diag_df = pred_df[gate_diag_cols].copy()
        if "actual_position_distance" in pred_df.columns:
            gate_diag_df["actual_position_distance"] = pred_df["actual_position_distance"].to_numpy()
            gate_diag_df["position_fail"] = pred_df["actual_position_distance"].to_numpy() >= POSITION_DISTANCE_THRESHOLD
        for expert_idx in range(test_moe_gate_pred.shape[1]):
            gate_diag_df[f"moe_gate_{expert_idx}"] = test_moe_gate_pred[:, expert_idx]
        for col in condition_feature_cols + [
            "feed_speed",
            "spindle_speed",
            "temp_median_mean",
            "temp_median_range",
            "pose_r_xy",
            "pose_r_xyz",
        ]:
            if col in test_diag_df.columns and col not in gate_diag_df.columns:
                gate_diag_df[col] = test_diag_df[col].to_numpy()
        gate_diag_df.to_csv(moe_gate_path, index=False, encoding="utf-8-sig")
        gate_mean = np.mean(test_moe_gate_pred, axis=0)
        gate_std = np.std(test_moe_gate_pred, axis=0)
        gate_text = ", ".join(
            f"expert{idx}: mean={gate_mean[idx]:.3f}, std={gate_std[idx]:.3f}"
            for idx in range(test_moe_gate_pred.shape[1])
        )
        print(f"\nMoE gate diagnostic -> {gate_text}")

    failure_path = os.path.join(OUTPUT_DIR, "tool_error_test_position_failures_v10_hybrid_spatiotemporal.csv")
    if "actual_position_distance" in pred_df.columns and len(test_diag_df) == len(pred_df):
        failure_df = pred_df[pred_df["actual_position_distance"] >= POSITION_DISTANCE_THRESHOLD].copy()

        diagnostic_cols = [
            "source_file_path",
            "original_mapped_file_index",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_r_xy",
            "pose_r_xyz",
            "feed_speed",
            "spindle_speed",
            "temp_median_mean",
            "temp_median_range",
            "temp_std_mean",
            "temp_ch3_median_c",
            "temp_ch4_median_c",
            "temp_ch5_median_c",
            "temp_ch6_median_c",
            "temp_spindle_mean",
            "temp_spindle_range",
            "temp_spindle_std_mean",
            "temp_ambient_c",
            "temp_spindle_ambient_delta",
            "temp_ch3_ambient_delta",
            "temp_ch4_ambient_delta",
            "temp_ch6_ambient_delta",
            "temp_spindle_ambient_delta_rise_from_start",
            "temp_spindle_ambient_delta_expanding_mean",
            "temp_spindle_ambient_delta_expanding_delta",
            "temp_ch3_ambient_delta_rise_from_start",
            "temp_ch4_ambient_delta_rise_from_start",
            "temp_ch6_ambient_delta_rise_from_start",
            "progress_spindle_ambient_delta_rise_interact",
            "progress_spindle_ambient_delta_expanding_mean_interact",
            "progress_ch3_ambient_delta_rise_interact",
            "progress_ch4_ambient_delta_rise_interact",
            "progress_ch6_ambient_delta_rise_interact",
        ]
        for col in diagnostic_cols:
            if col in test_diag_df.columns and col not in failure_df.columns:
                failure_df[col] = test_diag_df.loc[failure_df.index, col].to_numpy()

        failure_df = failure_df.sort_values("actual_position_distance", ascending=False)
        failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")

        source_failure_path = os.path.join(OUTPUT_DIR, "tool_error_test_source_failure_summary_v10.csv")
        if "source_file_name" in pred_df.columns:
            total_by_source = pred_df.groupby("source_file_name").size().rename("total")
            fail_by_source = failure_df.groupby("source_file_name").size().rename("fail")
            source_summary = pd.concat([total_by_source, fail_by_source], axis=1).fillna(0)
            source_summary["fail_rate"] = source_summary["fail"] / source_summary["total"]
            source_summary["pass_rate"] = 1.0 - source_summary["fail_rate"]
            source_summary = source_summary.sort_values(["fail_rate", "fail"], ascending=[False, False])
            source_summary.to_csv(source_failure_path, encoding="utf-8-sig")

        source_diagnostic_path = os.path.join(OUTPUT_DIR, "tool_error_test_source_diagnostic_summary_v10.csv")
        save_source_diagnostic_summary(pred_df, test_diag_df, source_diagnostic_path)
        near_boundary_diagnostic_path = os.path.join(OUTPUT_DIR, "tool_error_test_near_boundary_diagnostic_v10.csv")
        feature_bucket_diagnostic_path = os.path.join(OUTPUT_DIR, "tool_error_test_feature_bucket_diagnostic_v10.csv")
        line_window_diagnostic_path = os.path.join(OUTPUT_DIR, "tool_error_test_line_window_diagnostic_v10.csv")
        phase_progress_diagnostic_path = os.path.join(OUTPUT_DIR, "tool_error_test_phase_progress_diagnostic_v10.csv")
        save_near_boundary_diagnostic(pred_df, test_diag_df, near_boundary_diagnostic_path)
        save_failure_feature_bucket_diagnostic(pred_df, test_diag_df, feature_bucket_diagnostic_path)
        save_failure_line_window_diagnostic(pred_df, line_window_diagnostic_path)
        save_phase_progress_diagnostic(pred_df, test_diag_df, phase_progress_diagnostic_path)

    pred_path = os.path.join(OUTPUT_DIR, "tool_error_test_predictions_v10_hybrid_spatiotemporal.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"预测明细已保存: {pred_path}")
    if os.path.isfile(failure_path):
        print(f"Saved position-distance failures to: {failure_path}")
    if 'source_failure_path' in locals() and os.path.isfile(source_failure_path):
        print(f"Saved source failure summary to: {source_failure_path}")
    if 'source_diagnostic_path' in locals() and os.path.isfile(source_diagnostic_path):
        print(f"Saved source diagnostic summary to: {source_diagnostic_path}")
    if 'moe_gate_path' in locals() and os.path.isfile(moe_gate_path):
        print(f"Saved MoE gate diagnostic to: {moe_gate_path}")
    if 'near_boundary_diagnostic_path' in locals() and os.path.isfile(near_boundary_diagnostic_path):
        print(f"Saved near-boundary diagnostic to: {near_boundary_diagnostic_path}")
    if 'feature_bucket_diagnostic_path' in locals() and os.path.isfile(feature_bucket_diagnostic_path):
        print(f"Saved feature bucket diagnostic to: {feature_bucket_diagnostic_path}")
    if 'line_window_diagnostic_path' in locals() and os.path.isfile(line_window_diagnostic_path):
        print(f"Saved line/window diagnostic to: {line_window_diagnostic_path}")
    if 'phase_progress_diagnostic_path' in locals() and os.path.isfile(phase_progress_diagnostic_path):
        print(f"Saved phase/progress diagnostic to: {phase_progress_diagnostic_path}")
    if 'distance_bar_dir' in locals() and os.path.isdir(distance_bar_dir):
        print(f"Saved position-distance bar plots to: {distance_bar_dir}")
    if "source_file_name" in pred_df.columns:
        open_prediction_plots(
            OUTPUT_DIR,
            pred_df["source_file_name"].dropna().astype(str).drop_duplicates().tolist(),
        )
    print(f"预测图像已保存: {OUTPUT_DIR}")
    print("测试阶段完成。")


if __name__ == "__main__":
    main()



# import json

# input_path =  r"E:\采集卡编程\USB-5000-Python\input\output2000.json"
# output_path = r"E:\采集卡编程\USB-5000-Python\input\json\input6.json"

# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# data["point_to_frame_ai_block_avg"] = sorted(
#     data["point_to_frame_ai_block_avg"],
#     key=lambda x: (x["mapped_file_index"], x["mapped_nc_line"])
# )

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print("排序完成，已保存到", output_path)


# import json

# input_path = r"E:\采集卡编程\USB-5000-Python\input\json\input3.json"
# output_path = r"E:\采集卡编程\USB-5000-Python\input\json\input3-120.json"

# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 遍历每一条数据
# for item in data["point_to_frame_ai_block_avg"]:
#     if "mapped_machine_state" in item:
#         state = item["mapped_machine_state"]
#         if "spindle_speed" in state and state["spindle_speed"] is not None:
#             state["spindle_speed"] *= 1.2

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print("处理完成")
