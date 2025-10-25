#!/usr/bin/env python3
"""Train LSTM models for Lotto prediction using 12, 24, and 36 draw windows."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf

NUMBER_RANGE = 45
NUMBERS_PER_DRAW = 6
DEFAULT_WINDOWS = (12, 24, 36)


def set_seed(seed: int = 42) -> None:
    """Improve reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_draws(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load draw data from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"draw_no", "n1", "n2", "n3", "n4", "n5", "n6"}
    if not required_cols.issubset(set(df.columns)):
        missing = ", ".join(sorted(required_cols.difference(df.columns)))
        raise ValueError(f"CSV file is missing required columns: {missing}")

    draw_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    draws = df.sort_values("draw_no")[draw_cols].apply(lambda row: sorted(int(x) for x in row), axis=1)
    draws_df = pd.DataFrame(draws.tolist(), columns=draw_cols)
    draws_df.index = df.sort_values("draw_no")["draw_no"].values
    return draws_df


def to_multihot(numbers: Iterable[int]) -> np.ndarray:
    """Convert a sequence of numbers to a multi-hot vector."""
    vector = np.zeros(NUMBER_RANGE, dtype=np.float32)
    for num in numbers:
        if 1 <= num <= NUMBER_RANGE:
            vector[num - 1] = 1.0
    return vector


@dataclass
class DatasetBundle:
    window: int
    X: np.ndarray
    y: np.ndarray
    last_sequence: np.ndarray


def build_dataset(draws: pd.DataFrame, window: int) -> DatasetBundle:
    """Build training data for the provided rolling window."""
    encoded_draws = draws.apply(to_multihot, axis=1, result_type="expand").values
    samples = len(encoded_draws) - window
    if samples <= 0:
        raise ValueError(f"Not enough draws ({len(encoded_draws)}) for window size {window}.")

    X = np.stack(
        [encoded_draws[idx : idx + window] for idx in range(samples)],
        axis=0,
    )
    y = np.stack(
        [encoded_draws[idx + window] for idx in range(samples)],
        axis=0,
    )
    last_sequence = encoded_draws[-window:]
    return DatasetBundle(window=window, X=X, y=y, last_sequence=last_sequence)


def build_model(window: int) -> tf.keras.Model:
    """Create a simple LSTM-based predictor."""
    inputs = tf.keras.layers.Input(shape=(window, NUMBER_RANGE))
    x = tf.keras.layers.LSTM(128, return_sequences=False)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUMBER_RANGE, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"lstm_window_{window}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)],
    )
    return model


def train_model(model: tf.keras.Model, bundle: DatasetBundle, epochs: int, batch_size: int) -> tf.keras.callbacks.History:
    """Train the provided model."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        bundle.X,
        bundle.y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    return history


def predict_numbers(model: tf.keras.Model, bundle: DatasetBundle) -> list[int]:
    """Predict the top six numbers using the trained model."""
    sequence = np.expand_dims(bundle.last_sequence, axis=0)
    probabilities = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-NUMBERS_PER_DRAW:]
    return sorted(int(idx + 1) for idx in top_indices)


def run_pipeline(csv_path: pathlib.Path, windows: tuple[int, ...], epochs: int, batch_size: int) -> None:
    set_seed()
    draws = load_draws(csv_path)

    results = []
    for window in windows:
        bundle = build_dataset(draws, window)
        model = build_model(window)
        _ = train_model(model, bundle, epochs=epochs, batch_size=batch_size)
        predicted = predict_numbers(model, bundle)
        results.append((window, predicted))

    print("=== LSTM Prediction Comparison ===")
    print(f"Source file: {csv_path}")
    for window, prediction in results:
        numbers_str = ", ".join(f"{num:02d}" for num in prediction)
        print(f"- 최근 {window:2d}회 기반: {numbers_str}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM models for lotto predictions.")
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("lotto.csv"),
        help="CSV file containing columns draw_no,n1..n6 (default: lotto.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum training epochs for each window (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=DEFAULT_WINDOWS,
        help="Custom window sizes (default: 12 24 36)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    windows = tuple(args.windows) if args.windows else DEFAULT_WINDOWS
    run_pipeline(args.csv, windows, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main(sys.argv[1:])

