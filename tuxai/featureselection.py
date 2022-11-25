"""Feature selection tools."""
import logging
from collections import defaultdict
import pickle

from tqdm import tqdm
from pathlib import Path
import pandas as pd

from tuxai.dataset import Dataset, Columns

LOG = logging.getLogger(__name__)

OPTION_COMPARISON_MATRIX_CACHE = "ocm.pkl"  # concatenate version too


def options_comparison_matrix(
    dataset: Dataset, cache_dir: None | str = None, as_dataframe: bool = False
) -> dict[str, dict[str, float]] | pd.DataFrame:
    """Compute similarity score between options (~4h for 4.13)."""
    col_sims: dict[str, dict[str, float]] = defaultdict(dict)
    if cache_dir:
        cache_path = (
            Path(cache_dir) / f"{dataset.version}_{OPTION_COMPARISON_MATRIX_CACHE}"
        )
        if cache_path.exists():
            LOG.debug(f"loading matrix file {cache_path}")
            with open(cache_path, "rb") as cache_path_f:
                col_sims = pickle.load(cache_path_f)

    if not col_sims:
        LOG.info(
            f"computing comparison matrix between options for version {dataset.version} - it might take a few hours"
        )
        df = dataset.get_dataframe(Columns.options)
        for col_a in (pbar := tqdm(df.columns)):
            for col_b in df.columns:
                pbar.set_description(col_a)
                if col_a != col_b:
                    if col_a in col_sims[col_b]:
                        col_sims[col_a][col_b] = col_sims[col_b][col_a]
                    else:
                        col_sims[col_a][col_b] = (df[col_a] == df[col_b]).sum() / len(
                            df
                        )
        # serialize if needed
        if not cache_path.exists():
            LOG.debug(f"serializing comparison matrix: {cache_path}.")
            with open(cache_path, "wb") as cache_path_f:
                pickle.dump(col_sims, cache_path_f, protocol=pickle.HIGHEST_PROTOCOL)

    return pd.DataFrame.from_dict(col_sims) if as_dataframe else col_sims


def correlated_features(
    cols_matrix: dict[str, dict[str, float]], threshold=0.0
) -> list[set]:
    """Group same features."""

    def add_to_groups(groups: list, cols: list[str]) -> None:
        """Add correlated features to groups."""
        for group in groups:
            if group & {*cols}:
                group.update(cols)
                return
        groups.append({*cols})

    groups: list[set] = list()

    for col_a, cols_b in tqdm(cols_matrix.items()):
        for col_b, val in cols_b.items():
            if val >= (1 - threshold) or val <= threshold:
                add_to_groups(groups, [col_a, col_b])
    return groups
