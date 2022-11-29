"""Feature selection tools."""
import logging
from collections import defaultdict

from tqdm.auto import tqdm
import pandas as pd

from tuxai.dataset import Dataset, Columns
from tuxai.misc import config, cache

LOG = logging.getLogger(__name__)


class FeatureSelection:
    """Collection of feature collection methods."""

    def __init__(self, dataset: Dataset | str | int | float) -> None:
        """Enable dataset instance to save memory."""
        self._config = config()
        self._dataset = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        self._cache = cache()

    def options_comparison_matrix(
        self, use_cache: bool = True, as_dataframe: bool = False
    ) -> dict[str, dict[str, float]] | pd.DataFrame:
        """Compute similarity score between options (~4h for 4.13)."""
        col_sims: dict[str, dict[str, float]] = defaultdict(dict)
        if use_cache:
            key = f"ocm|{self._dataset.version}"
            if key in self._cache:
                LOG.debug(
                    f"loading matrix file for dataset version {self._dataset.version}"
                )
                col_sims = self._cache[key]

        if not col_sims:
            LOG.info(
                f"computing comparison matrix between options for version {self._dataset.version} "
                "- it might take a few hours"
            )
            df = self._dataset.get_dataframe(Columns.options)
            for col_a in (pbar := tqdm(df.columns)):
                for col_b in df.columns:
                    pbar.set_description(col_a)
                    if col_a != col_b:
                        if col_a in col_sims[col_b]:
                            col_sims[col_a][col_b] = col_sims[col_b][col_a]
                        else:
                            col_sims[col_a][col_b] = (
                                df[col_a] == df[col_b]
                            ).sum() / len(df)
            # serialize, even if use_cache is False
            LOG.debug(
                f"serializing comparison matrix for dataset {self._dataset.version}."
            )
            self._cache[key] = col_sims

        return pd.DataFrame.from_dict(col_sims) if as_dataframe else col_sims

    @staticmethod
    def correlated_features(
        cols_matrix: dict[str, dict[str, float]], threshold: float = 0.0
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

    def removable_columns(self, threshold: float = 0.0) -> list[str]:
        """Get a list of features that can be removed."""
        cols_matrix = self.options_comparison_matrix()
        to_drop = set()
        for group in self.correlated_features(cols_matrix, threshold=threshold):
            # remove one
            group.pop()
            # add others
            to_drop.update(group)
        return list(to_drop)


if __name__ == "__main__":
    from pathlib import Path
    import pickle

    # convert old cache to new cache

    # old
    OPTION_COMPARISON_MATRIX_CACHE = "ocm.pkl"
    cache_dir = config()["diskcache"]["path"]
    cache_path = (
        Path(cache_dir).parent / "cache_old" / f"{413}_{OPTION_COMPARISON_MATRIX_CACHE}"
    )
    with open(cache_path, "rb") as cache_path_f:
        col_sims = pickle.load(cache_path_f)

    # new
    if col_sims:
        new_cache = cache()
        new_cache["ocm|413"] = col_sims

        ocm_413 = new_cache["ocm|413"]

        print(ocm_413)
