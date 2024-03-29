"""Feature selection tools."""
import logging
import hashlib

import pandas as pd
import numpy as np

from tuxai2.misc import get_config, cache, filter_options, df2bio, bio2df

LOG = logging.getLogger(__name__)

CORR_PREFIX = "CORR|"
NB_YES_COL = "nb_yes"


class Collinearity:
    """Manage groups of colinear options."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Enable dataset instance, in order to save memory."""
        self._config = get_config()
        self._dataframe = dataframe
        self._cache = cache()

    def _correlation_matrix(self) -> pd.DataFrame:
        """Get correlation between options. WARNING: time consuming operation (~3h for 4.13)."""
        # compute a hash based on column names and dataframe size (maybe use data content later)
        key_part = "|".join(sorted(self._dataframe.columns))
        key_part += str(self._dataframe.shape)
        key = f"corr_matrix|{hashlib.md5(key_part.encode()).hexdigest()}"
        if key in self._cache:
            LOG.debug("loading correlation matrix from cache")
            return bio2df(self._cache[key])

        LOG.info(
            "computing correlation matrix between provided features "
            "- it might take a few hours"
        )
        # computing actual matrix (kind of slow)
        corr_mat = self._dataframe.corr()
        LOG.debug("serializing correlation matrix.")
        self._cache[key] = df2bio(corr_mat)
        return corr_mat

    def correlated_features(self, threshold: float = 0.0) -> list[list[str]]:
        """Group same features."""

        def add_to_groups(groups: list, cols: list[str]) -> None:
            """Add correlated features to groups."""
            for group in groups:
                if group & {*cols}:
                    group.update(cols)
                    return
            groups.append({*cols})

        groups: list[set] = list()

        corr_mat = self._correlation_matrix().abs()

        # remove duplicate and identity
        # 1 - lower triangle only
        corr_mat.loc[:, :] = np.tril(corr_mat.values, k=-1)
        # 2 - NaN
        corr_mat = corr_mat[corr_mat > 0]
        # 3 - Series containing 3 columns: [level_0, level_1 and 0]" (-> [option_1, option_2, correlation])
        kept = corr_mat.unstack().reset_index()
        kept = kept[kept[0] >= (1 - threshold)]

        # make groups
        for _, (option_1, option_2, _) in kept.iterrows():
            add_to_groups(groups, cols=[option_1, option_2])

        return [list(group) for group in groups]

    def group_correlated_features(
        self, threshold: float = 0.0
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Return a dataframe with correlated features merged.

        New column name is CORR|num.
        """
        LOG.debug("group collinear options")
        groups = dict()
        to_drop = list()
        to_rename = dict()

        # keep one of each group, drop remaining
        for idx, group in enumerate(self.correlated_features(threshold=threshold)):
            key = f"{CORR_PREFIX}{idx:04}"
            groups[key] = group
            to_rename[group[0]] = key
            to_drop.extend(group[1:])

        self._dataframe = self._dataframe.drop(columns=to_drop)
        self._dataframe = self._dataframe.rename(columns=to_rename)
        LOG.debug(
            f"collinear options: {len(to_rename)} groups found, {len(to_drop)} columns removed"
        )
        return self._dataframe, groups


class FeatureEngineering:
    """Add features to dataset."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Keep dataframe reference."""
        self._dataframe = dataframe

    def add_nb_yes_feature(
        self, val_range: tuple[float, float] | None = None
    ) -> pd.DataFrame:
        """Add nb_yes column, containing activated options count."""
        # drop if exists
        if NB_YES_COL in self._dataframe.columns:
            LOG.debug(f"drop existing feature {NB_YES_COL}")
            self._dataframe.drop(columns=[NB_YES_COL])

        LOG.debug(f"add {NB_YES_COL} feature.")
        # boolean count as 1
        # count options only
        self._dataframe[NB_YES_COL] = self._dataframe[
            filter_options(self._dataframe)
        ].sum(axis=1)

        if val_range:
            # linear interpolation
            LOG.debug(f"apply range {val_range}")
            min_val = self._dataframe[NB_YES_COL].min()
            max_val = self._dataframe[NB_YES_COL].max()
            self._dataframe[NB_YES_COL] = self._dataframe[NB_YES_COL].apply(
                lambda val: np.interp(val, (min_val, max_val), val_range)
            )
        return self._dataframe

    def add_random_features(self, name: str, add_clones: int = 0) -> pd.DataFrame:
        """Add random_00n column. add_clone parameter create multiple columns with same values."""
        if add_clones == 0:
            columns = [name]
        else:
            columns = [f"{name}_{num}" for num in range(add_clones + 1)]

        LOG.info(f"add {', '.join(columns)} feature(s).")

        data = np.random.choice(a=[True, False], size=len(self._dataframe))
        for column in columns:
            self._dataframe[column] = data
        return self._dataframe


if __name__ == "__main__":
    from tuxai2.misc import config_logger
    from tuxai2.dataset import Dataset, Columns

    config_logger()

    df = FeatureEngineering(
        Dataset(413).get_dataframe(Columns.options)
    ).add_random_features("random", add_clones=3)
    print(df)
