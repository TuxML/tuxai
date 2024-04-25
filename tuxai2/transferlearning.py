"""Use models from older kernel to train a newer model.
How:
- use (or train) an older version model and get its dataset
- compare older and newer versions dataset columns
- train a first new model to predict how new columns affect result
    - predict common columns using older model (set everything else to 0)
    - train difference (diff or ratio) with new columns
- now, build a new dataset with all new columns and a predicted result
- train a second new model with this temporary dataset
"""

import logging
from enum import Enum, unique
import random

from pathlib import Path

import pandas as pd
from tuxai2.dataset import Columns, Dataset
from tuxai2.models import XGBoost
from tuxai2.misc import version_str

LOG = logging.getLogger(__name__)


@unique
class Method(Enum):
    """How to compare datasets."""

    diff = 0
    ratio = 1


class TransferLearning:
    """Train or reuse old model and dataset for newer model."""

    # no need for collineraty while training (important for feature importance only)
    _dataframe_params = {
        "group_collinear_options": False,
        # "add_features": True,
    }

    def __init__(
        self,
        older_model_json_path: Path | str,
        newer_model_json_path: Path | str,
        older_version: str | int | float,
        newer_version: str | int | float,
        target: str = "vmlinux",
        use_cache: bool = True,
    ) -> None:
        """If model is not provided, it will be trained and saved."""
        self._use_cache = use_cache
        self._older_version = version_str(older_version)
        self._newer_version = version_str(newer_version)
        self._target = target
        self._older_model_json_path = Path(older_model_json_path).expanduser().resolve()
        self._newer_model_json_path = Path(newer_model_json_path).expanduser().resolve()

        self._older_dataset = Dataset(older_version)
        self._newer_dataset = Dataset(newer_version)

    def _train_model(
        self, df: pd.DataFrame, version: str, save_path: Path | None = None
    ) -> None:
        """Train tmp or final new model."""
        xgb = XGBoost(
            dataframe=df,
            target=self._target,
            version=version,
            test_size=0,
            use_cache=self._use_cache,
        )
        xgb.fit()
        if save_path:
            xgb.save(save_path)
        return xgb

    def train(
        self,
        method: Method = Method.ratio,
        limit: int | None = None,
        fake_dataset_size: int = 10000,
    ) -> dict:
        """Train new models. If limit is set, do not use full dataset."""
        # get older model
        if self._older_model_json_path.exists():
            older_model = XGBoost(
                load_model=self._older_model_json_path, target=self._target
            )
        else:
            LOG.warning(
                f"older model not found ({self._older_model_json_path}): build and train."
            )
            older_model = self._train_model(
                df=self._older_dataset.get_dataframe(
                    add_features=False, **self._dataframe_params
                ),
                version=self._older_version,
                save_path=self._older_model_json_path,
            )

        # get older dataframe
        older_df = self._older_dataset.get_dataframe(
            col_filter=Columns.options, **self._dataframe_params
        )

        # get newer dataframe
        newer_df = self._newer_dataset.get_dataframe(
            col_filter=Columns.options, **self._dataframe_params
        )
        newer_df_target = self._newer_dataset.get_dataframe(
            col_filter=Columns.targets, **self._dataframe_params
        )[self._target]
        if limit:
            newer_df = newer_df.iloc[:limit]
            newer_df_target = newer_df_target.iloc[:limit]
        else:
            limit = len(newer_df)

        # compare options
        # (be carefull with generated columns which are not options)
        older_options = set(
            option
            for option in self._older_dataset.raw_option_list()
            if option in older_df.columns
        )
        newer_options = set(
            option
            for option in self._newer_dataset.raw_option_list()
            if option in newer_df.columns
        )
        common_options = list(older_options & newer_options)
        newer_options_only = list(newer_options - older_options)
        older_options_only = list(older_options - newer_options)

        # build temporary dataset for old model using new dataset
        common_df = newer_df[common_options].copy()
        common_df[older_options_only] = False

        # get predictions for this temporary dataset using older model
        common_df_target = older_model.pred(common_df)

        # compute diff (or ratio)
        if method == Method.diff:
            diff_df_target = newer_df_target - common_df_target
        elif method == Method.ratio:
            diff_df_target = newer_df_target / common_df_target

        # build fake dataset
        common_idx = [random.randint(0, limit) for _ in range(fake_dataset_size)]
        diff_idx = [random.randint(0, limit) for _ in range(fake_dataset_size)]

        fake_df = newer_df[common_options].iloc[common_idx]
        fake_df[newer_options_only] = newer_df[newer_options_only].iloc[diff_idx]
        fake_df[self._target]

        # build another temporary dataset for new model
        # with only new options and size difference as a target
        tmp_df_2 = newer_df[newer_options_only].copy()

        # train a new temporary model to predict how new columns affect results
        diff_model = self._train_model(df=tmp_df_2, version=self._newer_version)

        # build a new dataset using

        # build a bigger dataset using both

    # #####- use (or train) an older version model and get its dataset
    # #####- compare older and newer versions dataset columns
    # - train a first new model to predict how new columns affect result
    #     - predict common columns using older model (set everything else to 0)
    #     - train difference (diff or ratio) with new columns
    # - now, build a new dataset with all new columns and a predicted result
    # - train a second new model with this temporary dataset


if __name__ == "__main__":
    # old model
    old_model = Path("/data/tuxai/models/413.json")
    new_model = Path("/data/tuxai/models/415.json")
    old_model.unlink(missing_ok=True)
    tf = TransferLearning(
        older_model_json_path=old_model,
        newer_model_json_path=new_model,
        older_version=413,
        newer_version=415,
        use_cache=False,
    )
    tf.train(limit=5000)
