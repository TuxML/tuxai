import logging
from typing import get_args
from pathlib import Path
from flask import Flask, request, render_template

from tuxai2.report import Report, TARGETS, FEATURE_IMPORTANCE_ARGS
from tuxai2.misc import get_config, config_logger, version_str

LOG = logging.getLogger(__name__)


app = Flask(__name__, template_folder="html", static_folder="html")


@app.route("/", methods=["GET", "POST"])
def index():
    config = get_config()
    versions = sorted(
        [version_str(version) for version in config["report"]["versions"]]
    )
    # fi_args = [fi_arg for fi_arg in get_args(FEATURE_IMPORTANCE_ARGS) if fi_arg != ""]
    fi_args = get_args(FEATURE_IMPORTANCE_ARGS)
    if request.method == "POST":
        target_filter = request.form.get("target_filter", "both")
        if target_filter == "both":
            target_filter_f = TARGETS
        else:
            target_filter_f = target_filter
        if is_top := request.form.get("is_top", False):
            top_n = request.form.get("top_n", "10")
            is_top_f = int(top_n)
        else:
            is_top_f = None
            top_n = "10"
        merge = request.form.get("merge", "and")
        has_version = request.form.getlist("has_versions")
        has_version_f = has_version if has_version else None
        has_not_version = request.form.getlist("has_not_versions")
        has_not_version_f = has_not_version if has_not_version else None
        has_collinearity = request.form.get("has_colinearity", False)
        has_not_kconfig = request.form.get("has_not_kconfig", False)
        has_unbalanced_yes_no_ratio = request.form.get(
            "has_unbalanced_yes_no_ratio", False
        )
        feature_importance = request.form.get("feature_importance", "")

        # search

        report = Report(json_path=config["web"]["json"])
        result = report.find(
            merge=merge,
            target_filter=target_filter_f,
            has_version=has_version_f,
            has_not_version=has_not_version_f,
            is_top=is_top_f,
            has_collinearity=has_collinearity,
            has_not_kconfig=has_not_kconfig,
            has_unbalanced_yes_no_ratio=has_unbalanced_yes_no_ratio,
            feature_importance=feature_importance,
            verbose=True,
        )

        return render_template(
            "index.html",
            result=result,
            versions=versions,
            fi_args=fi_args,
            merge=merge,
            target_filter=target_filter,
            has_version=has_version,
            has_not_version=has_not_version,
            is_top=is_top,
            top_n=top_n,
            has_collinearity=has_collinearity,
            has_not_kconfig=has_not_kconfig,
            has_unbalanced_yes_no_ratio=has_unbalanced_yes_no_ratio,
            feature_importance=feature_importance,
        )
    return render_template(
        "index.html",
        result={},
        versions=versions,
        fi_args=fi_args,
        merge="and",
        target_filter="both",
        has_version=[],
        has_not_version=[],
        is_top=False,
        top_n="10",
        has_collinearity=False,
        has_not_kconfig=False,
        has_unbalanced_yes_no_ratio=False,
        feature_importance="",
    )


if __name__ == "__main__":
    config_logger()
    app.run(debug=True)
