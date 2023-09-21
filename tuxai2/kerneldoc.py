import re
from pathlib import Path
from string import Template
import requests
import tarfile
import os
from collections import defaultdict
import logging

from tuxai.misc import get_config, config_logger

LOG = logging.getLogger(__name__)

ARCH_DIR = "/arch/"


class KernelDoc:
    """Download, extract data and generate documentation."""

    def __init__(self, version: str | float | int) -> None:
        self._config = get_config()
        self._worspace = Path(self._config["kernel"]["path"])
        self._version = self._version_format(version)
        self._download()

    def _version_format(self, version: str | float | int) -> str:
        """Version format."""
        version = str(version)
        version = version if "." in version else f"{version[0]}.{version[1:]}"
        return version

    @property
    def _src_path(self) -> Path:
        """Get source path."""
        return self._worspace / "src" / self._version

    def get_merged_kconfig_options(self) -> dict[str, dict]:
        """Download and read all available kconfig files."""
        merged_options = defaultdict(dict)
        for kconfig_path in self._get_kconfig_path_list():
            with open(kconfig_path, "r") as kconfig_file:
                content = kconfig_file.read()
            options = self._parse_options(content)
            for option, option_content in options.items():
                merged_options[option][
                    str(kconfig_path.relative_to(self._src_path))
                ] = {
                    "content": option_content,
                    "help": self._extract_help_section(option_content),
                    "local_path": str(
                        kconfig_path,
                    ),
                }
        return dict(merged_options)

    def _download(self) -> None:
        """Download and extract kernel."""
        if self._src_path.exists():
            return

        major, minor = self._version.split(".")
        try_maj_min = ((major, minor), (major, int(minor)))
        for major, minor in try_maj_min:
            kernel_url = Template(self._config["kernel"]["tar_url"]).substitute(
                {"major": major, "minor": minor}
            )
            tar_target = self._worspace / "tar" / kernel_url.rsplit("/", 1)[1]
            # download
            if not tar_target.exists():
                LOG.debug(f"downloading: {tar_target}")
                r = requests.get(kernel_url, allow_redirects=True)
                tar_target.parent.mkdir(parents=True, exist_ok=True)
                open(tar_target, "wb").write(r.content)
            # uncompress
            self._src_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(tar_target) as tar:
                    tar.extractall(str(self._src_path))
                    break
            except tarfile.ReadError:
                LOG.debug(f"failed to download/extract: {tar_target}")

    def _get_kconfig_path_list(self) -> list[Path]:
        """Get all available kconfig paths."""
        arch_wl = self._config["kernel"]["kconfig_arch_whitelist"]
        bls = [item.lower() for item in self._config["kernel"]["kconfig_blacklist"]]

        kconfig_files = list()
        for root, dirs, files in os.walk(self._src_path, topdown=True):
            # /arch/x86
            if ARCH_DIR in root:
                keep = False
                arch_id = root.index(ARCH_DIR) + len(ARCH_DIR)
                for arch in arch_wl:
                    if root[arch_id:].startswith(arch):
                        keep = True
                        break
                if not keep:
                    continue

            for f in files:
                if f.lower().startswith("kconfig"):
                    kconfig_file = Path(root) / f
                    # blacklist
                    str_path = str(kconfig_file).lower()
                    keep = True
                    for bl in bls:
                        if bl in str_path:
                            keep = False
                            break
                    if keep:
                        kconfig_files.append(kconfig_file)

        return kconfig_files

    def _extract_help_section(self, content: str) -> str:
        """Parse and extract help section from option in kconfig file."""
        # extract help section until next empty line or end of string
        # pattern = r"\thelp\n([\s\S]*?)(?=\n\n|\Z)"
        pattern = r"(?:help|\-\-\-help\-\-\-)\n\s*([\s\S]*?)(?=\n\n|\Z)"
        match = re.search(pattern, content)

        if match:
            return match.group(1).strip()
        else:
            if "help" in content:
                pass
            return ""

    def _parse_options(self, content: str) -> dict[str, str]:
        """Parse extracted text and store in dictionnary."""
        # extract option name and corresponding content
        if "SND_SOC_STORM" in content:
            pass
        pattern = r"config\s+(\w+)\s+([\s\S]*?)(?=\nconfig|\Z)"
        matches = re.finditer(pattern, content)
        return {match.group(1): match.group(2).strip() for match in matches}


if __name__ == "__main__":
    config_logger()
    kd = KernelDoc(version=413)
    d = kd.get_merged_kconfig_options()
    print(d)
    # d["DEBUG_RWSEMS"]
    # d["UBSAN_NULL"]
    d["KASAN"]
    {k: v for k, v in d.items() if len(v) > 1}

    # kd.get_html_options_page(fir_cache="fi_const_2023")

    # options = kd._get_merged_kconfig_options()
    # content = kd._get_page_content("Kconfig.debug")
    # options = kd._parse_options(content)
    # kconfigs = kd._get_kconfig_filenames()
    # print(options)
    # https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.13.tar.xz
