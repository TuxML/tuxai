import urllib.request
import re
import http
from pathlib import Path
from string import Template
import requests
import tarfile
import os
from collections import defaultdict

from tuxai.misc import get_config


KERNEL_TAR_URL = "https://git.kernel.org/torvalds/t/linux-$version.tar.gz"
# deprecated
DOC_URL = "https://github.com/torvalds/linux/blob/master/lib"
RAW_DOC_URL = "https://raw.githubusercontent.com/torvalds/linux/master/lib"


class Display:
    """"""

    def __init__(self) -> None:
        pass


class KernelDocumentation:
    """Download, extract data and generate documentation."""

    def __init__(self, version: str | float | int) -> None:
        config = get_config()
        self._worspace = Path(config["kernel"]["path"])
        self._version = self._version_format(version)
        self._download()

    def _version_format(self, version: str | float | int) -> str:
        """Version format"""
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
                merged_options[option][kconfig_path] = {
                    "help": self._extract_help_section(option_content)
                }
        return merged_options

    def _download(self) -> None:
        """Download and extract kernel."""
        if self._src_path.exists():
            return

        kernel_url = Template(KERNEL_TAR_URL).substitute({"version": self._version})
        tar_target = self._worspace / "tar" / kernel_url.rsplit("/", 1)[1]
        # download
        if not tar_target.exists():
            r = requests.get(kernel_url, allow_redirects=True)
            tar_target.parent.mkdir(parents=True, exist_ok=True)
            open(tar_target, "wb").write(r.content)
        # uncompress
        self._src_path.parent.mkdir(parents=True, exist_ok=True)
        tar_file = tarfile.open(tar_target)
        tar_file.extractall(str(self._src_path))
        tar_file.close()

    def _get_kconfig_path_list(self) -> list[Path]:
        """Get all available kconfig paths."""
        kconfig_files = list()
        for root, dirs, files in os.walk(self._src_path, topdown=True):
            for f in files:
                if f.lower().startswith("kconfig"):
                    kconfig_files.append(Path(root) / f)

        return kconfig_files

    def _extract_help_section(self, content: str) -> str:
        """Parse and extract help section from option in kconfig file."""

        # pattern = r"\n\thelp([A-Z_]+)\n(.*?)\n\n"
        # matches = re.findall(pattern, content, re.DOTALL)
        # return {match[0]: match[1].strip() for match in matches}
        lines = [line.strip() for line in content.split("\n")]
        if "help" in lines:
            help_index = lines.index("help") + 1
            return " ".join(lines[help_index:])
        return ""

    def _parse_options(self, content: str) -> dict[str, str]:
        """Parse extracted text and store in dictionnary."""
        pattern = r"config ([A-Z_]+)\n(.*?)\n\n"
        matches = re.findall(pattern, content, re.DOTALL)
        return {match[0]: match[1].strip() for match in matches}


if __name__ == "__main__":
    kd = KernelDocumentation(version=413)
    d = kd.get_merged_kconfig_options()
    print(d)
    d["DEBUG_RWSEMS"]
    d["UBSAN_NULL"]
    d["KASAN"]
    # options = kd._get_merged_kconfig_options()
    # content = kd._get_page_content("Kconfig.debug")
    # options = kd._parse_options(content)
    # kconfigs = kd._get_kconfig_filenames()
    # print(options)
    # https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.13.tar.xz
