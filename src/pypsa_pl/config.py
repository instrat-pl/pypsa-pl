from pathlib import Path
import inspect
import logging
import locale


def project_dir(*path):
    root = (
        Path(inspect.getframeinfo(inspect.currentframe()).filename).resolve().parents[2]
    )
    return Path(root, *path)


def data_dir(*path):
    return project_dir("data", *path)


def plots_dir(*path):
    return project_dir("plots", *path)


logging.basicConfig(
    format="%(asctime)s [%(levelname)-s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

try:
    locale.setlocale(locale.LC_COLLATE, "pl_PL.UTF-8")
except:
    logging.warning("Could not set locale to pl_PL.UTF-8")


def local_sort(val):
    if isinstance(val, str):
        return locale.strxfrm(val)
    else:
        # vectorized version
        return [locale.strxfrm(x) for x in val]
