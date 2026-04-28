"""Project-root and output-directory helpers for notebooks and scripts."""

from __future__ import annotations

from pathlib import Path


ROOT_MARKERS = ("src", "data", "notebooks")


def find_project_root(start=None, markers=ROOT_MARKERS):
    """Walk upward from start until all project marker paths exist."""
    path = Path.cwd() if start is None else Path(start)
    path = path.resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate
    marker_list = ", ".join(markers)
    raise FileNotFoundError(f"Could not find project root containing: {marker_list}")


def data_dir(root=None):
    """Return the project data directory."""
    return find_project_root(root) / "data"


def figures_dir(*parts, root=None, create=True):
    """Return a figure directory under figures/hopf model, creating it by default."""
    path = find_project_root(root) / "figures" / "hopf model"
    for part in parts:
        path = path / part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def paper_dir(root=None):
    """Return the project Paper directory."""
    return find_project_root(root) / "Paper"
