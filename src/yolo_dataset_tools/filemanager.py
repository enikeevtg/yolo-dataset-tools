import os
import shutil
from pathlib import Path


class FileManager:
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return (
            (self.base_dir / path).resolve()
            if self.base_dir
            else path.resolve()
        )

    def basename(self, path: str | Path):
        return path.split("/")[-1]

    # ==== Dirs ====

    def is_dir(self, path: str | Path) -> bool:
        return self.resolve_path(path).is_dir()

    def create_dir(self, path: str | Path, exist_ok: bool = True) -> Path:
        path = self.resolve_path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        return path

    def clear_dir(self, path: str | Path) -> Path | None:
        path = self.resolve_path(path)
        if path.exists():
            shutil.rmtree(path)
            path = self.create_dir(path)
        return path

    def move_dir(
        self,
        src: str | Path,
        dst: str | Path,
        auto_rename: bool = True,
        overwrite: bool = False,
    ):
        self._move(
            src=src,
            dst=dst,
            auto_rename=auto_rename,
            overwrite=overwrite,
        )

    def copy_dir(
        self,
        src: str | Path,
        dst: str | Path,
        auto_rename: bool = True,
        overwrite: bool = False,
    ):
        self._copy(
            src=src,
            dst=dst,
            auto_rename=auto_rename,
            overwrite=overwrite,
        )

    def remove_dir(self, path: str | Path):
        path = self.resolve_path(path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

    # ==== Files ====

    def is_file(self, path: str | Path) -> bool:
        return self.resolve_path(path).is_file()

    def copy_file(
        self,
        src: str | Path,
        dst: str | Path,
        auto_rename: bool = True,
        overwrite: bool = False,
    ):
        self._copy(
            src=src,
            dst=dst,
            auto_rename=auto_rename,
            overwrite=overwrite,
        )

    def move_file(
        self,
        src: str | Path,
        dst: str | Path,
        auto_rename: bool = True,
        overwrite: bool = False,
    ):
        self._move(
            src=src,
            dst=dst,
            auto_rename=auto_rename,
            overwrite=overwrite,
        )

    def remove_file(self, path: str | Path):
        path = self.resolve_path(path)
        if path.exists() and path.is_file():
            path.unlink()

    def _move(
        self,
        src: str | Path,
        dst: str | Path,
        auto_rename: bool = True,
        overwrite: bool = False,
    ):
        src = self.resolve_path(src)
        dst = self.resolve_path(dst)
        if dst.exists():
            if auto_rename:
                dst = self._increment_name(dst)
            elif not overwrite:
                raise FileExistsError(f"File {dst} already exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)

    def _copy(
        self,
        src: str | Path,
        dst: str | Path,
        auto_rename: bool = True,
        overwrite: bool = False,
    ):
        src = self.resolve_path(src)
        dst = self.resolve_path(dst)
        if dst.exists():
            if auto_rename:
                dst = self._increment_name(dst)
            elif not overwrite:
                raise FileExistsError(f"File {dst} already exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            shutil.copytree(src, dst)

    @staticmethod
    def _increment_name(path: str | Path) -> Path:
        path = Path(path)
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        incr = 1
        while True:

            new_name = f"{stem}({incr}){suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            incr += 1


filemanager = FileManager(".")
