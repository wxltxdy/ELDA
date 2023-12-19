import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: 当python_require = >= 3.8 '时直接导入(不需要条件)
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # 主要用于获取项目的当前版本号，并将其存储在特定的变量__version__
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
