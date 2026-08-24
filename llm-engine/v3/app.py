"""Flask application entry point.

Prefer running this module with ``python -m v3.app``.  The direct-execution
branch also keeps ``python v3/app.py`` working by importing from the package
root instead of treating this file as a standalone module.
"""

if __package__:
    from .factory import create_app
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from v3.factory import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
