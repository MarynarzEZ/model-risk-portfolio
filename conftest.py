"""Makes `import src...` work when tests run from anywhere.

pytest puts the directory holding the root `conftest.py` on `sys.path`,
so this file existing is the whole trick. The alternative -- installing
the project as a package -- would be right for something distributed,
but this repository is run from a checkout.
"""
