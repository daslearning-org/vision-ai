from pythonforandroid.recipe import PythonRecipe

class MpmathRecipe(PythonRecipe):
    # This should match the name of the directory containing this recipe
    name = 'mpmath'
    # Current stable version of mpmath
    version = '1.3.0' # Or the version you want, e.g., '1.2.1' for safety

    # URL to download the source. {version} is a placeholder.
    url = 'https://pypi.io/packages/source/m/mpmath/mpmath-{version}.tar.gz'

    # If it had C extensions, you might list things like 'hostpython3', 'setuptools', etc.
    depends = ['python3', 'setuptools']

    call_hostpython_via_targetpython = False
    install_in_hostpython = False
    install_in_targetpython = True

recipe = MpmathRecipe()