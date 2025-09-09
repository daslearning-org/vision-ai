from pythonforandroid.recipe import PythonRecipe

class SympyRecipe(PythonRecipe):
    name = 'sympy'
    version = '1.14.0'
    url = 'https://pypi.io/packages/source/s/sympy/sympy-{version}.tar.gz'
    depends = ['python3', 'setuptools', 'mpmath>=0.19']
    call_hostpython_via_targetpython = False
    install_in_hostpython = False
    install_in_targetpython = True

recipe = SympyRecipe()