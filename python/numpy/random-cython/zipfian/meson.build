project('random-cython', 'c', 'cpp', 'cython')

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)

cy = meson.get_compiler('cython')

base_cython_args = []
if cy.version().version_compare('>=3.1.0')
  base_cython_args += ['-Xfreethreading_compatible=True']
endif

_numpy_abs = run_command(py3, ['-c',
               'import os; os.chdir(".."); import numpy; print(os.path.abspath(numpy.get_include() + "../../.."))'],
                         check: true).stdout().strip()

npy_include_path = _numpy_abs / '_core' / 'include'

py3.extension_module(
    'random_variates',
    'random_variates.pyx',
    install: false,
    include_directories: [npy_include_path],
    cython_args: base_cython_args,
)
