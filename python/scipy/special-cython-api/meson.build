project('special-cythion-api-demo', 'c', 'cython')

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)

cy = meson.get_compiler('cython')

base_cython_args = []
if cy.version().version_compare('>=3.1.0')
  base_cython_args += ['-Xfreethreading_compatible=True']
endif

py3.extension_module(
    'special_cython_api_demo',
    'special_cython_api_demo.pyx',
    install: false,
    cython_args: base_cython_args,
)
