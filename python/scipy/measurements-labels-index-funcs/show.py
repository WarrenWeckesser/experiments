import inspect
import scipy.ndimage as ndi


callable_public_names = [name for name in dir(ndi) if callable(getattr(ndi, name)) and not name.startswith('_')]
funcs = []
for name in callable_public_names:
    obj = getattr(ndi, name)
    sig = inspect.signature(obj)
    params = sig.parameters
    if "labels" in params and "index" in params:
        funcs.append(obj)

print("Functions in ndimage with parameters 'labels' and 'index':")
for func in funcs:
    print(f"   {func.__name__}")
