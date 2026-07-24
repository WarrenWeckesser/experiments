import numpy as np

a = np.array([[10], [20], [30]])
b = np.array([-1, 5])
out = np.zeros((4, 3, 2))

it = np.nditer((a, b, out),
               flags=["multi_index"],
               op_flags=[["readonly"], ["readonly"], ["writeonly"]],
               op_axes=[[-1, 0, 1], [-1, -1, 0], [0, 1, 2]])
it.remove_axis(0)

iter_count = 0
with it:
    for i in it:
        print(f"{iter_count = }")
        print(i)
        print(f"{it.multi_index = }")
        print()
        iter_count += 1

# iter_count = 0
# with it:
#     for (a1, b1) in it:
#         print(f"{iter_count = }")
#         print(f"{a1 = }")
#         print(f"{b1 = }")
#         print(f"{it.multi_index = }")
#         print()
#         iter_count += 1

# it = np.nditer((a, b),
#                flags=["multi_index"],
#                op_flags=[["readonly"], ["readonly"]])

# iter_count = 0
# with it:
#     for (a1, b1) in it:
#         print(f"{iter_count = }")
#         print(f"{a1 = }")
#         print(f"{b1 = }")
#         print(f"{it.multi_index = }")
#         print()
#         iter_count += 1
