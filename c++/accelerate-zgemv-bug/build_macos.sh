clang++ check.cpp -DACCELERATE_NEW_LAPACK -framework Accelerate -std=c++17 -o check

clang demo_issue.c -DACCELERATE_NEW_LAPACK -framework Accelerate -o demo_issue
