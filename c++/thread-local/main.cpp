
#include <cstdio>
#include <thread>

constexpr size_t num_coefficients = 5000;
thread_local double c[num_coefficients];


void initialize_c(int k)
{
    for (size_t i = 0; i < num_coefficients; ++i) {
        c[i] = i + k;
    }
}

double calc() {
    double sum;
    for (size_t i = 0; i < num_coefficients; ++i) {
        sum += c[i];
    }
    return sum;    
}

void thread_func(int id)
{
    initialize_c(id);
    double result = calc();
    printf("id %3d: result = %12.0f  (%12.0f)\n", id, result, result - id*num_coefficients);
}

int main()
{
    constexpr int num_threads = 25;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread(thread_func, i));
    }

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
}