
#include <cstdio>
#include <thread>

constexpr size_t num_coefficients = 5000;


void initialize_c(double c[], int k)
{
    for (size_t i = 0; i < num_coefficients; ++i) {
        c[i] = i + k;
    }
}

double calc(double c[]) {
    double sum = 0.0;
    for (size_t i = 0; i < num_coefficients; ++i) {
        sum += c[i];
    }
    return sum;    
}

void thread_func(int id)
{
    // `thread_local` here implies `static`; according to the C++ standard
    // (e.g. https://port70.net/%7Ensz/c/c%2B%2B/c%2B%2B11_n3337.txt),
    // "When thread_local is applied to a variable of block scope the
    // storage-class-specifier static is implied if it does not appear
    // explicitly."
    thread_local double c[num_coefficients];
    initialize_c(c, id);
    double result = calc(c);
    printf("id %3d: result = %12.0f  (%12.0f)\n", id, result, result - id*num_coefficients);
}

int main()
{
    constexpr int num_threads = 25;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread(thread_func, i));
    }

    for (auto& t: threads) {
        t.join();
    }
}