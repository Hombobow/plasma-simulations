#include <iostream>
#include <chrono>
#include "initialization.h"
using namespace std;


int main(){
    initialize();
    // output_data(0);

    auto t_start = chrono::high_resolution_clock::now();
    run_loop();
    auto t_end = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double>(t_end - t_start).count();
    cout << "\nrun time: " << elapsed << " s" << endl;

    return 0;
}
