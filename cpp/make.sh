g++-8 pprlib.cpp -std=c++11 -march=native -fopenmp -Ofast -fPIC -o pprlib.so -shared -lgomp
g++-8 kmeanslib.cpp -std=c++11 -march=native -fopenmp -Ofast -fPIC -o kmeanslib.so -shared -lgomp