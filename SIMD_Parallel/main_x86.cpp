#include <cstdio>
#include <pmmintrin.h>
#include <cstdlib>
#include <algorithm>
#include <windows.h>
#include <iostream>
#include <time.h>

using namespace std;
const int N = 10000;
const int M = 100;
float input[3][N][N];
float kernel[3][M][M];
float result[N][N];
int inputHeight, inputWidth, channels;
int kernelHeight, kernelWidth;
int outputHeight, outputWidth;

class Timer
{
    public:
        Timer(): start_(), end_() {}

        void Start() {
            QueryPerformanceCounter(&start_);
        }

        void Stop() {
            QueryPerformanceCounter(&end_);
        }

        double GetElapsedMilliseconds() {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            return (end_.QuadPart - start_.QuadPart) * 1000.0 / freq.QuadPart;
        }


    protected:

    private:
        LARGE_INTEGER start_;
        LARGE_INTEGER end_;
};

void readFromFile() {
     FILE *fp;
    fp = fopen("input.txt", "r");
    if(fp == NULL) {
        printf("File open error\n");
        return;
    }
    fscanf(fp, "%d%d%d", &inputHeight, &inputWidth, &channels);
    for(int i = 0; i < inputHeight; i++) {
        for(int j = 0; j < inputWidth; j++) {
            for(int c = 0; c < channels; c++) {
                fscanf(fp, "%f", &input[c][i][j]);
            }
        }
    }
    fscanf(fp, "%d%d", &kernelHeight, &kernelWidth);
    for(int i = 0; i < kernelHeight; i++) {
        for(int j = 0; j < kernelWidth; j++) {
            for(int c = 0; c < channels; c++) {
                fscanf(fp, "%f", &kernel[c][i][j]);
            }
        }
    }
}

void randInit() {
    srand(time(NULL));
    for(int c = 0; c < 3; c++) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                input[c][i][j] = rand() %256;
            }
        }
    }
    for(int c = 0; c < 3; c++) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < M; j++) {
                kernel[c][i][j] = rand();
            }
        }
    }
}


void conv() {
    //Calculate
    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;

    for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            float sum = 0;
            for(int c = 0; c < channels; c++) {
                for(int k = 0; k < kernelHeight; k++) {         //kernelWidth
                    for(int t = 0; t < kernelWidth; t++) {
                        sum += input[c][i+k][j+t] * kernel[c][k][t];
                    }
                }
            }
            result[i][j] = sum;
        }
    }

    //Print
    /*for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            printf("%.0f\t", result[i][j]);

        }
        putchar('\n');
    }*/
}

void conv_simd() {
    __m128 t1, t2, s;
    //Calculate
    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;

    for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            float temp = 0;
            s = _mm_setzero_ps();
            for(int c = 0; c < channels; c++) {
                for(int k = 0; k < kernelHeight; k++) {         //kernelWidth
                    for(int t = kernelWidth - 4; t >= 0; t -= 4) {
                        t1 = _mm_loadu_ps(input[c][i+k]+j+t);
                        t2 = _mm_loadu_ps(kernel[c][k]+t);
                        t1 = _mm_mul_ps(t1, t2);
                        s = _mm_add_ps(s, t1);
                    }

                    for(int t = (kernelWidth % 4) - 1; t >= 0; t--) {
                        temp += input[c][i+k][j+t] * kernel[c][k][t];
                    }
                }
            }
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            _mm_store_ss(result[i]+j, s);
            result[i][j] += temp;
        }
    }

    //Print
    /*for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            printf("%.0f\t", result[i][j]);

        }
        putchar('\n');
    }*/
}

//valid mode

int main() {

    int args[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000};
    for(int x = 0; x < 22; x++) {
        int loops = 10;
        channels = 1;
        inputHeight = inputWidth = args[x];
        kernelHeight = kernelWidth = 64;

        //readFromFile();
        randInit();

        Timer* timer = new Timer();
        timer->Start();
        for(int i = 0; i < loops; i++) {
            conv_simd();
        }
        timer->Stop();
        printf("%fms\n", timer->GetElapsedMilliseconds());
    }

    return 0;
}
