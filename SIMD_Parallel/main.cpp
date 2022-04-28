#include <cstdio>
#include <pmmintrin.h>
#include <cstdlib>
#include <algorithm>
#include <windows.h>
#include <iostream>
#include <time.h>

using namespace std;
const int N = 1000;
float input[3][N][N];
float kernel[3][N][N]  ;
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
                kernel[c][i][j] = rand();
            }
        }
    }
}


void conv() {
    //Calculate
    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;
    float result[outputHeight][outputWidth];
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

//valid mode

int main() {

    int loops = 100;
    channels = 1;
    inputHeight = inputWidth = 10;
    kernelHeight = kernelWidth = 4;

    //readFromFile();
    randInit();

    Timer* timer = new Timer();
    timer->Start();
    for(int i = 0; i < loops; i++) {
        conv();
    }
    timer->Stop();
    printf("%fms\n", timer->GetElapsedMilliseconds());

    return 0;
}
