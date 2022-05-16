#include <cstdio>
#include <pmmintrin.h>
#include <cstdlib>
#include <algorithm>
#include <windows.h>
#include <iostream>
#include <time.h>
#include <immintrin.h>
#include <pthread.h>

using namespace std;
const int N = 10000;
const int M = 100;
const int THREAD_NUM = 4;
long long head, freq;
float input[3][N][N];
float kernel[3][M][M];
float result[N][N];
int inputHeight, inputWidth, channels;
int kernelHeight, kernelWidth;
int outputHeight, outputWidth;
int seg;
pthread_mutex_t mutex;


typedef struct {
    int threadId;
} threadParam_t;



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
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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

    long long tail;
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    printf("%lfms.\n", (tail - head) * 1000.0 / freq);
    //Print
    /*for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            printf("%.0f\t", result[i][j]);

        }
        putchar('\n');
    }*/
}

void conv_simd() {
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
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
    long long tail;
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    printf("%lfms.\n", (tail - head) * 1000.0 / freq);

    //Print
    /*for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            printf("%.0f\t", result[i][j]);

        }
        putchar('\n');
    }*/
}

void *pthread_calc_conv(void *parm) {
    threadParam_t *p = (threadParam_t *) parm;
    int r = p->threadId;
    long long tail;
    for(int i = r * seg; i < (r + 1) * seg; i++) {
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
    pthread_mutex_lock(&mutex);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    printf("Thread %d: %lfms.\n", r, (tail - head) * 1000.0 / freq);
    pthread_mutex_unlock(&mutex);
    pthread_exit(nullptr);
}

void *pthread_simd_calc_conv(void *parm) {
    __m128 t1, t2, s;
    threadParam_t *p = (threadParam_t *) parm;
    int r = p->threadId;
    long long tail;
    for(int i = r * seg; i < (r + 1) * seg; i++) {
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
    pthread_mutex_lock(&mutex);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    printf("Thread %d: %lfms.\n", r, (tail - head) * 1000.0 / freq);
    pthread_mutex_unlock(&mutex);
    pthread_exit(nullptr);
}

void conv_pthread() {
    //Calculate
    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    seg = outputHeight / THREAD_NUM;            //按照OutputHeight进行分割
    mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t thread[THREAD_NUM];
    threadParam_t threadParm[THREAD_NUM];
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < THREAD_NUM; i++) {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], nullptr, pthread_simd_calc_conv, (void *)&threadParm[i]);
    }
    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(thread[i], nullptr);
    }
    pthread_mutex_destroy(&mutex);


    //Print
    /*for(int i = 0; i < outputHeight; i++) {
        for(int j = 0; j < outputWidth; j++) {
            printf("%.0f\t", result[i][j]);

        }
        putchar('\n');
    }*/
    putchar('\n');
    putchar('\n');
}

int main() {

    int args[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000};
    for(int x = 0; x < 22; x++) {
        cout << args[x] << endl;
        int loops = 1;
        channels = 1;
        inputHeight = inputWidth = args[x];
        kernelHeight = kernelWidth = 64;

        //readFromFile();
        randInit();

        for(int i = 0; i < loops; i++) {
            //conv_simd();

            conv();
        }
    }

    return 0;
}
