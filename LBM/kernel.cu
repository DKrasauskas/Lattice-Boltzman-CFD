
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <corecrt_malloc.h>
#include "include/glad/glad.h"
#include "include/GLFW/glfw3.h"
#define uint unsigned int
#include "grid.h"
#include "Shader.h"
#include "Buffer.h"


const unsigned int SCR_WIDTH = 1900;
const unsigned int SCR_HEIGHT = 600;


#define NX  600
#define NY  200
#define SKIP_ITTER 20000

#define Re  80
#define max_v 0.04f
#define niu max_v * 10 / Re
#define omega 1.97 // 1 / (3 * niu + 0.5) // relaxation

#define heatmap_velocity 140
#define heatmap_curl 50

#define show_airfoil 0 // 1 for true 0 for false

struct vec2 {
    float x, y;
};

struct cell {
    float f[9] = { 0, 0, 0, 0, 0, 0, 0, 0,0 };
    float fe[9] = { 0, 0, 0, 0, 0, 0, 0, 0,0 };
    float rho;
    vec2 v;
};

vec2 es[9] = {
    { 0,  0},
    { 1,  0},
    { 0,  1},
    {-1,  0},
    { 0, -1},
    { 1,  1},
    {-1,  1},
    {-1, -1},
    { 1, -1},
};

float weights[9] = { 
    (float)4 / 9,
    (float)1 / 9, (float)1 / 9, (float)1 / 9, (float)1 / 9,
    (float)1 / 36, (float)1 / 36, (float)1 / 36, (float)1 / 36
};


#include "kernel.h"

__device__ cell* cells;
__device__ cell* aux;
__device__ cell* aux2;
__managed__ vec2* e;
__managed__ float* w;
__managed__ float* buffer;
__device__ float* mask;

dim3 block(30, 20);
dim3 thread(20, 10);
dim3 boundary(1, NY);
dim3 boundary2(1, NY);
dim3 str(NX, NY);
dim3 str2(NX -2, NY - 2);
float vx = 0.0f;
float yc = 0.0f;

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        vx += 0.1f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        vx -= 0.1f;
}
void swap_buffers() {
    aux2 = cells;
    cells = aux;
    aux = aux2;
}

void compute() {
    macro<<<block, thread>>>(cells, e, buffer,  NX);
    cudaDeviceSynchronize();   
    eq << <block, thread >> > (cells, e, w, NX);
    cudaDeviceSynchronize();
    inflow2 << <boundary2, 1 >> > (cells, NX);
    cudaDeviceSynchronize();
    collide<<<block, thread>>>(cells, aux,  mask, NX);
    cudaDeviceSynchronize();
    swap_buffers();
    stream<< <block, thread >> > (cells, aux, mask, NX, NY, vx);
    cudaDeviceSynchronize();
    swap_buffers();
    outflow << <boundary, 1 >> > (cells, NX);
    cudaDeviceSynchronize();
    walls<<<NX, 1>>>(cells, NX);
    cudaDeviceSynchronize();
    if (vx > 1.0f) {
        gradient << <str2, 1 >> > (cells, buffer, NX);
        cudaDeviceSynchronize();
    }
}

// allocates variables and sets up initial conditions
void begin() {
    cudaMalloc(&cells, sizeof(cell) * NX * NY);
    cudaMalloc(&aux, sizeof(cell) * NX * NY);
    cudaMalloc(&aux2, sizeof(cell) * 1);
    cudaMallocManaged(&e, sizeof(vec2) * 9);
    cudaMallocManaged(&w, sizeof(float) * 9);
    cudaMallocManaged(&buffer, sizeof(float) * NX * NY);
    cudaMalloc(&mask, sizeof(float) * NX * NY);
    cell* mem = (cell*)malloc(sizeof(cell) * NX * NY);
    float* mem2 = (float*)malloc(sizeof(float) * NX * NY);

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            if (show_airfoil) {
                float theta = 0.1f;
                float cx = (x - NX / 8) * cos(theta) - (y - NY / 2) * sin(theta);
                float cy = (x - NX / 8) * sin(theta) + (y - NY / 2) * cos(theta);
                float dx = (cx) * 0.012f;
                if (dx >= 0 && dx < 1) {
                    // NACA symmetric airfoil function
                    float dy = 0.2969 * sqrt(dx) - 0.126 * dx - 0.3516 * dx * dx + 0.2843 * dx * dx * dx - 0.1015 * dx * dx * dx * dx;
                    mem2[x + y * NX] = 0.0f;
                    if (abs(dy) > 0.02f * abs(cy)) mem2[x + y * NX] = 1.0f;
                }
            }
            else {
                if ((x - NX / 8) * (x - NX / 8) + (y - NY / 2) * (y - NY / 2) < 100) {
                    mem2[x + y * NX] = 1.0f;
                }
                else {
                    mem2[x + y * NX] = 0.0f;
                }
            }
        }
    }
    for (int i = 0; i < 9; i++) {
        e[i].x = es[i].x;
        e[i].y = es[i].y;
        w[i] = weights[i];
    }
    for (int i = 0; i < NX * NY; i++) {
        mem[i].v.x = 0.0;
        mem[i].v.y = 0;
        mem[i].rho = .01f;
        buffer[i] = 0.0f;

    }
    cudaMemcpy(cells, mem, sizeof(cell) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(mask, mem2, sizeof(float) * NX * NY, cudaMemcpyHostToDevice);
    eq << <block, thread >> > (cells, e, w, NX);
    cudaDeviceSynchronize();
    startup << <block, thread >> > (cells, NX);
    cudaDeviceSynchronize();
    free(mem);
    free(mem2);
}

void terminate() {
    cudaFree(cells);
    cudaFree(aux);
    cudaFree(aux2);
    cudaFree(mask);
    cudaFree(e);
    cudaFree(w);
    cudaFree(buffer);
}

int main()
{
   
    begin();
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LBM", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    Shader vertex("vertex.glsl", "fragment.glsl");
    Cshader computes("compute.glsl");
    Grid gd = grid(2);
    Buffer buff((void*)gd.vertices, (void*)gd.indices, gd.v_size, gd.i_size);
    GLuint vxb, vyb;
    unsigned int texture;
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, NX, NY, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    float begin = clock();
    float fps = 0;
    int counts = 0;
    int i = 0;
    while (!glfwWindowShouldClose(window))
    {  
        if (i > SKIP_ITTER) {
            glGenBuffers(1, &vxb);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vxb);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vxb);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * NX * NY, buffer, GL_STATIC_DRAW); //sizeof(data) only works for statically sized C/C++ arrays.
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vxb);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind  
            glUseProgram(computes.ID);        
            glUniform1f(glGetUniformLocation(computes.ID, "w"),vx);             
            glDispatchCompute(NX / 10, NY / 10, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            processInput(window);
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(vertex.ID);        
            glBindVertexArray(buff.VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glDrawElements(GL_TRIANGLES, 6 * gd.v_size / sizeof(float), GL_UNSIGNED_INT, (void*)0);
            glfwSwapBuffers(window);
            glfwPollEvents();
            compute();
        }
        else {
           
            compute();
        }   
        float dt = 1 / (((float)clock() - begin) / CLOCKS_PER_SEC);     
        fps += dt < 500 ? dt : 500;
        counts += 1;
        if (counts == 10) {
            cout << (float)fps/counts << "FPS " << i  << endl;
          
            counts = 0;
            fps = 0;
        }
        i++;      
        begin = clock();      
    }
    return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

