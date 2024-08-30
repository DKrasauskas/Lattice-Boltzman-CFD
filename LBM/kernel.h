#pragma once
__device__ void density(cell* cell) {
    cell->rho = 0;
    for (int i = 0; i < 9; i++) {
        cell->rho += cell->f[i];
    }
    //cell->rho /= 100.0f;
}
__device__ void velocity(cell* cell, vec2* e) {
    cell->v.x = 0;
    cell->v.y = 0;
    for (int i = 0; i < 9; i++) {
        cell->v.x += (cell->f[i] * e[i].x);
        cell->v.y += (cell->f[i] * e[i].y);
    }
    //to avoid potential division by 0
    if (cell->rho != 0) {
        cell->v.x /= cell->rho;
        cell->v.y /= cell->rho;
    }
    else {
        cell->v.x = 0;
        cell->v.y = 0;
    }
}
__device__ void equilibrium(cell* cell, vec2* e, float* w) {
    float udotu = cell->v.x * cell->v.x + cell->v.y * cell->v.y;
    for (int i = 0; i < 9; i++) {
        float cdotu = cell->v.x * e[i].x + cell->v.y * e[i].y;
        cell->fe[i] = cell->rho * w[i] * (1.0f + 3.0f * cdotu + 4.5f * cdotu * cdotu - 1.5f * udotu);
    }
}
__global__ void outflow(cell* cells, int n) {
    uint idx = NX - 1;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    cells[idx + idy * n].f[3] = cells[idx - 1 + idy * n].f[3];
    cells[idx + idy * n].f[6] = cells[idx - 1 + idy * n].f[6];
    cells[idx + idy * n].f[7] = cells[idx - 1 + idy * n].f[7];
}
__global__ void walls(cell* cells, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    cells[idx].f[2] = cells[idx + n].f[2];
    cells[idx].f[6] = cells[idx + n].f[6];
    cells[idx].f[5] = cells[idx + n].f[5];
    cells[idx + (NY - 1) * n].f[7] = cells[idx + (NY - 2) * n].f[7];
    cells[idx + (NY - 1) * n].f[4] = cells[idx + (NY - 2) * n].f[4];
    cells[idx + (NY - 1) * n].f[8] = cells[idx + (NY - 2) * n].f[8];
}
__global__ void inflow(cell* cells, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    cells[idy * n].v.x = (idy == 0 || idy == NY - 1) ? 0.0f : max_v;
    cells[idy * n].rho = 100.0f * cells[idx + idy * n].f[0] + 100.0f * cells[idx + idy * n].f[2] + 100.0f * cells[idx + idy * n].f[4] + 2 * (100.0f * cells[idx + idy * n].f[7] + 100.0f * cells[idx + idy * n].f[3] + 100.0f * cells[idx + idy * n].f[6]);
    cells[idy * n].rho /= (1.0f - cells[idx + idy * n].v.x) * 100.0f;

}
__global__ void macro(cell* cells, vec2* e, float* buffer, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    density(&cells[idx + idy * n]);
    velocity(&cells[idx + idy * n], e);
    if (idx == 0) {
        if (true) cells[idy * n].v.x = max_v;
        cells[idy * n].rho = cells[idx + idy * n].f[0] + cells[idx + idy * n].f[2] + cells[idx + idy * n].f[4] + 2 * (cells[idx + idy * n].f[7] + cells[idx + idy * n].f[3] + cells[idx + idy * n].f[6]);
        cells[idy * n].rho /= (1.0f - cells[idx + idy * n].v.x);
    }
    buffer[idx + idy * n] = heatmap_velocity * ((cells[idx + idy * n].v.x * cells[idx + idy * n].v.x + cells[idx + idy * n].v.y * cells[idx + idy * n].v.y));
}
__global__ void eq(cell* cells, vec2* e, float* w, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    equilibrium(&cells[idx + idy * n], e, w);
}

__global__ void inflow2(cell* cells, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    cells[idy * n].f[1] = cells[idy * n].fe[1];
    cells[idy * n].f[5] = cells[idy * n].fe[5];
    cells[idy * n].f[8] = cells[idy * n].fe[8];
}
__global__ void startup(cell* cells, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    for (int i = 0; i < 9; i++) {
        cells[idx + idy * n].f[i] = cells[idx + idy * n].fe[i];
    }
}

__global__ void collide(cell* cells, cell* buffer, float* mask, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    for (int i = 0; i < 9; i++) {
        cells[idx + idy * n].f[i] = cells[idx + idy * n].f[i] - omega * (cells[idx + idy * n].f[i] - cells[idx + idy * n].fe[i]);
        buffer[idx + idy * n].f[i] = cells[idx + idy * n].f[i];
    }
    if (mask[idx + idy * n] > 0.1f) {
        buffer[idx + idy * n].f[6] = cells[idx + idy * n].f[8];
        buffer[idx + idy * n].f[7] = cells[idx + idy * n].f[5];
        buffer[idx + idy * n].f[5] = cells[idx + idy * n].f[7];
        buffer[idx + idy * n].f[8] = cells[idx + idy * n].f[6];
        buffer[idx + idy * n].f[2] = cells[idx + idy * n].f[4];
        buffer[idx + idy * n].f[4] = cells[idx + idy * n].f[2];
        buffer[idx + idy * n].f[1] = cells[idx + idy * n].f[3];
        buffer[idx + idy * n].f[3] = cells[idx + idy * n].f[1];
    }
}

__global__ void set(cell* cells, float* buffer, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    buffer[idx + idy * n] = heatmap_velocity * (cells[idx + idy * n].v.x * cells[idx + idy * n].v.x + cells[idx + idy * n].v.y * cells[idx + idy * n].v.y);
}
__global__ void _init_(cell* cells, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    for (int i = 0; i < 9; i++) {
        cells[idx + idy * n].f[i] = 0;
        cells[idx + idy * n].fe[i] = 0;
    }
    cells[idx + idy * n].v.x = 0;
    cells[idx + idy * n].v.y = 0;
    cells[idx + idy * n].rho = 0;
}

__global__ void gradient(cell* cells, float* buffer, int n) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x + 1;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y + 1;
    float dx = cells[idx + 1 + idy * n].v.y - cells[idx - 1 + idy * n].v.y;
    float dy = cells[idx + (idy + 1) * n].v.x - cells[idx + (idy - 1) * n].v.x;
    buffer[idx + n * idy] = heatmap_curl * (dx - dy);
}

__global__ void obstacle(float* obstacle, int n, float theta) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    //float theta = 0.1f;
    if ((idx * 1.0 - NX / 8) * (idx * 1.0 - NX / 8) + (idy * 1.0 - NY / 2 - theta) * (idy * 1.0 - NY / 2 - theta) < 50) {
        obstacle[idx + idy * NX] = 1.0f;
    }
    else {
        obstacle[idx + idy * NX] = 0.0f;
    }
    //change for airfoil rendering (rotations may cause the simulation to become unstable)
    //float cx = (idx * 1.0f - NX / 4) * cos(theta) - (idy * 1.0f - NY / 2) * sin(theta);
    //float cy = (idx * 1.0f - NX / 4) * sin(theta) + (idy * 1.0f - NY / 2) * cos(theta);
    //float dx = (cx) * 0.006f;
    //if (dx >= 0 && dx < 1) {
    //    float dy = 0.2969 * sqrt(dx) - 0.126 * dx - 0.3516 * dx * dx + 0.2843 * dx * dx * dx - 0.1015 * dx * dx * dx * dx;
    //    obstacle[idx + idy * NX] = 0.0f;
    //    if (abs(dy) > 0.01f * abs(cy)) obstacle[idx + idy * NX] = 1.0f;
    //    //if (dy <= 0 && 0.01f * dy < (y - NY / 4))mem2[x + y * NX] = 1.0f;
    //}
}
__global__ void stream(cell* cells, cell* buffer, float* mask, int nx, int ny, float cx) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint idy = threadIdx.y + blockDim.y * blockIdx.y;
    uint upx = idx == nx - 1 ? 0 : idx + 1;
    uint upy = idy == ny - 1 ? 0 : idy + 1;
    uint downx = (idx == 0) ? nx - 1 : idx - 1;
    uint downy = (idy == 0) ? ny - 1 : idy - 1;
    int n = nx;
    buffer[downx + downy * n].f[7] = cells[idx + idy * n].f[7];
    buffer[downx + upy * n].f[6] = cells[idx + idy * n].f[6];
    buffer[downx + idy * n].f[3] = cells[idx + idy * n].f[3];
    buffer[idx + downy * n].f[4] = cells[idx + idy * n].f[4];
    buffer[upx + downy * n].f[8] = cells[idx + idy * n].f[8];
    buffer[upx + upy * n].f[5] = cells[idx + idy * n].f[5];
    buffer[idx + upy * n].f[2] = cells[idx + idy * n].f[2];
    buffer[upx + idy * n].f[1] = cells[idx + idy * n].f[1];
}