// Struct copy and field access benchmark (ABI, memcpy, field offset codegen)
#include <stdio.h>
#include <string.h>

#define N 2000000

typedef struct {
    double x, y, z;
    int id;
    char name[20];
} Particle;

typedef struct {
    Particle particles[4];
    int count;
    double total_energy;
} ParticleGroup;

static Particle make_particle(int i) {
    Particle p;
    p.x = (double)i * 0.1;
    p.y = (double)i * 0.2;
    p.z = (double)i * 0.3;
    p.id = i;
    p.name[0] = 'P';
    p.name[1] = '0' + (i % 10);
    p.name[2] = '\0';
    return p;
}

static double particle_distance(Particle a, Particle b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

static ParticleGroup make_group(int base) {
    ParticleGroup g;
    g.count = 4;
    g.total_energy = 0;
    for (int i = 0; i < 4; i++) {
        g.particles[i] = make_particle(base + i);
    }
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            g.total_energy += particle_distance(g.particles[i], g.particles[j]);
    return g;
}

int main(void) {
    double total = 0;
    for (int i = 0; i < N; i++) {
        ParticleGroup g = make_group(i);
        total += g.total_energy;
    }
    printf("struct_copy total: %.2f\n", total);
    return 0;
}
