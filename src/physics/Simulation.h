#pragma once
#include "Particle.h"
#include <vector>
#include <cmath>
#include <random>

extern float timeScale;

void computeForces(std::vector<Particle>& particles) {
    int n = (int)particles.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        Vec3 acc{};
        for (int j = 0; j < n; j++) {
            if (i==j) continue;
            Vec3  d     = particles[j].pos - particles[i].pos;
            float dist2 = d.norm2() + SOFTENING*SOFTENING;
            float dist  = std::sqrt(dist2);
            acc += d * (G * particles[j].mass / (dist2 * dist));
        }
        particles[i].acc = acc;
    }
}

void integrate(std::vector<Particle>& particles) {
    float dt = DT_BASE * timeScale;
    for (auto& p : particles) {
        p.vel += p.acc * (dt * 0.5f);
        p.pos += p.vel * dt;
    }
    computeForces(particles);
    for (auto& p : particles) {
        p.vel += p.acc * (dt * 0.5f);
        p.trail[p.trailHead] = p.pos;
        p.trailHead = (p.trailHead + 1) % TRAIL_LEN;
        if (p.trailCount < TRAIL_LEN) p.trailCount++;
    }
}

// logarithmic spiral arm galaxy
std::vector<Particle> makeGalaxy(Vec3 center, Vec3 drift,
                                  float coreMass, int numStars, int id) {
    std::vector<Particle> ps;
    std::mt19937 rng(id==0?42:id==1?99:7);
    std::uniform_real_distribution<float> uniD(0.0f, 1.0f);
    std::normal_distribution<float>       scatterD(0.0f, 0.05f);
    std::normal_distribution<float>       zD(0.0f, 0.015f);

    const int   NUM_ARMS = 2;
    const float SPIRAL_K = 3.2f;

    ps.push_back({center, drift, {}, coreMass, id});

    for (int i = 0; i < numStars; i++) {
        float u      = uniD(rng);
        float radius = 0.05f + (-std::log(1.0f - u * 0.98f)) * 0.10f;
        radius = std::min(radius, 0.55f);

        int   arm         = i % NUM_ARMS;
        float baseAngle   = arm * (2.0f * M_PI / NUM_ARMS);
        float spiralAngle = baseAngle + SPIRAL_K * std::log(radius + 0.1f) + scatterD(rng);

        Vec3 pos = {
            center.x + radius * std::cos(spiralAngle),
            center.y + radius * std::sin(spiralAngle),
            center.z + zD(rng)
        };
        float speed = std::sqrt(coreMass / radius);
        Vec3 vel = {
            drift.x - std::sin(spiralAngle) * speed,
            drift.y + std::cos(spiralAngle) * speed,
            drift.z
        };
        ps.push_back({pos, vel, {}, 0.001f, id});
    }
    return ps;
}

Particle makeBlackHole(Vec3 pos, Vec3 vel) {
    Particle bh;
    bh.pos    = pos;
    bh.vel    = vel;
    bh.acc    = {};
    bh.mass   = 50.0f;
    bh.galaxy = 3;
    return bh;
}
