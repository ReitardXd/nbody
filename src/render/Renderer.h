#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "../physics/Particle.h"
#include <vector>

const int   WIDTH       = 1920;
const int   HEIGHT      = 1080;
const int   BLUR_PASSES = 10;

struct FBO { GLuint fbo, tex; };

class Renderer {
public:
    Renderer();
    ~Renderer();

    void draw(const std::vector<Particle>& particles, const glm::mat4& MVP);

private:
    // shader programs
    GLuint starProg, trailProg, brightProg, blurProg, bloomProg;

    // VAOs + VBOs
    GLuint starVAO,  starVBO;
    GLuint trailVAO, trailVBO;
    GLuint quadVAO,  quadVBO;

    // bloom FBOs
    FBO sceneFBO, pingFBO, pongFBO;

    // cpu-side upload buffers
    std::vector<float> starData, trailData;

    FBO    makeFBO(int w, int h);
    GLuint makeProgram(const char* vsrc, const char* fsrc);
    void   drawTrails(const std::vector<Particle>& particles, const glm::mat4& MVP);
    void   drawStars (const std::vector<Particle>& particles, const glm::mat4& MVP);
    void   bloomPass ();
};
