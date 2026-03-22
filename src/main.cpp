#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "physics/Particle.h"
#include "physics/Simulation.h"
#include "render/Renderer.h"
#include "util/Camera.h"

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(WIDTH, HEIGHT, "N-Body 3D", nullptr, nullptr);
    if (!win) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    glewInit();
    glGetError();

    glfwSetKeyCallback(win,           keyCallback);
    glfwSetScrollCallback(win,        scrollCallback);
    glfwSetMouseButtonCallback(win,   mouseButtonCallback);
    glfwSetCursorPosCallback(win,     cursorCallback);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // build galaxies
    auto g1 = makeGalaxy({-0.8f,  0.1f,  0.0f}, { 0.08f,  0.02f,  0.00f}, 2.0f, 200, 0);
    auto g2 = makeGalaxy({ 0.8f, -0.1f,  0.0f}, {-0.08f, -0.02f,  0.00f}, 2.0f, 200, 1);
    auto g3 = makeGalaxy({ 0.0f,  0.2f,  0.6f}, { 0.02f, -0.05f, -0.09f}, 3.0f, 250, 2);

    std::vector<Particle> particles;
    particles.insert(particles.end(), g1.begin(), g1.end());
    particles.insert(particles.end(), g2.begin(), g2.end());
    particles.insert(particles.end(), g3.begin(), g3.end());
    particles.push_back(makeBlackHole(
    {0.0f, 2.0f, 0.5f},    // starts above the scene
    {0.0f, -0.15f, 0.0f}   // slowly falls in
));
    computeForces(particles);

    Renderer renderer;

    double lastFPSTime = glfwGetTime();
    int    frameCount  = 0;

    while (!glfwWindowShouldClose(win)) {
        if (!paused) integrate(particles);

        // FPS counter
        frameCount++;
        double now = glfwGetTime();
        if (now - lastFPSTime >= 1.0) {
            int fps = (int)(frameCount / (now - lastFPSTime));
            std::string title = "N-Body 3D  |  "
                + std::to_string(particles.size()) + " particles  |  "
                + std::to_string(fps) + " FPS"
                + (paused ? "  |  PAUSED" : "  |  " + std::to_string(timeScale) + "x");
            glfwSetWindowTitle(win, title.c_str());
            lastFPSTime = now;
            frameCount  = 0;
        }

        // build MVP
        float az = glm::radians(camAzimuth);
        float el = glm::radians(camElevation);
        glm::vec3 eye(
            camDist * std::cos(el) * std::sin(az),
            camDist * std::sin(el),
            camDist * std::cos(el) * std::cos(az)
        );
        glm::mat4 view = glm::lookAt(eye, glm::vec3(0,0,0), glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), (float)WIDTH/HEIGHT, 0.01f, 100.0f);
        glm::mat4 MVP  = proj * view;

        renderer.draw(particles, MVP);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
