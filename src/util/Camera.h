#pragma once
#include <GLFW/glfw3.h>
#include <algorithm>

// ── camera state ──────────────────────────────────────────────────
inline float  camAzimuth   = 30.0f;
inline float  camElevation = 25.0f;
inline float  camDist      = 5.0f;
inline bool   leftDrag     = false;
inline double prevMX       = 0.0;
inline double prevMY       = 0.0;

// ── sim state (shared with main + Simulation) ─────────────────────
inline bool  paused    = false;
inline float timeScale = 1.0f;

// ── callbacks ─────────────────────────────────────────────────────
inline void keyCallback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_SPACE) paused = !paused;
    if (key == GLFW_KEY_EQUAL) timeScale = std::min(timeScale * 2.0f,  16.0f);
    if (key == GLFW_KEY_MINUS) timeScale = std::max(timeScale * 0.5f, 0.0625f);
    if (key == GLFW_KEY_R)   { camAzimuth=30; camElevation=25; camDist=5; }
}

inline void scrollCallback(GLFWwindow*, double, double yoff) {
    camDist *= (yoff > 0) ? 0.9f : 1.1f;
    camDist  = std::max(0.5f, std::min(camDist, 20.0f));
}

inline void mouseButtonCallback(GLFWwindow* win, int btn, int action, int) {
    if (btn != GLFW_MOUSE_BUTTON_LEFT) return;
    leftDrag = (action == GLFW_PRESS);
    glfwGetCursorPos(win, &prevMX, &prevMY);
}

inline void cursorCallback(GLFWwindow*, double x, double y) {
    if (!leftDrag) return;
    camAzimuth   += (float)(x - prevMX) * 0.4f;
    camElevation -= (float)(y - prevMY) * 0.4f;
    camElevation  = std::max(-89.0f, std::min(89.0f, camElevation));
    prevMX = x; prevMY = y;
}
