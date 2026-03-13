#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// ── constants ─────────────────────────────────────────────────────
const int   WIDTH     = 1280;
const int   HEIGHT    = 720;
const float G         = 6.0623f;
const float SOFTENING = 0.15f;
const float DT_BASE   = 0.0005f;
const int   TRAIL_LEN = 40;

// ── sim state ─────────────────────────────────────────────────────
bool  paused    = false;
float timeScale = 1.0f;

// ── camera ────────────────────────────────────────────────────────
float  camAzimuth   =  30.0f;
float  camElevation =  25.0f;
float  camDist      =   5.0f;
bool   leftDrag     = false;
double prevMX, prevMY;

// ── Vec3 ──────────────────────────────────────────────────────────
struct Vec3 {
    float x,y,z;
    Vec3(float x=0,float y=0,float z=0):x(x),y(y),z(z){}
    Vec3 operator+(const Vec3& o)const{return{x+o.x,y+o.y,z+o.z};}
    Vec3 operator-(const Vec3& o)const{return{x-o.x,y-o.y,z-o.z};}
    Vec3 operator*(float s)      const{return{x*s,  y*s,  z*s  };}
    Vec3& operator+=(const Vec3& o){x+=o.x;y+=o.y;z+=o.z;return*this;}
    float norm2()const{return x*x+y*y+z*z;}
    float norm() const{return std::sqrt(norm2());}
};

// ── Particle ──────────────────────────────────────────────────────
struct Particle {
    Vec3  pos,vel,acc;
    float mass;
    int   galaxy;
    Vec3  trail[TRAIL_LEN];
    int   trailHead=0, trailCount=0;
};

// ── shaders ───────────────────────────────────────────────────────
const char* starVert = R"(
#version 330 core
layout(location=0) in vec3  aPos;
layout(location=1) in float aSpeed;
uniform mat4 MVP;
out float vSpeed;
void main(){
    gl_Position  = MVP * vec4(aPos, 1.0);
    gl_PointSize = clamp(8.0 / gl_Position.w, 1.5, 14.0);
    vSpeed = aSpeed;
}
)";

const char* starFrag = R"(
#version 330 core
in float vSpeed;
out vec4 FragColor;
void main(){
    vec2  c = gl_PointCoord - 0.5;
    float r = length(c);
    if (r > 0.5) discard;
    float alpha = 1.0 - smoothstep(0.05, 0.5, r);
    float t = clamp(vSpeed / 12.0, 0.0, 1.0);
    vec3  col;
    if (t < 0.5) col = mix(vec3(0.3,0.5,1.0), vec3(1.0,1.0,1.0), t*2.0);
    else         col = mix(vec3(1.0,1.0,1.0), vec3(1.0,0.45,0.1), (t-0.5)*2.0);
    FragColor = vec4(col, alpha);
}
)";

const char* trailVert = R"(
#version 330 core
layout(location=0) in vec3  aPos;
layout(location=1) in float aAlpha;
layout(location=2) in float aSpeed;
uniform mat4 MVP;
out float vAlpha;
out float vSpeed;
void main(){
    gl_Position = MVP * vec4(aPos, 1.0);
    vAlpha = aAlpha;
    vSpeed = aSpeed;
}
)";

const char* trailFrag = R"(
#version 330 core
in float vAlpha;
in float vSpeed;
out vec4 FragColor;
void main(){
    float t = clamp(vSpeed / 12.0, 0.0, 1.0);
    vec3  col;
    if (t < 0.5) col = mix(vec3(0.3,0.5,1.0), vec3(1.0,1.0,1.0), t*2.0);
    else         col = mix(vec3(1.0,1.0,1.0), vec3(1.0,0.45,0.1), (t-0.5)*2.0);
    FragColor = vec4(col, vAlpha * 0.5);
}
)";

// ── shader compile ────────────────────────────────────────────────
GLuint makeProgram(const char* vsrc, const char* fsrc) {
    auto compile = [](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512]; glGetShaderInfoLog(s,512,nullptr,log);
            std::cerr << "Shader error: " << log << "\n";
        }
        return s;
    };
    GLuint vs = compile(GL_VERTEX_SHADER,   vsrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsrc);
    GLuint p  = glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs);
    glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// ── physics ───────────────────────────────────────────────────────
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

// ── galaxy setup ──────────────────────────────────────────────────
std::vector<Particle> makeGalaxy(Vec3 center, Vec3 drift,
                                  float coreMass, int numStars, int id) {
    std::vector<Particle> ps;
    std::mt19937 rng(id==0?42:id==1?99:7);
    std::uniform_real_distribution<float> angD(0, 2*M_PI);
    std::uniform_real_distribution<float> radD(0.05f, 0.4f);
    std::normal_distribution<float>       zD(0, 0.02f);  // thin disc

    ps.push_back({center, drift, {}, coreMass, id});

    for (int i = 0; i < numStars; i++) {
        float angle  = angD(rng);
        float radius = radD(rng);
        Vec3  pos    = { center.x + radius*std::cos(angle),
                         center.y + radius*std::sin(angle),
                         center.z + zD(rng) };
        float speed  = std::sqrt(coreMass / radius);
        Vec3  vel    = { drift.x - std::sin(angle)*speed,
                         drift.y + std::cos(angle)*speed,
                         drift.z };
        ps.push_back({pos, vel, {}, 0.001f, id});
    }
    return ps;
}

// ── callbacks ─────────────────────────────────────────────────────
void keyCallback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_SPACE) paused = !paused;
    if (key == GLFW_KEY_EQUAL) timeScale = std::min(timeScale*2.0f,  16.0f);
    if (key == GLFW_KEY_MINUS) timeScale = std::max(timeScale*0.5f, 0.0625f);
    if (key == GLFW_KEY_R)   { camAzimuth=30; camElevation=25; camDist=5; }
}

void scrollCallback(GLFWwindow*, double, double yoff) {
    camDist *= (yoff > 0) ? 0.9f : 1.1f;
    camDist  = std::max(0.5f, std::min(camDist, 20.0f));
}

void mouseButtonCallback(GLFWwindow* win, int btn, int action, int) {
    if (btn != GLFW_MOUSE_BUTTON_LEFT) return;
    leftDrag = (action == GLFW_PRESS);
    glfwGetCursorPos(win, &prevMX, &prevMY);
}

void cursorCallback(GLFWwindow*, double x, double y) {
    if (!leftDrag) return;
    camAzimuth   += (float)(x - prevMX) * 0.4f;
    camElevation -= (float)(y - prevMY) * 0.4f;
    camElevation  = std::max(-89.0f, std::min(89.0f, camElevation));
    prevMX = x; prevMY = y;
}

// ── main ──────────────────────────────────────────────────────────
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
    glGetError(); // clear the spurious error GLEW generates on core profile

    glfwSetKeyCallback(win,         keyCallback);
    glfwSetScrollCallback(win,      scrollCallback);
    glfwSetMouseButtonCallback(win, mouseButtonCallback);
    glfwSetCursorPosCallback(win,   cursorCallback);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);       // additive — stars glow and stack
    glEnable(GL_PROGRAM_POINT_SIZE);

    GLuint starProg  = makeProgram(starVert,  starFrag);
    GLuint trailProg = makeProgram(trailVert, trailFrag);

    // star VBO layout: [x, y, z, speed]
    GLuint starVAO, starVBO;
    glGenVertexArrays(1, &starVAO); glGenBuffers(1, &starVBO);
    glBindVertexArray(starVAO);
    glBindBuffer(GL_ARRAY_BUFFER, starVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // trail VBO layout: [x, y, z, alpha, speed]
    GLuint trailVAO, trailVBO;
    glGenVertexArrays(1, &trailVAO); glGenBuffers(1, &trailVBO);
    glBindVertexArray(trailVAO);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(4*sizeof(float)));
    glEnableVertexAttribArray(2);

    // g3 placed off-plane so 3D rotation actually shows depth
    auto g1 = makeGalaxy({-0.8f,  0.1f,  0.0f}, { 0.08f,  0.02f,  0.00f}, 2.0f, 150, 0);
    auto g2 = makeGalaxy({ 0.8f, -0.1f,  0.0f}, {-0.08f, -0.02f,  0.00f}, 2.0f, 150, 1);
    auto g3 = makeGalaxy({ 0.0f,  0.2f,  0.6f}, { 0.02f, -0.05f, -0.09f}, 3.0f, 200, 2);

    std::vector<Particle> particles;
    particles.insert(particles.end(), g1.begin(), g1.end());
    particles.insert(particles.end(), g2.begin(), g2.end());
    particles.insert(particles.end(), g3.begin(), g3.end());
    computeForces(particles);

    std::vector<float> starData, trailData;

    while (!glfwWindowShouldClose(win)) {
        if (!paused) integrate(particles);

        glClearColor(0.02f, 0.02f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // build MVP from orbital camera
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

        // upload + draw stars
        starData.clear();
        for (const auto& p : particles) {
            starData.push_back(p.pos.x); starData.push_back(p.pos.y); starData.push_back(p.pos.z);
            starData.push_back(p.vel.norm());
        }
        glBindVertexArray(starVAO);
        glBindBuffer(GL_ARRAY_BUFFER, starVBO);
        glBufferData(GL_ARRAY_BUFFER, starData.size()*sizeof(float), starData.data(), GL_DYNAMIC_DRAW);
        glUseProgram(starProg);
        glUniformMatrix4fv(glGetUniformLocation(starProg,"MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
        glDrawArrays(GL_POINTS, 0, (GLsizei)particles.size());

        // upload + draw trails
        trailData.clear();
        for (const auto& p : particles) {
            if (p.trailCount < 2) continue;
            float speed = p.vel.norm();
            int start = ((p.trailHead - p.trailCount) % TRAIL_LEN + TRAIL_LEN) % TRAIL_LEN;
            for (int k = 0; k < p.trailCount - 1; k++) {
                int   ia = (start+k)   % TRAIL_LEN;
                int   ib = (start+k+1) % TRAIL_LEN;
                float a0 = (float)k     / p.trailCount;
                float a1 = (float)(k+1) / p.trailCount;
                trailData.insert(trailData.end(),
                    {p.trail[ia].x, p.trail[ia].y, p.trail[ia].z, a0, speed,
                     p.trail[ib].x, p.trail[ib].y, p.trail[ib].z, a1, speed});
            }
        }
        if (!trailData.empty()) {
            glBindVertexArray(trailVAO);
            glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
            glBufferData(GL_ARRAY_BUFFER, trailData.size()*sizeof(float), trailData.data(), GL_DYNAMIC_DRAW);
            glUseProgram(trailProg);
            glUniformMatrix4fv(glGetUniformLocation(trailProg,"MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
            glDrawArrays(GL_LINES, 0, (GLsizei)(trailData.size()/5));
        }

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1,&starVAO); glDeleteBuffers(1,&starVBO);
    glDeleteVertexArrays(1,&trailVAO); glDeleteBuffers(1,&trailVBO);
    glDeleteProgram(starProg); glDeleteProgram(trailProg);
    glfwTerminate();
    return 0;
}
