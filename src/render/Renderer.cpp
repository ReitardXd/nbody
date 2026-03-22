#include "Renderer.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

// ── shader sources ────────────────────────────────────────────────
static const char* starVert = R"(
#version 330 core
layout(location=0) in vec3  aPos;
layout(location=1) in float aSpeed;
uniform mat4 MVP;
out float vSpeed;
void main(){
    gl_Position  = MVP * vec4(aPos, 1.0);
    gl_PointSize = vSpeed > 100.0 ? clamp(30.0 / gl_Position.w, 8.0, 40.0)
                               : clamp(12.0  / gl_Position.w, 2.0, 18.0);
    vSpeed = aSpeed;
}
)";

static const char* starFrag = R"(
#version 330 core
in float vSpeed;
out vec4 FragColor;
void main(){
    vec2  c = gl_PointCoord - 0.5;
    float r = length(c);
    if(r > 0.5) discard;
    float alpha = 1.0 - smoothstep(0.0, 0.5, r);
    float t = clamp(vSpeed / 12.0, 0.0, 1.0);
    vec3 col;
    if(t < 0.5) col = mix(vec3(0.3,0.5,1.0), vec3(1.0,1.0,1.0), t*2.0);
    else        col = mix(vec3(1.0,1.0,1.0), vec3(1.0,0.45,0.1), (t-0.5)*2.0);
    FragColor = vec4(col * 1.6, alpha);
}
)";

static const char* trailVert = R"(
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

static const char* trailFrag = R"(
#version 330 core
in float vAlpha;
in float vSpeed;
out vec4 FragColor;
void main(){
    float t = clamp(vSpeed / 12.0, 0.0, 1.0);
    vec3 col;
    if(t < 0.5) col = mix(vec3(0.3,0.5,1.0), vec3(1.0,1.0,1.0), t*2.0);
    else        col = mix(vec3(1.0,1.0,1.0), vec3(1.0,0.45,0.1), (t-0.5)*2.0);
    FragColor = vec4(col, vAlpha * 0.4);
}
)";

static const char* screenVert = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); vUV = aUV; }
)";

static const char* brightFrag = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D scene;
void main(){
    vec3  col        = texture(scene, vUV).rgb;
    float brightness = dot(col, vec3(0.2126, 0.7152, 0.0722));
    FragColor = vec4(brightness > 0.2 ? col : vec3(0.0), 1.0);
}
)";

static const char* blurFrag = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D image;
uniform bool horizontal;
const float weight[5] = float[](0.227027,0.1945946,0.1216216,0.054054,0.016216);
void main(){
    vec2 texel  = 1.0 / textureSize(image, 0);
    vec3 result = texture(image, vUV).rgb * weight[0];
    for(int i = 1; i < 5; i++){
        vec2 off = horizontal ? vec2(texel.x*i, 0.0) : vec2(0.0, texel.y*i);
        result  += texture(image, vUV + off).rgb * weight[i];
        result  += texture(image, vUV - off).rgb * weight[i];
    }
    FragColor = vec4(result, 1.0);
}
)";

static const char* bloomFrag = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D scene;
uniform sampler2D bloomBlur;
uniform float     bloomStrength;
void main(){
    vec3 col   = texture(scene,    vUV).rgb;
    vec3 bloom = texture(bloomBlur,vUV).rgb;
    col  += bloom * bloomStrength;
    col   = col / (col + vec3(1.0));
    col   = pow(col, vec3(1.0 / 2.2));
    FragColor = vec4(col, 1.0);
}
)";

// ── implementation ────────────────────────────────────────────────
GLuint Renderer::makeProgram(const char* vsrc, const char* fsrc) {
    auto compile = [](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
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

FBO Renderer::makeFBO(int w, int h) {
    FBO f;
    glGenFramebuffers(1, &f.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, f.fbo);
    glGenTextures(1, &f.tex);
    glBindTexture(GL_TEXTURE_2D, f.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, f.tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return f;
}

Renderer::Renderer() {
    starProg   = makeProgram(starVert,   starFrag);
    trailProg  = makeProgram(trailVert,  trailFrag);
    brightProg = makeProgram(screenVert, brightFrag);
    blurProg   = makeProgram(screenVert, blurFrag);
    bloomProg  = makeProgram(screenVert, bloomFrag);

    sceneFBO = makeFBO(WIDTH, HEIGHT);
    pingFBO  = makeFBO(WIDTH, HEIGHT);
    pongFBO  = makeFBO(WIDTH, HEIGHT);

    // fullscreen quad
    float quadVerts[] = {
        -1,-1, 0,0,   1,-1, 1,0,   1, 1, 1,1,
        -1,-1, 0,0,   1, 1, 1,1,  -1, 1, 0,1
    };
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // star VAO [x,y,z, speed]
    glGenVertexArrays(1, &starVAO); glGenBuffers(1, &starVBO);
    glBindVertexArray(starVAO);
    glBindBuffer(GL_ARRAY_BUFFER, starVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // trail VAO [x,y,z, alpha, speed]
    glGenVertexArrays(1, &trailVAO); glGenBuffers(1, &trailVBO);
    glBindVertexArray(trailVAO);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(4*sizeof(float)));
    glEnableVertexAttribArray(2);
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1,&starVAO);  glDeleteBuffers(1,&starVBO);
    glDeleteVertexArrays(1,&trailVAO); glDeleteBuffers(1,&trailVBO);
    glDeleteVertexArrays(1,&quadVAO);  glDeleteBuffers(1,&quadVBO);
    glDeleteFramebuffers(1,&sceneFBO.fbo); glDeleteTextures(1,&sceneFBO.tex);
    glDeleteFramebuffers(1,&pingFBO.fbo);  glDeleteTextures(1,&pingFBO.tex);
    glDeleteFramebuffers(1,&pongFBO.fbo);  glDeleteTextures(1,&pongFBO.tex);
    glDeleteProgram(starProg);  glDeleteProgram(trailProg);
    glDeleteProgram(brightProg); glDeleteProgram(blurProg); glDeleteProgram(bloomProg);
}

void Renderer::drawTrails(const std::vector<Particle>& particles, const glm::mat4& MVP) {
    trailData.clear();
    for (const auto& p : particles) {
        if (p.trailCount < 2) continue;
        float speed = p.vel.norm();
        int   start = ((p.trailHead - p.trailCount) % TRAIL_LEN + TRAIL_LEN) % TRAIL_LEN;
        for (int k = 0; k < p.trailCount - 1; k++) {
            int   ia = (start+k)   % TRAIL_LEN;
            int   ib = (start+k+1) % TRAIL_LEN;
            float a0 = (float)k     / p.trailCount;
            float a1 = (float)(k+1) / p.trailCount;
            trailData.insert(trailData.end(), {
                p.trail[ia].x, p.trail[ia].y, p.trail[ia].z, a0, speed,
                p.trail[ib].x, p.trail[ib].y, p.trail[ib].z, a1, speed
            });
        }
    }
    if (trailData.empty()) return;
    glBindVertexArray(trailVAO);
    glBindBuffer(GL_ARRAY_BUFFER, trailVBO);
    glBufferData(GL_ARRAY_BUFFER, trailData.size()*sizeof(float), trailData.data(), GL_DYNAMIC_DRAW);
    glUseProgram(trailProg);
    glUniformMatrix4fv(glGetUniformLocation(trailProg,"MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glDrawArrays(GL_LINES, 0, (GLsizei)(trailData.size()/5));
}

void Renderer::drawStars(const std::vector<Particle>& particles, const glm::mat4& MVP) {
    starData.clear();
    for (const auto& p : particles) {
        starData.push_back(p.pos.x); starData.push_back(p.pos.y); starData.push_back(p.pos.z);
        // black holes render with a special speed value that makes them large + white-hot
starData.push_back(p.galaxy == 3 ? 999.0f : p.vel.norm()); 
    }
    glBindVertexArray(starVAO);
    glBindBuffer(GL_ARRAY_BUFFER, starVBO);
    glBufferData(GL_ARRAY_BUFFER, starData.size()*sizeof(float), starData.data(), GL_DYNAMIC_DRAW);
    glUseProgram(starProg);
    glUniformMatrix4fv(glGetUniformLocation(starProg,"MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glDrawArrays(GL_POINTS, 0, (GLsizei)(starData.size()/4));
}

void Renderer::bloomPass() {
    glDisable(GL_BLEND);
    glBindVertexArray(quadVAO);

    // extract bright regions
    glBindFramebuffer(GL_FRAMEBUFFER, pingFBO.fbo);
    glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(brightProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneFBO.tex);
    glUniform1i(glGetUniformLocation(brightProg,"scene"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // ping-pong blur
    glUseProgram(blurProg);
    GLint horizLoc = glGetUniformLocation(blurProg,"horizontal");
    GLint imgLoc   = glGetUniformLocation(blurProg,"image");
    for (int i = 0; i < BLUR_PASSES; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, pongFBO.fbo);
        glUniform1i(horizLoc, 1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, pingFBO.tex);
        glUniform1i(imgLoc, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindFramebuffer(GL_FRAMEBUFFER, pingFBO.fbo);
        glUniform1i(horizLoc, 0);
        glBindTexture(GL_TEXTURE_2D, pongFBO.tex);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    // composite to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(bloomProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sceneFBO.tex);
    glUniform1i(glGetUniformLocation(bloomProg,"scene"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pingFBO.tex);
    glUniform1i(glGetUniformLocation(bloomProg,"bloomBlur"), 1);
    glUniform1f(glGetUniformLocation(bloomProg,"bloomStrength"), 3.0f);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Renderer::draw(const std::vector<Particle>& particles, const glm::mat4& MVP) {
    // pass 1: scene → HDR FBO
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO.fbo);
    glClearColor(0.02f, 0.02f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    drawTrails(particles, MVP);
    drawStars (particles, MVP);

    // passes 2-4: bloom pipeline
    bloomPass();
}
