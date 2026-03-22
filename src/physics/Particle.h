#pragma once
#include <cmath>

const float G         = 6.0623f;
const float SOFTENING = 0.15f;
const float DT_BASE   = 0.0005f;
const int   TRAIL_LEN = 40;

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

struct Particle {
    Vec3  pos,vel,acc;
    float mass;
    int   galaxy;
    Vec3  trail[TRAIL_LEN];
    int   trailHead=0, trailCount=0;
};
