#pragma once
#include "Particle.h"
#include <vector>
#include <memory>
#include <array>

// Barnes-Hut opening angle — lower = more accurate, slower
// 0.5 is the standard research value
const float THETA = 0.5f;

struct OctNode {
    Vec3  center;       // center of this octant's bounding cube
    float halfSize;     // half-width of the cube

    float totalMass = 0.0f;
    Vec3  com       = {};   // center of mass of all particles in subtree

    int  particleIdx = -1;  // >=0 only for a leaf holding exactly one particle
    bool isLeaf      = true;

    std::array<std::unique_ptr<OctNode>, 8> children;

    OctNode(Vec3 c, float hs) : center(c), halfSize(hs) {}
};

class Octree {
public:
    // builds the full tree from the particle list in O(N log N)
    explicit Octree(const std::vector<Particle>& particles);

    // returns gravitational acceleration on particle p, skipping self
    Vec3 computeAcc(const Particle& p, int skipIdx) const;

private:
    std::unique_ptr<OctNode> root;

    // which of the 8 octants does pos fall into?
    // encoded as 3 bits: bit0=x, bit1=y, bit2=z
    int  octant    (const OctNode* node, const Vec3& pos) const;

    void insert     (OctNode* node, const std::vector<Particle>& particles, int idx);
    void computeMass(OctNode* node, const std::vector<Particle>& particles);
    Vec3 traverse   (const OctNode* node, const Particle& p, int skipIdx) const;
};
