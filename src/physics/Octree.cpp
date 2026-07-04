#include "Octree.h"
#include <algorithm>
#include <cmath>

// ── helpers ───────────────────────────────────────────────────────

// returns the center of child octant `oct` of `node`
static Vec3 childCenter(const OctNode* node, int oct) {
    float q = node->halfSize * 0.5f;
    return {
        node->center.x + ((oct & 1) ? q : -q),
        node->center.y + ((oct & 2) ? q : -q),
        node->center.z + ((oct & 4) ? q : -q)
    };
}

// ── Octree ────────────────────────────────────────────────────────

Octree::Octree(const std::vector<Particle>& particles) {
    if (particles.empty()) return;

    // find axis-aligned bounding box of all particles
    Vec3 mn = particles[0].pos;
    Vec3 mx = particles[0].pos;
    for (const auto& p : particles) {
        mn.x = std::min(mn.x, p.pos.x);
        mn.y = std::min(mn.y, p.pos.y);
        mn.z = std::min(mn.z, p.pos.z);
        mx.x = std::max(mx.x, p.pos.x);
        mx.y = std::max(mx.y, p.pos.y);
        mx.z = std::max(mx.z, p.pos.z);
    }

    Vec3 center = {
        (mn.x + mx.x) * 0.5f,
        (mn.y + mx.y) * 0.5f,
        (mn.z + mx.z) * 0.5f
    };
    // use the largest axis so the root cube contains everything
    float halfSize = std::max({ mx.x-mn.x, mx.y-mn.y, mx.z-mn.z }) * 0.5f + 0.01f;

    root = std::make_unique<OctNode>(center, halfSize);

    for (int i = 0; i < (int)particles.size(); i++)
        insert(root.get(), particles, i);

    // bottom-up mass accumulation
    computeMass(root.get(), particles);
}

int Octree::octant(const OctNode* node, const Vec3& pos) const {
    return (pos.x > node->center.x ? 1 : 0)
         | (pos.y > node->center.y ? 2 : 0)
         | (pos.z > node->center.z ? 4 : 0);
}

void Octree::insert(OctNode* node, const std::vector<Particle>& particles, int idx) {
    if (node->isLeaf) {
        if (node->particleIdx == -1) {
            // empty leaf — just store the particle index here
            node->particleIdx = idx;
            return;
        }

        // occupied leaf — subdivide: push existing particle down then fall through
        node->isLeaf      = false;
        int existing      = node->particleIdx;
        node->particleIdx = -1;

        int oct = octant(node, particles[existing].pos);
        if (!node->children[oct])
            node->children[oct] = std::make_unique<OctNode>(
                childCenter(node, oct), node->halfSize * 0.5f);
        insert(node->children[oct].get(), particles, existing);
    }

    // insert new particle into the correct child octant
    int oct = octant(node, particles[idx].pos);
    if (!node->children[oct])
        node->children[oct] = std::make_unique<OctNode>(
            childCenter(node, oct), node->halfSize * 0.5f);
    insert(node->children[oct].get(), particles, idx);
}

void Octree::computeMass(OctNode* node, const std::vector<Particle>& particles) {
    if (node->isLeaf) {
        if (node->particleIdx >= 0) {
            node->totalMass = particles[node->particleIdx].mass;
            node->com       = particles[node->particleIdx].pos;
        }
        return;
    }

    node->totalMass = 0.0f;
    node->com       = {};
    for (auto& child : node->children) {
        if (!child) continue;
        computeMass(child.get(), particles);
        // weighted sum for center of mass
        node->com       += child->com * child->totalMass;
        node->totalMass += child->totalMass;
    }
    if (node->totalMass > 0.0f)
        node->com = node->com * (1.0f / node->totalMass);
}

Vec3 Octree::traverse(const OctNode* node, const Particle& p, int skipIdx) const {
    if (!node || node->totalMass == 0.0f) return {};

    if (node->isLeaf) {
        // skip empty leaves and self
        if (node->particleIdx < 0 || node->particleIdx == skipIdx) return {};

        Vec3  d     = node->com - p.pos;
        float dist2 = d.norm2() + SOFTENING * SOFTENING;
        float dist  = std::sqrt(dist2);
        return d * (G * node->totalMass / (dist2 * dist));
    }

    // Barnes-Hut criterion: s/d < theta
    // s = side length of this node = halfSize * 2
    // d = distance from particle to node's center of mass
    Vec3  d    = node->com - p.pos;
    float dist = std::sqrt(d.norm2() + SOFTENING * SOFTENING);
    float s    = node->halfSize * 2.0f;

    if (s / dist < THETA) {
        // far enough — treat entire subtree as a single point mass
        float dist2 = d.norm2() + SOFTENING * SOFTENING;
        return d * (G * node->totalMass / (dist2 * dist));
    }

    // too close — recurse into children
    Vec3 acc{};
    for (const auto& child : node->children) {
        if (child) acc += traverse(child.get(), p, skipIdx);
    }
    return acc;
}

Vec3 Octree::computeAcc(const Particle& p, int skipIdx) const {
    if (!root) return {};
    return traverse(root.get(), p, skipIdx);
}
