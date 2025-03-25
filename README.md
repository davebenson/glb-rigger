# GLB Rigger

A tool to automatically generate a rigged skeleton from a GLB/GLTF model, perfect for when you need to add some backbone to your 3D models!

> "Why did the 3D character become a comedian? It had great rigging timing!" 🥁

## Overview

GLB Rigger analyzes a 3D mesh, creates a voxel representation, extracts a skeleton, and generates a rigged model ready for animation. No manual weight painting required - it's like having a digital orthopedist for your 3D models!

## Algorithm

The tool follows these steps to transform a static mesh into a rigged model:

### 1. Mesh Analysis
- Extract vertices and triangles from GLB/GLTF file
- Build an adjacency table for mesh topology analysis
- Find boundary edges, interior edges, and non-manifold edges

### 2. Voxelization
- Create a uniform 3D grid based on model dimensions
- Use a scanning plane approach moving along the z-axis
- Find triangle-plane intersections and voxelize the intersection lines
- Fill interior regions using 2D flood fill algorithm on each slice

### 3. Skeletonization
- Apply 3D thinning algorithm to create a one-voxel-width skeleton
- Preserve connectivity of the model during thinning
- Ensure the skeleton maintains the topological structure of the original model

### 4. Line Segment Approximation
- Convert skeleton voxels to world coordinates
- Approximate the skeleton with minimal line segments
- Keep error below specified threshold
- Optimize line segment endpoints to minimize error

### 5. Hierarchical Skeleton Construction
- Sort segments by length (longest first)
- Build a hierarchical joint structure starting from the longest segment
- Calculate joint positions and orientations
- Establish parent-child relationships between joints

### 6. Skinning Weight Generation
- For each vertex in the original mesh:
  - Find the two closest joints
  - Calculate weights using inverse distance weighting
  - Normalize weights to sum to 1.0

### 7. GLB Export
- Generate a GLB file with:
  - Original mesh geometry
  - Hierarchical skeleton joints
  - Skinning information (joints and weights)
  - All required buffer data and accessors

## Usage

```
cargo run -- <path_to_glb_file>
```

## Implementation Notes

The algorithm prioritizes automation over absolute precision. For complex models, manual refinement of the rig might be needed, but this provides an excellent starting point.

Remember, just like a bad skinning job can make your character's elbows look weird, a bad joke about rigging can make your audience cringe! But we're willing to take that risk.

## License

MIT

## Acknowledgements

Inspired by research in digital topology, skeletonization algorithms, and automatic rigging techniques.

This was written mostly by ClaudeAI with some help from Dave Benson.
