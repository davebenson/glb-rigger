#[macro_use]
extern crate serde_json;

//use gltf_kun::graph::{Graph, GraphNodeWeight, gltf::document::GltfDocument};
use gltf;
use std::{fs, io, collections::{HashMap, HashSet}};
use std::path::Path;
use std::boxed::Box;
use std::error::Error as StdError;
use std::cmp::{min, max};

fn run(path: &str) -> Result<(), Box<dyn StdError>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let gltf = gltf::Gltf::from_reader(reader)?;
    
    // Get the binary data for buffer access
    let buffer_data = load_buffers(&gltf, path)?;
    
    // Find the first mesh
    for mesh in gltf.meshes() {
        println!("Found mesh: {}", mesh.name().unwrap_or("unnamed"));
        
        // Process the primitive
        for primitive in mesh.primitives() {
            let mode = primitive.mode();
            println!("Primitive mode: {:?}", mode);
            
            // Check if the mode is triangles
            if mode == gltf::mesh::Mode::Triangles {
                println!("Triangle mode detected, extracting vertices...");
                
                // Store vertices and triangle indices
                let mut vertices = Vec::new();
                let mut indices = Vec::new();
                
                // Get the vertex position accessor
                if let Some(positions) = primitive.get(&gltf::Semantic::Positions) {
                    let accessor = positions;
                    let view = accessor.view().unwrap();
                    let buffer = view.buffer();
                    
                    let buffer_index = buffer.index();
                    let offset = view.offset() + accessor.offset();
                    let count = accessor.count();
                    let stride = accessor.size();
                    
                    println!("Vertex count: {}", count);
                    
                    // Extract vertices
                    let buffer_slice = &buffer_data[buffer_index];
                    for i in 0..count {
                        let pos_offset = offset + i * stride;
                        let x = read_f32(&buffer_slice[pos_offset..pos_offset + 4]);
                        let y = read_f32(&buffer_slice[pos_offset + 4..pos_offset + 8]);
                        let z = read_f32(&buffer_slice[pos_offset + 8..pos_offset + 12]);
                        
                        vertices.push([x, y, z]);
                        if i < 10 { // Print just a few vertices as samples
                            println!("Vertex {}: ({}, {}, {})", i, x, y, z);
                        }
                    }
                }
                
                // Get the indices accessor
                if let Some(indices_accessor) = primitive.indices() {
                    let view = indices_accessor.view().unwrap();
                    let buffer = view.buffer();
                    
                    let buffer_index = buffer.index();
                    let offset = view.offset() + indices_accessor.offset();
                    let count = indices_accessor.count();
                    let component_type = indices_accessor.data_type();
                    
                    println!("Index count: {} (triangle count: {})", count, count / 3);
                    
                    // Extract indices based on component type
                    let buffer_slice = &buffer_data[buffer_index];
                    
                    // Read indices based on their data type
                    for i in 0..count {
                        let index = match component_type {
                            gltf::accessor::DataType::U8 => {
                                buffer_slice[offset + i] as u32
                            },
                            gltf::accessor::DataType::U16 => {
                                let bytes = [
                                    buffer_slice[offset + i * 2],
                                    buffer_slice[offset + i * 2 + 1],
                                ];
                                u16::from_le_bytes(bytes) as u32
                            },
                            gltf::accessor::DataType::U32 => {
                                let bytes = [
                                    buffer_slice[offset + i * 4],
                                    buffer_slice[offset + i * 4 + 1],
                                    buffer_slice[offset + i * 4 + 2],
                                    buffer_slice[offset + i * 4 + 3],
                                ];
                                u32::from_le_bytes(bytes)
                            },
                            _ => panic!("Unsupported index component type"),
                        };
                        
                        indices.push(index);
                    }
                    
                    // Print some triangle samples
                    for i in 0..std::cmp::min(3, count / 3) {
                        let a = indices[i * 3] as usize;
                        let b = indices[i * 3 + 1] as usize;
                        let c = indices[i * 3 + 2] as usize;
                        println!("Triangle {}: ({}, {}, {})", i, a, b, c);
                    }
                    
                    // Build adjacency table: Map edge (vertex pair) to triangle indices
                    let mut edge_to_triangles: std::collections::HashMap<(u32, u32), Vec<usize>> = std::collections::HashMap::new();
                    
                    for triangle_idx in 0..(count / 3) {
                        let a = indices[triangle_idx * 3];
                        let b = indices[triangle_idx * 3 + 1];
                        let c = indices[triangle_idx * 3 + 2];
                        
                        // For each edge of the triangle, store the triangle index
                        // We'll normalize the edge by ensuring the smaller vertex index comes first
                        add_edge_to_map(&mut edge_to_triangles, a, b, triangle_idx);
                        add_edge_to_map(&mut edge_to_triangles, b, c, triangle_idx);
                        add_edge_to_map(&mut edge_to_triangles, c, a, triangle_idx);
                    }
                    
                    // Print adjacency statistics
                    println!("Built adjacency table with {} unique edges", edge_to_triangles.len());
                    
                    // Find the number of boundary edges (edges with only one adjacent triangle)
                    let boundary_edges = edge_to_triangles.values()
                        .filter(|triangles| triangles.len() == 1)
                        .count();
                    println!("Boundary edges: {}", boundary_edges);
                    
                    // Find the number of interior edges (edges with two adjacent triangles)
                    let interior_edges = edge_to_triangles.values()
                        .filter(|triangles| triangles.len() == 2)
                        .count();
                    println!("Interior edges: {}", interior_edges);
                    
                    // Find any non-manifold edges (edges with more than two adjacent triangles)
                    let non_manifold_edges = edge_to_triangles.values()
                        .filter(|triangles| triangles.len() > 2)
                        .count();
                    println!("Non-manifold edges: {}", non_manifold_edges);
                    
                    // Print a few examples of adjacent triangles
                    let mut example_count = 0;
                    for ((v1, v2), triangles) in edge_to_triangles.iter() {
                        if triangles.len() == 2 && example_count < 5 {
                            println!("Edge ({}, {}): Adjacent triangles {:?}", v1, v2, triangles);
                            example_count += 1;
                        }
                    }
                    
                    // Create a voxel representation of the mesh using a scanning grid approach
                    println!("\nCreating voxel representation using scanning grid...");
                    
                    // First, determine the bounding box of the mesh
                    let mut min_x = f32::MAX;
                    let mut min_y = f32::MAX;
                    let mut min_z = f32::MAX;
                    let mut max_x = f32::MIN;
                    let mut max_y = f32::MIN;
                    let mut max_z = f32::MIN;
                    
                    for [x, y, z] in &vertices {
                        min_x = min_x.min(*x);
                        min_y = min_y.min(*y);
                        min_z = min_z.min(*z);
                        max_x = max_x.max(*x);
                        max_y = max_y.max(*y);
                        max_z = max_z.max(*z);
                    }
                    
                    println!("Model bounding box:");
                    println!("  Min: ({:.3}, {:.3}, {:.3})", min_x, min_y, min_z);
                    println!("  Max: ({:.3}, {:.3}, {:.3})", max_x, max_y, max_z);
                    
                    // Define voxel grid resolution (adjust as needed)
                    let resolution = 64; // Number of voxels along the longest dimension
                    
                    // Calculate the size of the bounding box
                    let size_x = max_x - min_x;
                    let size_y = max_y - min_y;
                    let size_z = max_z - min_z;
                    let max_size = size_x.max(size_y).max(size_z);
                    
                    // Calculate voxel size
                    let voxel_size = max_size / resolution as f32;
                    
                    // Calculate grid dimensions
                    let grid_size_x = (size_x / voxel_size).ceil() as usize + 1;
                    let grid_size_y = (size_y / voxel_size).ceil() as usize + 1;
                    let grid_size_z = (size_z / voxel_size).ceil() as usize + 1;
                    
                    println!("Voxel grid dimensions: {}x{}x{} (voxel size: {:.3})",
                             grid_size_x, grid_size_y, grid_size_z, voxel_size);
                    
                    // Create a 3D grid to store voxels (using a HashSet for sparse representation)
                    let mut voxels: HashSet<(usize, usize, usize)> = HashSet::new();
                    
                    // Scanning approach: move a plane along the z-axis
                    for z_idx in 0..grid_size_z {
                        // Current z position in world space
                        let z_pos = min_z + z_idx as f32 * voxel_size;
                        
                        // For each triangle, check if it intersects with this z plane
                        for triangle_idx in 0..(indices.len() / 3) {
                            let v1_idx = indices[triangle_idx * 3] as usize;
                            let v2_idx = indices[triangle_idx * 3 + 1] as usize;
                            let v3_idx = indices[triangle_idx * 3 + 2] as usize;
                            
                            let v1 = vertices[v1_idx];
                            let v2 = vertices[v2_idx];
                            let v3 = vertices[v3_idx];
                            
                            // Check if the triangle intersects with current z plane
                            if (v1[2] <= z_pos && v2[2] >= z_pos) || 
                               (v2[2] <= z_pos && v1[2] >= z_pos) ||
                               (v2[2] <= z_pos && v3[2] >= z_pos) ||
                               (v3[2] <= z_pos && v2[2] >= z_pos) ||
                               (v3[2] <= z_pos && v1[2] >= z_pos) ||
                               (v1[2] <= z_pos && v3[2] >= z_pos) {
                                
                                // If the triangle crosses this z plane, find the line segments where it intersects
                                let mut intersections = Vec::new();
                                
                                // Check each edge of the triangle
                                let edges = [(v1, v2), (v2, v3), (v3, v1)];
                                for (va, vb) in edges.iter() {
                                    // If the edge crosses the z plane
                                    if (va[2] <= z_pos && vb[2] >= z_pos) || (va[2] >= z_pos && vb[2] <= z_pos) {
                                        // Calculate t for parametric equation of the line
                                        let t = if (vb[2] - va[2]).abs() > 1e-6 {
                                            (z_pos - va[2]) / (vb[2] - va[2])
                                        } else {
                                            0.0 // Avoid division by zero if edge is parallel to z-plane
                                        };
                                        
                                        // Calculate the intersection point
                                        let x = va[0] + t * (vb[0] - va[0]);
                                        let y = va[1] + t * (vb[1] - va[1]);
                                        
                                        intersections.push((x, y));
                                    }
                                }
                                
                                // If we have 2 intersection points, we can determine the line segment
                                if intersections.len() >= 2 {
                                    let (x1, y1) = intersections[0];
                                    let (x2, y2) = intersections[1];
                                    
                                    // Convert to voxel grid coordinates
                                    let voxel_x1 = ((x1 - min_x) / voxel_size).floor() as isize;
                                    let voxel_y1 = ((y1 - min_y) / voxel_size).floor() as isize;
                                    let voxel_x2 = ((x2 - min_x) / voxel_size).floor() as isize;
                                    let voxel_y2 = ((y2 - min_y) / voxel_size).floor() as isize;
                                    
                                    // Voxelize the line segment using Bresenham's algorithm
                                    for (x, y) in bresenham_line(voxel_x1, voxel_y1, voxel_x2, voxel_y2) {
                                        if x >= 0 && y >= 0 && x < grid_size_x as isize && y < grid_size_y as isize {
                                            voxels.insert((x as usize, y as usize, z_idx));
                                        }
                                    }
                                    
                                    // If we have a 3rd intersection point (for a triangle that has one vertex exactly on the plane)
                                    if intersections.len() > 2 {
                                        let (x3, y3) = intersections[2];
                                        let voxel_x3 = ((x3 - min_x) / voxel_size).floor() as isize;
                                        let voxel_y3 = ((y3 - min_y) / voxel_size).floor() as isize;
                                        
                                        // Also voxelize the line segment from 2nd to 3rd point
                                        for (x, y) in bresenham_line(voxel_x2, voxel_y2, voxel_x3, voxel_y3) {
                                            if x >= 0 && y >= 0 && x < grid_size_x as isize && y < grid_size_y as isize {
                                                voxels.insert((x as usize, y as usize, z_idx));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Progress indicator for longer voxelizations
                        if z_idx % 10 == 0 || z_idx == grid_size_z - 1 {
                            println!("  Processed z-slice {}/{}", z_idx + 1, grid_size_z);
                        }
                    }
                    
                    println!("Created voxel representation with {} voxels", voxels.len());
                    
                    // Print a few 2D slices of the voxel grid for visualization
                    let num_slices = 3;
                    let slice_step = grid_size_z / (num_slices + 1);
                    
                    for i in 1..=num_slices {
                        let slice_z = i * slice_step;
                        println!("\nVoxel grid slice at z={} (showing a {}x{} section):", 
                                slice_z, min(grid_size_x, 50), min(grid_size_y, 20));
                        
                        for y in 0..min(grid_size_y, 20) {
                            let mut line = String::new();
                            for x in 0..min(grid_size_x, 50) {
                                if voxels.contains(&(x, y, slice_z)) {
                                    line.push('#');
                                } else {
                                    line.push('.');
                                }
                            }
                            println!("{}", line);
                        }
                    }
                    
                    // Calculate the number of surface voxels
                    let mut surface_voxels = 0;
                    for &(x, y, z) in &voxels {
                        let mut is_surface = false;
                        
                        // Check 6-connected neighbors
                        for (dx, dy, dz) in &[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                            let nx = x as isize + dx;
                            let ny = y as isize + dy;
                            let nz = z as isize + dz;
                            
                            // Check if neighbor is outside grid bounds or empty
                            if nx < 0 || ny < 0 || nz < 0 || 
                               nx >= grid_size_x as isize || 
                               ny >= grid_size_y as isize || 
                               nz >= grid_size_z as isize ||
                               !voxels.contains(&(nx as usize, ny as usize, nz as usize)) {
                                is_surface = true;
                                break;
                            }
                        }
                        
                        if is_surface {
                            surface_voxels += 1;
                        }
                    }
                    
                    // Initial voxel statistics before filling
                    println!("\nInitial voxel statistics (before filling):");
                    println!("Total solid voxels: {}", voxels.len());
                    println!("Surface voxels: {}", surface_voxels);
                    println!("Interior voxels: {}", voxels.len() - surface_voxels);
                    
                    // Perform 2D flood fill on each slice and merge interior regions into voxels
                    println!("\nPerforming 2D flood fill on slices and merging interior regions into solid voxels...");
                    
                    // Track the total number of added voxels
                    let mut total_filled_voxels = 0;
                    let mut filled_slices = 0;
                    
                    // Process each z-slice separately
                    for z in 0..grid_size_z {
                        // Create a 2D grid representing this slice
                        let mut solid_grid = vec![vec![false; grid_size_y]; grid_size_x];
                        
                        // Mark solid voxels in the 2D grid
                        for x in 0..grid_size_x {
                            for y in 0..grid_size_y {
                                if voxels.contains(&(x, y, z)) {
                                    solid_grid[x][y] = true;
                                }
                            }
                        }
                        
                        // First, identify the exterior region of this slice using a 2D flood fill from the boundary
                        let mut exterior = HashSet::new();
                        let mut visited = HashSet::new();
                        
                        // Start flood fill from the boundary (edges of the slice)
                        let mut queue = Vec::new();
                        
                        // Add boundary points to the queue
                        for x in 0..grid_size_x {
                            if !solid_grid[x][0] {
                                queue.push((x, 0));
                                visited.insert((x, 0));
                            }
                            if !solid_grid[x][grid_size_y - 1] {
                                queue.push((x, grid_size_y - 1));
                                visited.insert((x, grid_size_y - 1));
                            }
                        }
                        
                        for y in 0..grid_size_y {
                            if !solid_grid[0][y] {
                                queue.push((0, y));
                                visited.insert((0, y));
                            }
                            if !solid_grid[grid_size_x - 1][y] {
                                queue.push((grid_size_x - 1, y));
                                visited.insert((grid_size_x - 1, y));
                            }
                        }
                        
                        // Perform flood fill to identify exterior
                        while !queue.is_empty() {
                            let (cx, cy) = queue.remove(0);
                            exterior.insert((cx, cy));
                            
                            // Check 4-connected neighbors
                            for (dx, dy) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
                                let nx = (cx as isize + dx) as usize;
                                let ny = (cy as isize + dy) as usize;
                                
                                // Check if neighbor is inside grid bounds, is empty, and hasn't been visited yet
                                if nx < grid_size_x && ny < grid_size_y && !solid_grid[nx][ny] && !visited.contains(&(nx, ny)) {
                                    visited.insert((nx, ny));
                                    queue.push((nx, ny));
                                }
                            }
                        }
                        
                        // Track how many voxels were filled in this slice
                        let mut filled_in_slice = 0;
                        
                        // Now find interior regions (not exterior and not solid) and add them to voxels
                        for x in 0..grid_size_x {
                            for y in 0..grid_size_y {
                                if !solid_grid[x][y] && !exterior.contains(&(x, y)) && !visited.contains(&(x, y)) {
                                    // We found an interior region, start a new flood fill
                                    let mut region_size = 0;
                                    let mut region_queue = vec![(x, y)];
                                    visited.insert((x, y));
                                    
                                    // Perform flood fill (breadth-first search) in 2D
                                    while !region_queue.is_empty() {
                                        let (cx, cy) = region_queue.remove(0);
                                        
                                        // Add to voxels array - this is the key change!
                                        voxels.insert((cx, cy, z));
                                        region_size += 1;
                                        
                                        // Check 4-connected neighbors
                                        for (dx, dy) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
                                            let nx = (cx as isize + dx) as usize;
                                            let ny = (cy as isize + dy) as usize;
                                            
                                            // Check if neighbor is inside grid bounds, is empty, and hasn't been visited yet
                                            if nx < grid_size_x && ny < grid_size_y && 
                                               !solid_grid[nx][ny] && !visited.contains(&(nx, ny)) {
                                                visited.insert((nx, ny));
                                                region_queue.push((nx, ny));
                                            }
                                        }
                                    }
                                    
                                    filled_in_slice += region_size;
                                }
                            }
                        }
                        
                        // Only report if we filled something
                        if filled_in_slice > 0 {
                            println!("  Slice z={}: Filled {} interior voxels", z, filled_in_slice);
                            total_filled_voxels += filled_in_slice;
                            filled_slices += 1;
                        }
                    }
                    
                    // Report filling statistics
                    println!("\nFill analysis:");
                    println!("Filled {} interior voxels across {} slices", total_filled_voxels, filled_slices);
                    println!("Voxels before filling: {}", voxels.len() - total_filled_voxels);
                    println!("Voxels after filling: {}", voxels.len());
                    
                    // Calculate new surface voxels after filling
                    let mut new_surface_voxels = 0;
                    for &(x, y, z) in &voxels {
                        let mut is_surface = false;
                        
                        // Check 6-connected neighbors
                        for (dx, dy, dz) in &[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                            let nx = x as isize + dx;
                            let ny = y as isize + dy;
                            let nz = z as isize + dz;
                            
                            // Check if neighbor is outside grid bounds or empty
                            if nx < 0 || ny < 0 || nz < 0 || 
                               nx >= grid_size_x as isize || 
                               ny >= grid_size_y as isize || 
                               nz >= grid_size_z as isize ||
                               !voxels.contains(&(nx as usize, ny as usize, nz as usize)) {
                                is_surface = true;
                                break;
                            }
                        }
                        
                        if is_surface {
                            new_surface_voxels += 1;
                        }
                    }
                    
                    println!("Surface voxels after filling: {}", new_surface_voxels);
                    println!("Interior voxels after filling: {}", voxels.len() - new_surface_voxels);
                    
                    // Visualize a few slices to show the result of filling
                    let slices_to_show = min(5, filled_slices);
                    let slice_step = if filled_slices > 0 { grid_size_z / (slices_to_show + 1) } else { grid_size_z / 4 };
                    
                    for i in 1..=slices_to_show {
                        let slice_z = i * slice_step;
                        if slice_z < grid_size_z {
                            println!("\nVoxel grid after filling at z={} (showing a {}x{} section):", 
                                    slice_z, min(grid_size_x, 50), min(grid_size_y, 20));
                            
                            for y in 0..min(grid_size_y, 20) {
                                let mut line = String::new();
                                for x in 0..min(grid_size_x, 50) {
                                    if voxels.contains(&(x, y, slice_z)) {
                                        line.push('#');
                                    } else {
                                        line.push('.');
                                    }
                                }
                                println!("{}", line);
                            }
                        }
                    }
                    
                    // Now perform thinning to create a skeleton representation
                    println!("\nPerforming 3D thinning to create skeleton...");
                    
                    // First, convert our sparse voxel representation to a dense 3D array for faster lookup
                    let mut dense_grid = vec![vec![vec![false; grid_size_z]; grid_size_y]; grid_size_x];
                    for &(x, y, z) in &voxels {
                        dense_grid[x][y][z] = true;
                    }
                    
                    // Create new empty set for the skeleton
                    let mut skeleton = HashSet::new();
                    
                    // Helper function to check if a voxel can be removed safely
                    // This is the core of the thinning algorithm
                    fn can_remove_voxel(grid: &Vec<Vec<Vec<bool>>>, x: usize, y: usize, z: usize, grid_size_x: usize, grid_size_y: usize, grid_size_z: usize) -> bool {
                        // If the voxel is at the boundary of the grid, don't remove it
                        if x == 0 || y == 0 || z == 0 || x == grid_size_x - 1 || y == grid_size_y - 1 || z == grid_size_z - 1 {
                            return false;
                        }
                        
                        // Check if this is a surface voxel (has at least one empty 6-neighbor)
                        let mut is_surface = false;
                        for (dx, dy, dz) in &[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                            let nx = (x as isize + dx) as usize;
                            let ny = (y as isize + dy) as usize;
                            let nz = (z as isize + dz) as usize;
                            
                            if !grid[nx][ny][nz] {
                                is_surface = true;
                                break;
                            }
                        }
                        
                        if !is_surface {
                            return false; // Not a surface voxel
                        }
                        
                        // Now we need to check if removing this voxel would disconnect the component
                        // We do this by counting the number of connected components in the 26-neighborhood
                        // after temporarily removing the center voxel
                        
                        // Create a temporary grid of the 3x3x3 neighborhood
                        let mut temp_grid = vec![vec![vec![false; 3]; 3]; 3];
                        
                        // Fill the temporary grid from the main grid (excluding the center)
                        for dx in -1..=1 {
                            for dy in -1..=1 {
                                for dz in -1..=1 {
                                    if dx == 0 && dy == 0 && dz == 0 {
                                        continue; // Skip the center
                                    }
                                    
                                    let nx = (x as isize + dx) as usize;
                                    let ny = (y as isize + dy) as usize;
                                    let nz = (z as isize + dz) as usize;
                                    
                                    temp_grid[(dx + 1) as usize][(dy + 1) as usize][(dz + 1) as usize] = grid[nx][ny][nz];
                                }
                            }
                        }
                        
                        // Count connected components in the 26-neighborhood using BFS
                        let mut visited = vec![vec![vec![false; 3]; 3]; 3];
                        let mut component_count = 0;
                        
                        for cx in 0..3 {
                            for cy in 0..3 {
                                for cz in 0..3 {
                                    if temp_grid[cx][cy][cz] && !visited[cx][cy][cz] {
                                        // Found a new component, perform BFS
                                        component_count += 1;
                                        let mut queue = vec![(cx, cy, cz)];
                                        visited[cx][cy][cz] = true;
                                        
                                        while !queue.is_empty() {
                                            let (qx, qy, qz) = queue.remove(0);
                                            
                                            // Check all 26 neighbors
                                            for nx in (qx as isize - 1)..=(qx as isize + 1) {
                                                for ny in (qy as isize - 1)..=(qy as isize + 1) {
                                                    for nz in (qz as isize - 1)..=(qz as isize + 1) {
                                                        if nx >= 0 && ny >= 0 && nz >= 0 && nx < 3 && ny < 3 && nz < 3 {
                                                            let nx = nx as usize;
                                                            let ny = ny as usize;
                                                            let nz = nz as usize;
                                                            
                                                            if temp_grid[nx][ny][nz] && !visited[nx][ny][nz] {
                                                                visited[nx][ny][nz] = true;
                                                                queue.push((nx, ny, nz));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Can remove if there's only one connected component in the neighborhood
                        component_count <= 1
                    }
                    
                    // Implement thinning algorithm
                    let mut changed = true;
                    let mut iteration = 0;
                    let mut removed_count = 0;
                    
                    // We'll use a directional thinning approach (6 directions)
                    let directions = [
                        (1, 0, 0), (-1, 0, 0),   // x-axis
                        (0, 1, 0), (0, -1, 0),   // y-axis
                        (0, 0, 1), (0, 0, -1)    // z-axis
                    ];
                    
                    // Start with the original filled voxels
                    let mut thinned_grid = dense_grid.clone();
                    
                    // Limit iterations to prevent infinite loop
                    let max_iterations = 100;
                    
                    while changed && iteration < max_iterations {
                        changed = false;
                        iteration += 1;
                        let mut removed_in_iteration = 0;
                        
                        // Process each direction separately
                        for (direction_idx, &(dx, dy, dz)) in directions.iter().enumerate() {
                            // Collect voxels that can be removed in this iteration
                            let mut to_remove = Vec::new();
                            
                            for x in 1..grid_size_x-1 {
                                for y in 1..grid_size_y-1 {
                                    for z in 1..grid_size_z-1 {
                                        if thinned_grid[x][y][z] {
                                            // Check if this voxel has a neighbor in the current direction
                                            let nx = (x as isize + dx) as usize;
                                            let ny = (y as isize + dy) as usize;
                                            let nz = (z as isize + dz) as usize;
                                            
                                            // Only consider voxels that have an empty neighbor in the current direction
                                            if !thinned_grid[nx][ny][nz] && can_remove_voxel(&thinned_grid, x, y, z, grid_size_x, grid_size_y, grid_size_z) {
                                                to_remove.push((x, y, z));
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Remove the collected voxels
                            for (x, y, z) in to_remove {
                                thinned_grid[x][y][z] = false;
                                removed_in_iteration += 1;
                            }
                        }
                        
                        if removed_in_iteration > 0 {
                            changed = true;
                            removed_count += removed_in_iteration;
                            println!("  Iteration {}: Removed {} voxels", iteration, removed_in_iteration);
                        }
                    }
                    
                    // Convert back to sparse representation
                    for x in 0..grid_size_x {
                        for y in 0..grid_size_y {
                            for z in 0..grid_size_z {
                                if thinned_grid[x][y][z] {
                                    skeleton.insert((x, y, z));
                                }
                            }
                        }
                    }
                    
                    println!("\nSkeleton analysis:");
                    println!("Total voxels before thinning: {}", voxels.len());
                    println!("Total voxels after thinning: {}", skeleton.len());
                    println!("Removed {} voxels ({:.1}%)", voxels.len() - skeleton.len(), 
                             100.0 * (voxels.len() - skeleton.len()) as f32 / voxels.len() as f32);
                    
                    // Visualize a few slices of the skeleton
                    for i in 1..=slices_to_show {
                        let slice_z = i * slice_step;
                        if slice_z < grid_size_z {
                            println!("\nSkeleton at z={} (showing a {}x{} section):", 
                                    slice_z, min(grid_size_x, 50), min(grid_size_y, 20));
                            
                            for y in 0..min(grid_size_y, 20) {
                                let mut line = String::new();
                                for x in 0..min(grid_size_x, 50) {
                                    if skeleton.contains(&(x, y, slice_z)) {
                                        line.push('#');
                                    } else {
                                        line.push('.');
                                    }
                                }
                                println!("{}", line);
                            }
                        }
                    }
                    
                    // Now approximate the skeleton with minimal line segments
                    println!("\nApproximating skeleton with minimal line segments...");
                    
                    // Convert skeleton coordinates to real-world coordinates
                    let mut real_points = Vec::new();
                    for &(x, y, z) in &skeleton {
                        let real_x = min_x + x as f32 * voxel_size;
                        let real_y = min_y + y as f32 * voxel_size;
                        let real_z = min_z + z as f32 * voxel_size;
                        real_points.push([real_x, real_y, real_z]);
                    }
                    
                    // Define squared error threshold
                    let max_error_squared = voxel_size * voxel_size * 2.0; // Allow error up to sqrt(2) voxels
                    
                    // Define a line segment structure
                    #[derive(Clone, Debug)]
                    struct LineSegment {
                        start: [f32; 3],
                        end: [f32; 3],
                        points_covered: Vec<usize>, // Indices of points covered by this segment
                    }
                    
                    // Helper function to calculate squared distance from a point to a line segment
                    fn point_to_segment_distance_squared(point: &[f32; 3], segment: &LineSegment) -> f32 {
                        let start = &segment.start;
                        let end = &segment.end;
                        
                        // Vector from start to end
                        let segment_vector = [
                            end[0] - start[0],
                            end[1] - start[1],
                            end[2] - start[2]
                        ];
                        
                        // Vector from start to point
                        let point_vector = [
                            point[0] - start[0],
                            point[1] - start[1],
                            point[2] - start[2]
                        ];
                        
                        // Calculate squared length of segment
                        let segment_length_squared = segment_vector[0] * segment_vector[0] +
                                                   segment_vector[1] * segment_vector[1] +
                                                   segment_vector[2] * segment_vector[2];
                        
                        if segment_length_squared < 1e-6 {
                            // Segment is essentially a point, return distance to start point
                            return point_vector[0] * point_vector[0] +
                                   point_vector[1] * point_vector[1] +
                                   point_vector[2] * point_vector[2];
                        }
                        
                        // Calculate dot product of segment_vector and point_vector
                        let t = (point_vector[0] * segment_vector[0] +
                               point_vector[1] * segment_vector[1] +
                               point_vector[2] * segment_vector[2]) / segment_length_squared;
                        
                        if t < 0.0 {
                            // Point is beyond the start of the segment
                            return point_vector[0] * point_vector[0] +
                                   point_vector[1] * point_vector[1] +
                                   point_vector[2] * point_vector[2];
                        } else if t > 1.0 {
                            // Point is beyond the end of the segment
                            let end_to_point = [
                                point[0] - end[0],
                                point[1] - end[1],
                                point[2] - end[2]
                            ];
                            return end_to_point[0] * end_to_point[0] +
                                   end_to_point[1] * end_to_point[1] +
                                   end_to_point[2] * end_to_point[2];
                        } else {
                            // Closest point is on the segment
                            let closest = [
                                start[0] + t * segment_vector[0],
                                start[1] + t * segment_vector[1],
                                start[2] + t * segment_vector[2]
                            ];
                            
                            let diff = [
                                point[0] - closest[0],
                                point[1] - closest[1],
                                point[2] - closest[2]
                            ];
                            
                            return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                        }
                    }
                    
                    // Helper function to optimize segment endpoints to minimize error
                    fn optimize_segment_endpoints(points: &Vec<[f32; 3]>, segment: &LineSegment) -> LineSegment {
                        // For simplicity, use the endpoints of the covered points
                        if segment.points_covered.len() <= 2 {
                            return segment.clone();
                        }
                        
                        // Find the two points that are furthest apart
                        let mut max_dist_squared = 0.0;
                        let mut furthest_pair = (0, 0);
                        
                        for i in 0..segment.points_covered.len() {
                            let idx1 = segment.points_covered[i];
                            let p1 = points[idx1];
                            
                            for j in i+1..segment.points_covered.len() {
                                let idx2 = segment.points_covered[j];
                                let p2 = points[idx2];
                                
                                let dx = p2[0] - p1[0];
                                let dy = p2[1] - p1[1];
                                let dz = p2[2] - p1[2];
                                
                                let dist_squared = dx * dx + dy * dy + dz * dz;
                                
                                if dist_squared > max_dist_squared {
                                    max_dist_squared = dist_squared;
                                    furthest_pair = (idx1, idx2);
                                }
                            }
                        }
                        
                        // Use the furthest points as the new endpoints
                        let new_start = points[furthest_pair.0];
                        let new_end = points[furthest_pair.1];
                        
                        LineSegment {
                            start: new_start,
                            end: new_end,
                            points_covered: segment.points_covered.clone(),
                        }
                    }
                    
                    // Function to find the best segment from a point
                    fn find_best_segment(
                        start_idx: usize,
                        uncovered_points: &HashSet<usize>,
                        real_points: &Vec<[f32; 3]>,
                        max_error_squared: f32
                    ) -> Option<LineSegment> {
                        let start_point = real_points[start_idx];
                        let mut best_segment = None;
                        let mut best_coverage = 0;
                        
                        // Try various end points and pick the one that covers the most points
                        for &end_idx in uncovered_points {
                            if end_idx == start_idx {
                                continue;
                            }
                            
                            let end_point = real_points[end_idx];
                            
                            // Create a candidate segment
                            let candidate = LineSegment {
                                start: start_point,
                                end: end_point,
                                points_covered: Vec::new(),
                            };
                            
                            // Count points covered by this segment
                            let mut covered = Vec::new();
                            
                            for &point_idx in uncovered_points {
                                let point = real_points[point_idx];
                                let dist_squared = point_to_segment_distance_squared(&point, &candidate);
                                
                                if dist_squared <= max_error_squared {
                                    covered.push(point_idx);
                                }
                            }
                            
                            if covered.len() > best_coverage {
                                best_coverage = covered.len();
                                let mut best_candidate = candidate;
                                best_candidate.points_covered = covered;
                                best_segment = Some(best_candidate);
                            }
                        }
                        
                        best_segment
                    }
                    
                    // Use a greedy algorithm to find good segments
                    let mut segments = Vec::new();
                    let mut uncovered_points: HashSet<usize> = (0..real_points.len()).collect();
                    
                    while !uncovered_points.is_empty() {
                        // Sample a few starting points to find a good segment
                        let mut best_segment = None;
                        let mut best_coverage = 0;
                        
                        // Sample up to 10 different starting points
                        let sample_size = min(10, uncovered_points.len());
                        let start_indices: Vec<usize> = uncovered_points.iter().take(sample_size).cloned().collect();
                        
                        for &start_idx in &start_indices {
                            if let Some(segment) = find_best_segment(start_idx, &uncovered_points, &real_points, max_error_squared) {
                                if segment.points_covered.len() > best_coverage {
                                    best_coverage = segment.points_covered.len();
                                    best_segment = Some(segment);
                                }
                            }
                        }
                        
                        // Add best segment to our list and remove covered points
                        if let Some(segment) = best_segment {
                            // Further optimize the segment endpoints to minimize error
                            let optimized_segment = optimize_segment_endpoints(&real_points, &segment);
                            
                            // Remove covered points from uncovered set
                            for &idx in &optimized_segment.points_covered {
                                uncovered_points.remove(&idx);
                            }
                            
                            println!("  Added segment covering {} points, {} points remain", 
                                     optimized_segment.points_covered.len(), uncovered_points.len());
                            
                            segments.push(optimized_segment);
                        } else {
                            // Fallback: just add a single point as a degenerate segment
                            let idx = *uncovered_points.iter().next().unwrap();
                            uncovered_points.remove(&idx);
                            
                            let point = real_points[idx];
                            segments.push(LineSegment {
                                start: point,
                                end: point,
                                points_covered: vec![idx],
                            });
                        }
                        
                        // Limit the number of segments to avoid excessive processing
                        if segments.len() >= 100 {
                            println!("  Reached maximum segment count, stopping approximation");
                            break;
                        }
                    }
                    
                    println!("\nSegment approximation complete:");
                    println!("Approximated {} skeleton points with {} line segments", 
                             real_points.len(), segments.len());
                    
                    // Calculate average error
                    let mut total_error: f32 = 0.0;
                    let mut max_error: f32 = 0.0;
                    
                    for (i, point) in real_points.iter().enumerate() {
                        // Find closest segment
                        let mut min_dist_squared = f32::MAX;
                        
                        for segment in &segments {
                            if segment.points_covered.contains(&i) {
                                let dist_squared = point_to_segment_distance_squared(point, segment);
                                min_dist_squared = min_dist_squared.min(dist_squared);
                            }
                        }
                        
                        total_error += min_dist_squared;
                        max_error = max_error.max(min_dist_squared);
                    }
                    
                    let avg_error = (total_error / real_points.len() as f32).sqrt();
                    let max_error = max_error.sqrt();
                    
                    println!("Average error: {:.3} voxels", avg_error / voxel_size);
                    println!("Maximum error: {:.3} voxels", max_error / voxel_size);
                    
                    // Output segments in world coordinates
                    println!("\nSegment endpoints in world coordinates:");
                    for (i, segment) in segments.iter().enumerate().take(10) {
                        println!("Segment {}: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3}) - covers {} points",
                                 i, segment.start[0], segment.start[1], segment.start[2],
                                 segment.end[0], segment.end[1], segment.end[2],
                                 segment.points_covered.len());
                    }
                    if segments.len() > 10 {
                        println!("... and {} more segments", segments.len() - 10);
                    }
                    
                    // Create a hierarchical structure of line segments
                    println!("\nCreating hierarchical skeleton structure...");
                    
                    // Sort segments by length (descending)
                    let mut segments_with_length: Vec<(usize, f32)> = segments.iter().enumerate()
                        .map(|(i, seg)| {
                            let dx = seg.end[0] - seg.start[0];
                            let dy = seg.end[1] - seg.start[1];
                            let dz = seg.end[2] - seg.start[2];
                            let length = (dx * dx + dy * dy + dz * dz).sqrt();
                            (i, length)
                        })
                        .collect();
                    segments_with_length.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    
                    // Define a joint structure
                    #[derive(Debug, Clone)]
                    struct Joint {
                        index: usize,                   // Joint index
                        name: String,                   // Joint name
                        position: [f32; 3],             // Joint position
                        orientation: [f32; 4],          // Joint orientation as quaternion
                        length: f32,                    // Length to child
                        parent_idx: Option<usize>,      // Parent joint index
                        children: Vec<usize>,           // Child joint indices
                        segment_idx: Option<usize>,     // Associated segment index
                    }
                    
                    // Create joints array
                    let mut joints: Vec<Joint> = Vec::new();
                    let mut joint_to_segment: Vec<usize> = Vec::new(); // Maps joint index to segment index
                    let mut segment_to_joints: Vec<(usize, usize)> = Vec::new(); // Maps segment index to (start_joint, end_joint)
                    
                    // Create a root joint
                    let root_joint = Joint {
                        index: 0,
                        name: "root".to_string(),
                        position: [0.0, 0.0, 0.0], // Will be updated
                        orientation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
                        length: 0.0,
                        parent_idx: None,
                        children: Vec::new(),
                        segment_idx: None,
                    };
                    joints.push(root_joint);
                    
                    // Function to calculate rotation quaternion from vec3 to vec3
                    fn rotation_between_vectors(from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
                        // Normalize vectors
                        let from_length = (from[0] * from[0] + from[1] * from[1] + from[2] * from[2]).sqrt();
                        let to_length = (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]).sqrt();
                        
                        if from_length < 1e-6 || to_length < 1e-6 {
                            return [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
                        }
                        
                        let from_normalized = [
                            from[0] / from_length,
                            from[1] / from_length,
                            from[2] / from_length
                        ];
                        
                        let to_normalized = [
                            to[0] / to_length,
                            to[1] / to_length,
                            to[2] / to_length
                        ];
                        
                        // Compute dot product
                        let dot = from_normalized[0] * to_normalized[0] +
                                 from_normalized[1] * to_normalized[1] +
                                 from_normalized[2] * to_normalized[2];
                        
                        // If vectors are nearly aligned
                        if dot > 0.99999 {
                            return [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
                        }
                        
                        // If vectors are nearly opposite
                        if dot < -0.99999 {
                            // Find an orthogonal vector to 'from'
                            let mut axis = [1.0, 0.0, 0.0]; // Try x-axis
                            if from_normalized[0].abs() > 0.9 {
                                axis = [0.0, 1.0, 0.0]; // Try y-axis instead
                            }
                            
                            // Cross product to get perpendicular vector
                            let perpendicular = [
                                from_normalized[1] * axis[2] - from_normalized[2] * axis[1],
                                from_normalized[2] * axis[0] - from_normalized[0] * axis[2],
                                from_normalized[0] * axis[1] - from_normalized[1] * axis[0]
                            ];
                            
                            // Normalize
                            let perp_length = (perpendicular[0] * perpendicular[0] + 
                                             perpendicular[1] * perpendicular[1] + 
                                             perpendicular[2] * perpendicular[2]).sqrt();
                            
                            return [
                                perpendicular[0] / perp_length,
                                perpendicular[1] / perp_length,
                                perpendicular[2] / perp_length,
                                0.0 // 180 degree rotation
                            ];
                        }
                        
                        // Cross product to get rotation axis
                        let axis = [
                            from_normalized[1] * to_normalized[2] - from_normalized[2] * to_normalized[1],
                            from_normalized[2] * to_normalized[0] - from_normalized[0] * to_normalized[2],
                            from_normalized[0] * to_normalized[1] - from_normalized[1] * to_normalized[0]
                        ];
                        
                        // Normalize axis
                        let axis_length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
                        
                        // Calculate quaternion components
                        let s = (1.0 + dot).sqrt() * 0.5;
                        let recip_s = 0.5 / s;
                        
                        [
                            axis[0] * recip_s,
                            axis[1] * recip_s,
                            axis[2] * recip_s,
                            s
                        ]
                    }
                    
                    // Helper function to find the closest joint to a point
                    fn find_closest_joint_to_point(point: [f32; 3], joints: &Vec<Joint>) -> Option<usize> {
                        let mut closest_idx = None;
                        let mut min_distance = f32::MAX;
                        
                        for (i, joint) in joints.iter().enumerate() {
                            let dx = joint.position[0] - point[0];
                            let dy = joint.position[1] - point[1];
                            let dz = joint.position[2] - point[2];
                            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                            
                            if distance < min_distance {
                                min_distance = distance;
                                closest_idx = Some(i);
                            }
                        }
                        
                        closest_idx
                    }
                    
                    // Process segments in order of length to build hierarchy
                    // Start with longest segment as root joint's child
                    if !segments_with_length.is_empty() {
                        let (root_segment_idx, root_segment_length) = segments_with_length[0];
                        let root_segment = &segments[root_segment_idx];
                        
                        // Set root joint position to longest segment's start
                        joints[0].position = root_segment.start;
                        
                        // Create end joint for longest segment
                        let root_child_idx = joints.len();
                        let mut root_child = Joint {
                            index: root_child_idx,
                            name: format!("joint_{}", root_child_idx),
                            position: root_segment.end,
                            orientation: [0.0, 0.0, 0.0, 1.0], // Will be computed
                            length: root_segment_length,
                            parent_idx: Some(0), // Root is parent
                            children: Vec::new(),
                            segment_idx: Some(root_segment_idx),
                        };
                        
                        // Calculate orientation (pointing toward end)
                        let direction = [
                            root_segment.end[0] - root_segment.start[0],
                            root_segment.end[1] - root_segment.start[1],
                            root_segment.end[2] - root_segment.start[2]
                        ];
                        
                        // Convert to quaternion (from Y-up to direction)
                        let default_dir = [0.0, 1.0, 0.0]; // Y-up
                        root_child.orientation = rotation_between_vectors(default_dir, direction);
                        
                        // Update parent's children list
                        joints[0].children.push(root_child_idx);
                        
                        // Add to joints array
                        joints.push(root_child);
                        
                        // Add mapping
                        segment_to_joints.push((0, root_child_idx));
                        joint_to_segment.push(root_segment_idx);
                        
                        // Process remaining segments
                        for &(segment_idx, segment_length) in segments_with_length.iter().skip(1) {
                            let segment = &segments[segment_idx];
                            
                            // Find closest existing joint to segment start
                            if let Some(closest_start_joint) = find_closest_joint_to_point(segment.start, &joints) {
                                // Create a new joint for segment end
                                let new_joint_idx = joints.len();
                                let mut new_joint = Joint {
                                    index: new_joint_idx,
                                    name: format!("joint_{}", new_joint_idx),
                                    position: segment.end,
                                    orientation: [0.0, 0.0, 0.0, 1.0], // Will be computed
                                    length: segment_length,
                                    parent_idx: Some(closest_start_joint),
                                    children: Vec::new(),
                                    segment_idx: Some(segment_idx),
                                };
                                
                                // Calculate orientation
                                let direction = [
                                    segment.end[0] - segment.start[0],
                                    segment.end[1] - segment.start[1],
                                    segment.end[2] - segment.start[2]
                                ];
                                
                                // Convert to quaternion
                                let default_dir = [0.0, 1.0, 0.0]; // Y-up
                                new_joint.orientation = rotation_between_vectors(default_dir, direction);
                                
                                // Update parent's children list
                                joints[closest_start_joint].children.push(new_joint_idx);
                                
                                // Add to joints array
                                joints.push(new_joint);
                                
                                // Add mapping
                                segment_to_joints.push((closest_start_joint, new_joint_idx));
                                joint_to_segment.push(segment_idx);
                            }
                        }
                    }
                    
                    println!("\nJoint hierarchy created with {} joints", joints.len());
                    println!("Joint hierarchy:");
                    
                    // Print joint hierarchy
                    fn print_joint_hierarchy(joint_idx: usize, joints: &Vec<Joint>, depth: usize) {
                        let joint = &joints[joint_idx];
                        let indent = "  ".repeat(depth);
                        println!("{}Joint {}: \"{}\" at ({:.3}, {:.3}, {:.3})", 
                                 indent, joint.index, joint.name, 
                                 joint.position[0], joint.position[1], joint.position[2]);
                        
                        for &child_idx in &joint.children {
                            print_joint_hierarchy(child_idx, joints, depth + 1);
                        }
                    }
                    
                    print_joint_hierarchy(0, &joints, 0);
                    
                    // Now create skinning weights for the vertices
                    println!("\nCalculating skinning weights for vertices...");
                    
                    // Create an array of weights for each vertex
                    let mut vertex_weights = Vec::new();
                    
                    // Calculate weights for each original vertex in the mesh
                    for vertex_idx in 0..vertices.len() {
                        let vertex = vertices[vertex_idx];
                        
                        // Find the two closest joints
                        let mut closest_joints = Vec::new();
                        let mut closest_distances = Vec::new();
                        
                        for (joint_idx, joint) in joints.iter().enumerate() {
                            // Skip root joint for weighting
                            if joint_idx == 0 {
                                continue;
                            }
                            
                            // Calculate distance to joint
                            let dx = joint.position[0] - vertex[0];
                            let dy = joint.position[1] - vertex[1];
                            let dz = joint.position[2] - vertex[2];
                            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                            
                            // Insert into sorted array of closest joints
                            let mut inserted = false;
                            for i in 0..closest_joints.len() {
                                if distance < closest_distances[i] {
                                    closest_joints.insert(i, joint_idx);
                                    closest_distances.insert(i, distance);
                                    inserted = true;
                                    break;
                                }
                            }
                            
                            if !inserted && closest_joints.len() < 2 {
                                closest_joints.push(joint_idx);
                                closest_distances.push(distance);
                            }
                            
                            // Keep only the closest 2
                            if closest_joints.len() > 2 {
                                closest_joints.pop();
                                closest_distances.pop();
                            }
                        }
                        
                        // Calculate weights using inverse distance
                        let mut weights = Vec::new();
                        
                        if closest_joints.len() == 2 {
                            // Use inverse distance weighting
                            let inv_dist1 = 1.0 / (closest_distances[0] + 1e-6);
                            let inv_dist2 = 1.0 / (closest_distances[1] + 1e-6);
                            let sum = inv_dist1 + inv_dist2;
                            
                            let w1 = inv_dist1 / sum;
                            let w2 = inv_dist2 / sum;
                            
                            weights.push((closest_joints[0], w1));
                            weights.push((closest_joints[1], w2));
                        } else if closest_joints.len() == 1 {
                            // If only one joint is close, give it full weight
                            weights.push((closest_joints[0], 1.0));
                        }
                        
                        vertex_weights.push(weights);
                    }
                    
                    println!("Successfully calculated skinning weights for {} vertices", vertex_weights.len());
                    
                    // Output the skinned model to GLTF format
                    println!("\nPreparing to export skinned model as GLTF...");
                    
                    // Generate a simple preview of the skeleton
                    println!("\nSkeleton preview (showing joint connections):");
                    for (i, joint) in joints.iter().enumerate() {
                        if i == 0 { continue; } // Skip root
                        
                        let parent_idx = joint.parent_idx.unwrap();
                        let parent = &joints[parent_idx];
                        
                        println!("  Joint line: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3})",
                                 parent.position[0], parent.position[1], parent.position[2],
                                 joint.position[0], joint.position[1], joint.position[2]);
                    }
                    
                    // Export the model as a GLB file with skin
                    println!("\nExporting rigged model as GLB...");
                    
                    // Define the output file path
                    let output_path = "rigged_model.glb";
                    
                    // Create a new GLTF document
                    use std::collections::HashMap as StdHashMap;
                    use std::io::Write;
                    
                    // We'll use GLB binary format which combines JSON and binary data in one file
                    // Create a basic GLTF structure for the JSON chunk
                    let mut gltf_json = json!({
                        "asset": {
                            "version": "2.0",
                            "generator": "glb-rigger"
                        },
                        "scene": 0,
                        "scenes": [
                            {
                                "nodes": [0]  // Root node index
                            }
                        ],
                        "nodes": [],
                        "meshes": [],
                        "skins": [],
                        "accessors": [],
                        "bufferViews": [],
                        "buffers": [
                            {
                                "byteLength": 0  // Will be updated
                            }
                        ]
                    });
                    
                    // Build buffer data
                    let mut buffer_data = Vec::new();
                    let mut buffer_views = Vec::new();
                    let mut accessors = Vec::new();
                    
                    // Function to align buffer to 4-byte boundary
                    fn align_to_4bytes(buffer: &mut Vec<u8>) {
                        let remainder = buffer.len() % 4;
                        if remainder > 0 {
                            for _ in 0..(4 - remainder) {
                                buffer.push(0);
                            }
                        }
                    }
                    
                    // 1. Create nodes for the joint hierarchy
                    let mut gltf_nodes = Vec::new();
                    let mut joint_node_map = StdHashMap::new(); // Maps joint index to node index
                    
                    // First, create nodes for all joints
                    for joint in &joints {
                        let mut node = json!({
                            "name": joint.name,
                            "translation": [
                                joint.position[0],
                                joint.position[1],
                                joint.position[2]
                            ],
                            "rotation": [
                                joint.orientation[0],
                                joint.orientation[1],
                                joint.orientation[2],
                                joint.orientation[3]
                            ]
                        });
                        
                        // Store the node index for this joint
                        joint_node_map.insert(joint.index, gltf_nodes.len());
                        
                        // Add children if they exist
                        if !joint.children.is_empty() {
                            let mut children = Vec::new();
                            for &child_idx in &joint.children {
                                // Child indices will be updated later once we know all node indices
                                children.push(child_idx);
                            }
                            node["children"] = json!(children);
                        }
                        
                        gltf_nodes.push(node);
                    }
                    
                    // Update children indices based on the node map
                    for joint_idx in 0..joints.len() {
                        if let Some(&node_idx) = joint_node_map.get(&joint_idx) {
                            if !joints[joint_idx].children.is_empty() {
                                let mut updated_children = Vec::new();
                                for &child_joint_idx in &joints[joint_idx].children {
                                    if let Some(&child_node_idx) = joint_node_map.get(&child_joint_idx) {
                                        updated_children.push(child_node_idx);
                                    }
                                }
                                
                                if node_idx < gltf_nodes.len() {
                                    gltf_nodes[node_idx]["children"] = json!(updated_children);
                                }
                            }
                        }
                    }
                    
                    // 2. Create mesh data
                    
                    // Position accessor
                    let positions_byte_offset = buffer_data.len();
                    for vertex in &vertices {
                        buffer_data.extend_from_slice(&vertex[0].to_le_bytes());
                        buffer_data.extend_from_slice(&vertex[1].to_le_bytes());
                        buffer_data.extend_from_slice(&vertex[2].to_le_bytes());
                    }
                    align_to_4bytes(&mut buffer_data);
                    
                    let positions_byte_length = buffer_data.len() - positions_byte_offset;
                    let positions_buffer_view = json!({
                        "buffer": 0,
                        "byteOffset": positions_byte_offset,
                        "byteLength": positions_byte_length,
                        "target": 34962  // ARRAY_BUFFER
                    });
                    
                    let positions_accessor = json!({
                        "bufferView": buffer_views.len(),
                        "componentType": 5126,  // FLOAT
                        "count": vertices.len(),
                        "type": "VEC3",
                        "min": [min_x, min_y, min_z],
                        "max": [max_x, max_y, max_z]
                    });
                    
                    buffer_views.push(positions_buffer_view);
                    accessors.push(positions_accessor);
                    let positions_accessor_idx = accessors.len() - 1;
                    
                    // Indices accessor
                    let indices_byte_offset = buffer_data.len();
                    for &index in &indices {
                        buffer_data.extend_from_slice(&(index as u32).to_le_bytes());
                    }
                    align_to_4bytes(&mut buffer_data);
                    
                    let indices_byte_length = buffer_data.len() - indices_byte_offset;
                    let indices_buffer_view = json!({
                        "buffer": 0,
                        "byteOffset": indices_byte_offset,
                        "byteLength": indices_byte_length,
                        "target": 34963  // ELEMENT_ARRAY_BUFFER
                    });
                    
                    let indices_accessor = json!({
                        "bufferView": buffer_views.len(),
                        "componentType": 5125,  // UNSIGNED_INT
                        "count": indices.len(),
                        "type": "SCALAR"
                    });
                    
                    buffer_views.push(indices_buffer_view);
                    accessors.push(indices_accessor);
                    let indices_accessor_idx = accessors.len() - 1;
                    
                    // 3. Create joint indices and weights accessors for skinning
                    let joints_byte_offset = buffer_data.len();
                    let mut max_influences = 0;
                    
                    for weights in &vertex_weights {
                        max_influences = max(max_influences, weights.len());
                        
                        // Write up to 4 joint indices (standard GLTF maximum)
                        let mut written = 0;
                        for &(joint_idx, _) in weights.iter().take(4) {
                            let node_idx = *joint_node_map.get(&joint_idx).unwrap_or(&0) as u16;
                            buffer_data.extend_from_slice(&node_idx.to_le_bytes());
                            written += 1;
                        }
                        
                        // Pad with zeros if necessary
                        for _ in written..4 {
                            buffer_data.extend_from_slice(&(0 as u16).to_le_bytes());
                        }
                    }
                    align_to_4bytes(&mut buffer_data);
                    
                    let joints_byte_length = buffer_data.len() - joints_byte_offset;
                    let joints_buffer_view = json!({
                        "buffer": 0,
                        "byteOffset": joints_byte_offset,
                        "byteLength": joints_byte_length,
                        "target": 34962  // ARRAY_BUFFER
                    });
                    
                    let joints_accessor = json!({
                        "bufferView": buffer_views.len(),
                        "componentType": 5123,  // UNSIGNED_SHORT
                        "count": vertex_weights.len(),
                        "type": "VEC4"
                    });
                    
                    buffer_views.push(joints_buffer_view);
                    accessors.push(joints_accessor);
                    let joints_accessor_idx = accessors.len() - 1;
                    
                    // Weights accessor
                    let weights_byte_offset = buffer_data.len();
                    
                    for weights in &vertex_weights {
                        // Write up to 4 weights (standard GLTF maximum)
                        let mut written = 0;
                        for &(_, weight) in weights.iter().take(4) {
                            buffer_data.extend_from_slice(&weight.to_le_bytes());
                            written += 1;
                        }
                        
                        // Pad with zeros if necessary
                        for _ in written..4 {
                            buffer_data.extend_from_slice(&(0.0_f32).to_le_bytes());
                        }
                    }
                    align_to_4bytes(&mut buffer_data);
                    
                    let weights_byte_length = buffer_data.len() - weights_byte_offset;
                    let weights_buffer_view = json!({
                        "buffer": 0,
                        "byteOffset": weights_byte_offset,
                        "byteLength": weights_byte_length,
                        "target": 34962  // ARRAY_BUFFER
                    });
                    
                    let weights_accessor = json!({
                        "bufferView": buffer_views.len(),
                        "componentType": 5126,  // FLOAT
                        "count": vertex_weights.len(),
                        "type": "VEC4"
                    });
                    
                    buffer_views.push(weights_buffer_view);
                    accessors.push(weights_accessor);
                    let weights_accessor_idx = accessors.len() - 1;
                    
                    // 4. Create inverse bind matrices accessor
                    let ibm_byte_offset = buffer_data.len();
                    
                    // Create inverse bind matrices for each joint
                    for joint in &joints {
                        // For simplicity, we'll use identity matrices as inverses
                        // In a real application, you would compute proper inverse bind matrices
                        let identity_matrix: [f32;16] = [
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0
                        ];
                        
                        for &value in &identity_matrix {
                            buffer_data.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                    align_to_4bytes(&mut buffer_data);
                    
                    let ibm_byte_length = buffer_data.len() - ibm_byte_offset;
                    let ibm_buffer_view = json!({
                        "buffer": 0,
                        "byteOffset": ibm_byte_offset,
                        "byteLength": ibm_byte_length
                    });
                    
                    let ibm_accessor = json!({
                        "bufferView": buffer_views.len(),
                        "componentType": 5126,  // FLOAT
                        "count": joints.len(),
                        "type": "MAT4"
                    });
                    
                    buffer_views.push(ibm_buffer_view);
                    accessors.push(ibm_accessor);
                    let ibm_accessor_idx = accessors.len() - 1;
                    
                    // 5. Create mesh
                    let mesh = json!({
                        "primitives": [
                            {
                                "attributes": {
                                    "POSITION": positions_accessor_idx,
                                    "JOINTS_0": joints_accessor_idx,
                                    "WEIGHTS_0": weights_accessor_idx
                                },
                                "indices": indices_accessor_idx,
                                "mode": 4  // TRIANGLES
                            }
                        ]
                    });
                    
                    // 6. Create skin
                    let skin = json!({
                        "inverseBindMatrices": ibm_accessor_idx,
                        "joints": joint_node_map.values().collect::<Vec<_>>(),
                        "skeleton": joint_node_map[&0]  // Root joint
                    });
                    
                    // 7. Create the mesh node
                    let mesh_node = json!({
                        "mesh": 0,
                        "skin": 0
                    });
                    
                    // Add to the top of the node list
                    gltf_nodes.insert(0, mesh_node);
                    
                    // Update node references in the joint map
                    for (joint_idx, node_idx) in &mut joint_node_map {
                        *node_idx += 1; // Shift all joint node indices by 1
                    }
                    
                    // Update children references in all nodes
                    for node in &mut gltf_nodes {
                        if let Some(children) = node.get_mut("children") {
                            if let Some(children_array) = children.as_array_mut() {
                                for child_idx in children_array {
                                    if let Some(idx) = child_idx.as_u64() {
                                        *child_idx = json!(idx + 1); // Add 1 to each child index
                                    }
                                }
                            }
                        }
                    }
                    
                    // Update the main GLTF structure
                    gltf_json["nodes"] = json!(gltf_nodes);
                    gltf_json["meshes"] = json!([mesh]);
                    gltf_json["skins"] = json!([skin]);
                    gltf_json["accessors"] = json!(accessors);
                    gltf_json["bufferViews"] = json!(buffer_views);
                    gltf_json["buffers"][0]["byteLength"] = json!(buffer_data.len());
                    
                    // Write the GLB file
                    // GLB structure:
                    // 1. Header (magic, version, length)
                    // 2. JSON chunk (type, length, data)
                    // 3. BIN chunk (type, length, data)
                    
                    // Prepare the JSON chunk
                    let json_string = serde_json::to_string(&gltf_json).unwrap();
                    let mut json_bytes = json_string.into_bytes();
                    
                    // JSON chunk must be 4-byte aligned
                    while json_bytes.len() % 4 != 0 {
                        json_bytes.push(0x20); // Pad with spaces
                    }
                    
                    // Ensure binary chunk is 4-byte aligned
                    while buffer_data.len() % 4 != 0 {
                        buffer_data.push(0); // Pad with zeros
                    }
                    
                    // Calculate sizes
                    let header_length = 12; // Fixed GLB header size
                    let json_chunk_header_length = 8; // Fixed JSON chunk header size
                    let bin_chunk_header_length = 8; // Fixed BIN chunk header size
                    
                    let total_length = header_length + 
                                      json_chunk_header_length + json_bytes.len() +
                                      bin_chunk_header_length + buffer_data.len();
                    
                    // Create a file to write the GLB data
                    let mut file = fs::File::create(output_path)?;
                    
                    // Write GLB header
                    // Magic: "glTF"
                    file.write_all(&[0x67, 0x6C, 0x54, 0x46])?;
                    // Version: 2
                    file.write_all(&2u32.to_le_bytes())?;
                    // Total length
                    file.write_all(&(total_length as u32).to_le_bytes())?;
                    
                    // Write JSON chunk header
                    // Chunk length
                    file.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
                    // Chunk type: JSON (0x4E4F534A)
                    file.write_all(&[0x4A, 0x53, 0x4F, 0x4E])?;
                    // Write JSON chunk data
                    file.write_all(&json_bytes)?;
                    
                    // Write BIN chunk header
                    // Chunk length
                    file.write_all(&(buffer_data.len() as u32).to_le_bytes())?;
                    // Chunk type: BIN (0x004E4942)
                    file.write_all(&[0x42, 0x49, 0x4E, 0x00])?;
                    // Write BIN chunk data
                    file.write_all(&buffer_data)?;
                    
                    println!("Successfully exported rigged model to: {}", output_path);
                    println!("You can view this model in any GLB viewer that supports skinning.");
                } else {
                    println!("No indices found in the primitive");
                }
                
                // We only process the first mesh with triangles
                return Ok(());
            }
        }
    }
    
    println!("No mesh with triangle mode found");
    Ok(())
}

// Helper function to load buffer data
fn load_buffers(gltf: &gltf::Gltf, path: &str) -> Result<Vec<Vec<u8>>, Box<dyn StdError>> {
    let parent = Path::new(path).parent().unwrap_or_else(|| Path::new(""));
    let mut buffers = Vec::new();
    
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                if let Some(bin) = gltf.blob.as_ref() {
                    buffers.push(bin.clone());
                } else {
                    return Err("Binary buffer not found".into());
                }
            },
            gltf::buffer::Source::Uri(uri) => {
                if uri.starts_with("data:") {
                    // Handle base64 encoded data
                    return Err("Base64 data URIs not supported yet".into());
                } else {
                    // Load external buffer
                    let buffer_path = parent.join(uri);
                    let buffer_data = fs::read(buffer_path)?;
                    buffers.push(buffer_data);
                }
            }
        }
    }
    
    Ok(buffers)
}

// Helper function to read an f32 from a byte slice
fn read_f32(bytes: &[u8]) -> f32 {
    let bytes_array = [bytes[0], bytes[1], bytes[2], bytes[3]];
    f32::from_le_bytes(bytes_array)
}

// Helper function to add an edge to the adjacency map
// Normalizes the edge so that the smaller vertex index comes first
fn add_edge_to_map(
    edge_map: &mut std::collections::HashMap<(u32, u32), Vec<usize>>,
    v1: u32,
    v2: u32,
    triangle_idx: usize
) {
    // Normalize the edge (smaller index first)
    let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
    
    // Add the triangle to the edge's list
    edge_map.entry(edge)
        .or_insert_with(Vec::new)
        .push(triangle_idx);
}

// Implementation of Bresenham's line algorithm to voxelize a line segment in 2D
// Returns a vector of (x, y) coordinates that represent the voxels the line passes through
fn bresenham_line(x1: isize, y1: isize, x2: isize, y2: isize) -> Vec<(isize, isize)> {
    let mut result = Vec::new();
    
    // Calculate changes in x and y
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    
    // Determine the direction to step in x and y
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    
    // Error term for deciding which voxel to move to next
    let mut err = if dx > dy { dx } else { -dy } / 2;
    let mut err2;
    
    // Start at the first point
    let mut x = x1;
    let mut y = y1;
    
    loop {
        // Add current point to the result
        result.push((x, y));
        
        // Check if we've reached the end point
        if x == x2 && y == y2 { break; }
        
        // Update the error term
        err2 = err;
        
        // Move in x direction if needed
        if err2 > -dx {
            err -= dy;
            x += sx;
        }
        
        // Move in y direction if needed
        if err2 < dy {
            err += dx;
            y += sy;
        }
    }
    
    result
}

fn main() {
    if let Some(path) = std::env::args().nth(1) {
        run(&path).expect("runtime error");
    } else {
        println!("usage: gltf-display <FILE>");
    }
}
