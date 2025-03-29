#![allow(dead_code)]

use vector2d::Vector2D;
use vector3d::Vector3d;
use std::collections::VecDeque;

pub type Vec2 = Vector2D<f32>;
pub type Vec3 = Vector3d<f32>;


//
// Micropatch API.
//
// A Micropatch is a 8x8x8 set of bits.
// Most code shouldn't assume anything about
// its structure.
//

// hardcoded everywhere
const MICROPATCH_SIZE: u32 = 8;
#[derive(Clone, Debug, PartialEq)]
pub struct Micropatch {
    bits: [u8; 64]
}
const MICROPATCH_FULL: Micropatch = Micropatch { bits: [0xff; 64] };
const MICROPATCH_EMPTY: Micropatch = Micropatch { bits: [0x00; 64] };

impl Micropatch {
  fn set(self: &mut Micropatch, xf: u32,yf: u32,zf: u32,v: bool) -> bool {
      let p = &mut self.bits[(yf + zf * 8) as usize];
      let old = (*p >> xf) & 1;
      if v {
          *p |= 1 << xf
      } else {
          *p &= !(1 << xf)
      }
      if old == 0 { false } else { true }
  }
  fn get(self: &Micropatch, xf: u32,yf: u32,zf: u32) -> bool {
    let p = self.bits[(yf + zf * 8) as usize];
    if ((p >> xf) & 1) == 1 {
        true
    } else {
        false
    }
  }
  fn invert_inplace(self: &mut Micropatch) {
      for i in 0..64 { self.bits[i] = !self.bits[i] }
  }
  fn is_all_zeros(self: &Micropatch) -> bool {
    for i in 0..64 {
        if self.bits[i] != 0 { return false }
    }
    true
  }
  fn is_all_ones(self: &Micropatch) -> bool {
    for i in 0..64 {
        if self.bits[i] != 0xff { return false }
    }
    true
  }
  fn intersect(self: &Micropatch, other: &Micropatch) -> Micropatch {
    let mut bits: [u8; 64] = self.bits;
    for i in 0..64 {
      bits[i] &= other.bits[i]
    }
    Micropatch { bits: bits }
  }
  fn union(self: &Micropatch, other: &Micropatch) -> Micropatch {
    let mut bits: [u8; 64] = self.bits;
    for i in 0..64 {
      bits[i] |= other.bits[i]
    }
    Micropatch { bits: bits }
  }
  fn difference(self: &Micropatch, other: &Micropatch) -> Micropatch {
    let mut bits: [u8; 64] = self.bits;
    for i in 0..64 {
      bits[i] &= !other.bits[i]
    }
    Micropatch { bits: bits }
  }
  fn intersect_inplace(self: &mut Micropatch, other: &Micropatch) {
    for i in 0..64 {
      self.bits[i] &= other.bits[i]
    }
  }
  fn union_inplace(self: &mut Micropatch, other: &Micropatch) {
    for i in 0..64 {
      self.bits[i] |= other.bits[i]
    }
  }
  fn difference_inplace(self: &mut Micropatch, other: &Micropatch) {
    for i in 0..64 {
      self.bits[i] &= !other.bits[i]
    }
  }

}

//
// A Micropatch, or a cube of all 1s or 0s.
//
// For efficiency, I think we may want
// to simply use a u32 here with
// constant values (eg -2 -1) for Empty/Full.
//
#[derive(Clone, Debug)]
enum MicropatchStatus {
    Empty,
    Full,
    Subdivided(u32)
}

//
// A set of voxels in a 3-d grid of size 'size'.
// Each 8x8x8 cube of voxels is a micropatch.
//
// Empty and Full cubes are treated separately,
// without an actual array of bits.
//
pub struct VoxelSet {

    size: (u32,u32,u32),
    size_i: (usize,usize,usize),      // in micropatches
    total_micropatches: usize,
    grid: Vec<MicropatchStatus>,
    micropatches: Vec<Micropatch>,
    free_micropatches: Vec<u32>,
}

pub fn micropatch_index(v: u32) -> (u32, u32) {
   (v / MICROPATCH_SIZE, v % MICROPATCH_SIZE)
}

fn round_up(v: u32) -> usize {
    ((v + MICROPATCH_SIZE - 1) / MICROPATCH_SIZE * MICROPATCH_SIZE) as usize
}

impl VoxelSet {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        let size_i = (round_up(x), round_up(y), round_up(z));
        let total_size = size_i.0 * size_i.1 * size_i.2;
        Self {
          size: (x,y,z),
          size_i: size_i,
          total_micropatches: total_size,
          grid: vec!(MicropatchStatus::Empty; total_size),
          micropatches: Vec::new(),
          free_micropatches: Vec::new()
        }
    }

    // Create a deep clone of this VoxelSet
    pub fn clone(&self) -> Self {
        let mut new_set = Self {
            size: self.size,
            size_i: self.size_i,
            total_micropatches: self.total_micropatches,
            grid: self.grid.clone(),
            micropatches: Vec::with_capacity(self.micropatches.len()),
            free_micropatches: self.free_micropatches.clone(),
        };
        
        // Deep clone the micropatches
        for patch in &self.micropatches {
            new_set.micropatches.push(Micropatch { bits: patch.bits });
        }
        
        new_set
    }

    // Allocate a micropatch, initializing it with 'data'.
    // Return the micropatch_index, such that self.micropatches[micropatch_index]
    // will be the newly allocated patch (copied from 'data').
    fn _allocate_micropatch(self: &mut Self, data: &Micropatch) -> u32 {
          if self.free_micropatches.len() > 0 {
              let mp_index = self.free_micropatches.pop().unwrap() as usize;
              self.micropatches[mp_index] = data.clone();
              mp_index as u32
          } else {
              self.micropatches.push(data.clone());
              (self.micropatches.len() - 1) as u32
          }
    }

    // Convert the cell at 'idx' to a micropatch, if it isn't one already.
    fn _force_micropatch(self: &mut Self, idx: usize) -> &mut Micropatch {
        match self.grid[idx] {
            MicropatchStatus::Empty => {
                let mp_index = self._allocate_micropatch(&MICROPATCH_EMPTY);
                self.grid[idx] = MicropatchStatus::Subdivided(mp_index);
                &mut self.micropatches[mp_index as usize]
            },
            MicropatchStatus::Full => {
                let mp_index = self._allocate_micropatch(&MICROPATCH_FULL);
                self.grid[idx] = MicropatchStatus::Subdivided(mp_index);
                &mut self.micropatches[mp_index as usize]
            },
            MicropatchStatus::Subdivided(mp_index) =>
                &mut self.micropatches[mp_index as usize]
        }
    }

        
    pub fn set(self: &mut Self, x: u32, y: u32, z: u32, value: bool) -> bool {
        if x >= self.size.0 || y >= self.size.1 || z >= self.size.2 {
            return false
        }
        let (xi,xf) = micropatch_index(x);
        let (yi,yf) = micropatch_index(y);
        let (zi,zf) = micropatch_index(z);
        let idx = (((zi * self.size.1) * yi * self.size.2) + xi) as usize;
        let elt = &mut self.grid[idx];
        match elt {
            MicropatchStatus::Empty => {
                if value {
                    self._force_micropatch(idx).set(xf, yf, zf, value);
                }
                false
            },
            MicropatchStatus::Full => {
                if !value {
                    self._force_micropatch(idx).set(xf, yf, zf, value);
                }
                true
            },
            MicropatchStatus::Subdivided(si) => {
                let patch = &mut self.micropatches[*si as usize];

                // in the current represention, it's better to
                // clean up Full and Empty patches at a later phase.
                patch.set(xf, yf, zf, value)

                /*
                if value != set_rv {
                    let liberate = if value {
                                     if patch.is_all_ones() {
                                         *elt = MicropatchStatus::Full;
                                         true
                                     } else {
                                         false
                                     }
                                   } else {
                                     if patch.is_all_zeros() {
                                         *elt = MicropatchStatus::Empty;
                                         true
                                     } else {
                                         false
                                     }
                                   };
                    if liberate {
                        self.free_micropatches.push(mp_index)
                    }
                    !value
                } else {
                    value
                }
                */
            }
        }
    }

    pub fn get(self: &Self, x: u32, y: u32, z: u32) -> bool {
        if x >= self.size.0 || y >= self.size.1 || z >= self.size.2 {
            return false
        }
        let (xi,xf) = micropatch_index(x);
        let (yi,yf) = micropatch_index(y);
        let (zi,zf) = micropatch_index(z);
        let idx = (((zi * self.size.1) * yi * self.size.2) + xi) as usize;
        let elt = &self.grid[idx];
        match elt {
            MicropatchStatus::Empty => {
                false
            },
            MicropatchStatus::Full => {
                true
            },
            MicropatchStatus::Subdivided(si) => {
                let patch = &self.micropatches[*si as usize];
                patch.get(xf, yf, zf)
            }
        }
    }

    // Boolean union operation: self |= other
    pub fn union(&mut self, other: &Self) {
        // Ensure dimensions match
        assert_eq!(self.size, other.size, "VoxelSet dimensions must match for boolean operations");
        
        // Loop through grid cells
        for idx in 0..self.total_micropatches {
            //let self_grid = &mut self.grid[idx];
            //let other_grid = &other.grid[idx];
            match (self.grid[idx].clone(), other.grid[idx].clone()) {
                (_, MicropatchStatus::Empty) => {},
                (MicropatchStatus::Full, _) => {},
                (MicropatchStatus::Empty, MicropatchStatus::Subdivided(si)) => {
                    self.grid[idx] = MicropatchStatus::Subdivided(self._allocate_micropatch(&other.micropatches[si as usize]))
                },
                (MicropatchStatus::Empty, MicropatchStatus::Full) => {
                    self.grid[idx] = MicropatchStatus::Full
                },
                (MicropatchStatus::Subdivided(a_si), MicropatchStatus::Subdivided(b_si)) => {
                    let a = &mut self.micropatches[a_si as usize];
                    let b = &other.micropatches[b_si as usize];
                    a.union_inplace(b);
                    if a.is_all_ones() {
                        self.free_micropatches.push(a_si);
                        self.grid[idx] = MicropatchStatus::Full
                    }
                },
                (MicropatchStatus::Subdivided(a_si), MicropatchStatus::Full) =>  {
                    self.free_micropatches.push(a_si);
                    self.grid[idx] = MicropatchStatus::Full
                },
            }
        }
    }

    // Boolean intersection: self &= other
    pub fn intersection(&mut self, other: &Self) {
        // Ensure dimensions match
        assert_eq!(self.size, other.size, "VoxelSet dimensions must match for boolean operations");
        
        // Loop through grid cells
        for xi in 0..self.size_i.0 / MICROPATCH_SIZE as usize {
            for yi in 0..self.size_i.1 / MICROPATCH_SIZE as usize {
                for zi in 0..self.size_i.2 / MICROPATCH_SIZE as usize {
                    let idx = (zi * (self.size_i.1 / MICROPATCH_SIZE as usize) * (self.size_i.0 / MICROPATCH_SIZE as usize) + 
                              yi * (self.size_i.0 / MICROPATCH_SIZE as usize) + xi) as usize;
                    
                    // Fast path: if other grid cell is Full, nothing to do
                    // If other is Empty, result is Empty
                    match other.grid[idx] {
                        MicropatchStatus::Empty => {
                            // Result is Empty
                            match &self.grid[idx] {
                                MicropatchStatus::Subdivided(si) => {
                                    self.free_micropatches.push(*si);
                                },
                                _ => {}
                            }
                            self.grid[idx] = MicropatchStatus::Empty;
                        },
                        MicropatchStatus::Full => {
                            // No change needed to self
                            continue;
                        },
                        MicropatchStatus::Subdivided(other_si) => {
                            // Need to merge patches
                            match self.grid[idx] {
                                MicropatchStatus::Empty => {
                                    // Result is empty, nothing to do
                                },
                                MicropatchStatus::Full => {
                                    // Copy the entire micropatch from other
                                    let other_patch = &other.micropatches[other_si as usize];
                                    
                                    // If it's all ones, just keep Full
                                    if other_patch.is_all_ones() {
                                        continue;
                                    } else {
                                        // Otherwise create a new micropatch with the same bits
                                        if self.free_micropatches.len() > 0 {
                                            let mp_index = self.free_micropatches.pop().unwrap() as usize;
                                            self.grid[idx] = MicropatchStatus::Subdivided(mp_index as u32);
                                            self.micropatches[mp_index].bits.copy_from_slice(&other_patch.bits);
                                        } else {
                                            let new_patch = Micropatch { bits: other_patch.bits };
                                            self.micropatches.push(new_patch);
                                            let mp_index = (self.micropatches.len() - 1) as usize;
                                            self.grid[idx] = MicropatchStatus::Subdivided(mp_index as u32);
                                        }
                                    }
                                },
                                MicropatchStatus::Subdivided(self_si) => {
                                    // Need to AND the bits
                                    let self_patch = &mut self.micropatches[self_si as usize];
                                    let other_patch = &other.micropatches[other_si as usize];
                                    
                                    // AND the bits
                                    for i in 0..64 {
                                        self_patch.bits[i] &= other_patch.bits[i];
                                    }
                                    
                                    // Check if it's now all zeros
                                    if self_patch.is_all_zeros() {
                                        self.grid[idx] = MicropatchStatus::Empty;
                                        self.free_micropatches.push(self_si);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Boolean difference: self -= other
    pub fn difference(&mut self, other: &Self) {
        // Ensure dimensions match
        assert_eq!(self.size, other.size, "VoxelSet dimensions must match for boolean operations");
        
        // Loop through grid cells
        for xi in 0..self.size_i.0 / MICROPATCH_SIZE as usize {
            for yi in 0..self.size_i.1 / MICROPATCH_SIZE as usize {
                for zi in 0..self.size_i.2 / MICROPATCH_SIZE as usize {
                    let idx = (zi * (self.size_i.1 / MICROPATCH_SIZE as usize) * (self.size_i.0 / MICROPATCH_SIZE as usize) + 
                              yi * (self.size_i.0 / MICROPATCH_SIZE as usize) + xi) as usize;
                    
                    // Fast paths
                    // If other is Empty, no change
                    // If other is Full, result is Empty
                    match other.grid[idx] {
                        MicropatchStatus::Empty => {
                            // No change needed
                            continue;
                        },
                        MicropatchStatus::Full => {
                            // Result is Empty
                            match &self.grid[idx] {
                                MicropatchStatus::Subdivided(si) => {
                                    self.free_micropatches.push(*si);
                                },
                                _ => {}
                            }
                            self.grid[idx] = MicropatchStatus::Empty;
                        },
                        MicropatchStatus::Subdivided(other_si) => {
                            match self.grid[idx] {
                                MicropatchStatus::Empty => {
                                    // Result is empty, nothing to do
                                },
                                MicropatchStatus::Full => {
                                    // Need to create a new micropatch with NOT of other's bits
                                    let other_patch = &other.micropatches[other_si as usize];
                                    
                                    // Create a new micropatch with the inverted bits
                                    if self.free_micropatches.len() > 0 {
                                        let mp_index = self.free_micropatches.pop().unwrap() as usize;
                                        self.grid[idx] = MicropatchStatus::Subdivided(mp_index as u32);
                                        
                                        for i in 0..64 {
                                            self.micropatches[mp_index].bits[i] = !other_patch.bits[i];
                                        }
                                    } else {
                                        let mut new_patch = MICROPATCH_EMPTY;
                                        for i in 0..64 {
                                            new_patch.bits[i] = !other_patch.bits[i];
                                        }
                                        self.micropatches.push(new_patch);
                                        let mp_index = (self.micropatches.len() - 1) as usize;
                                        self.grid[idx] = MicropatchStatus::Subdivided(mp_index as u32);
                                    }
                                    
                                    // Check if all zeros
                                    if self.micropatches[match self.grid[idx] {
                                        MicropatchStatus::Subdivided(si) => si as usize,
                                        _ => panic!("Expected Subdivided")
                                    }].is_all_zeros() {
                                        let si = match &self.grid[idx] {
                                            MicropatchStatus::Subdivided(si) => *si,
                                            _ => panic!("Expected Subdivided")
                                        };
                                        self.grid[idx] = MicropatchStatus::Empty;
                                        self.free_micropatches.push(si);
                                    }
                                },
                                MicropatchStatus::Subdivided(self_si) => {
                                    // Need to AND with NOT of other's bits
                                    let self_patch = &mut self.micropatches[self_si as usize];
                                    let other_patch = &other.micropatches[other_si as usize];
                                    
                                    // self &= ~other
                                    for i in 0..64 {
                                        self_patch.bits[i] &= !other_patch.bits[i];
                                    }
                                    
                                    // Check if it's now all zeros
                                    if self_patch.is_all_zeros() {
                                        self.grid[idx] = MicropatchStatus::Empty;
                                        self.free_micropatches.push(self_si);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn invert_inplace(&mut self) {
        let n = (self.size_i.0 * self.size_i.1 * self.size_i.2) as usize;
        for i in 0..n {
            match self.grid[i] {
                MicropatchStatus::Empty => self.grid[i] = MicropatchStatus::Full,
                MicropatchStatus::Full => self.grid[i] = MicropatchStatus::Empty,
                MicropatchStatus::Subdivided(si) => self.micropatches[si as usize].invert_inplace()
            }
        }
    }

    // Flood fill starting from the given coordinates
    // Returns a new VoxelSet containing only the filled region
    pub fn flood_fill(&self, start_x: u32, start_y: u32, start_z: u32) -> Self {
        // Check if start point is in bounds
        if start_x >= self.size.0 || start_y >= self.size.1 || start_z >= self.size.2 {
            return Self::new(self.size.0, self.size.1, self.size.2);
        }
        
        // Check if start point is already filled
        let target_value = self.get(start_x, start_y, start_z);
        
        // Create a new VoxelSet for the result
        let mut result = Self::new(self.size.0, self.size.1, self.size.2);
        
        // Create a boolean array to track visited cells
        let mut visited = vec![false; (self.size.0 * self.size.1 * self.size.2) as usize];
        
        // Queue for BFS
        let mut queue = VecDeque::new();
        
        // Add the start point to queue
        queue.push_back((start_x, start_y, start_z));
        
        // Mark start point as visited
        let flat_index = (start_z * self.size.1 * self.size.0 + start_y * self.size.0 + start_x) as usize;
        visited[flat_index] = true;
        
        // BFS traversal
        while !queue.is_empty() {
            let (x, y, z) = queue.pop_front().unwrap();
            
            // Set this cell in the result
            result.set(x, y, z, true);
            
            // Check 6 neighbors (6-connected in 3D)
            let neighbors = [
                (x.checked_sub(1), Some(y), Some(z)),
                (Some(x + 1), Some(y), Some(z)),
                (Some(x), y.checked_sub(1), Some(z)),
                (Some(x), Some(y + 1), Some(z)),
                (Some(x), Some(y), z.checked_sub(1)),
                (Some(x), Some(y), Some(z + 1)),
            ];
            
            for (nx_opt, ny_opt, nz_opt) in neighbors {
                // Check if neighbor coordinates are valid
                if let (Some(nx), Some(ny), Some(nz)) = (nx_opt, ny_opt, nz_opt) {
                    // Check if neighbor is in bounds
                    if nx < self.size.0 && ny < self.size.1 && nz < self.size.2 {
                        // Calculate flat index for the neighbor
                        let neighbor_index = (nz * self.size.1 * self.size.0 + ny * self.size.0 + nx) as usize;
                        
                        // Check if neighbor is not visited and has the same value as the target
                        if !visited[neighbor_index] && self.get(nx, ny, nz) == target_value {
                            // Mark as visited
                            visited[neighbor_index] = true;
                            
                            // Add to queue
                            queue.push_back((nx, ny, nz));
                        }
                    }
                }
            }
        }
        
        result
    }
}

fn normalize_v3(v: Vec3) -> Vec3 {
    let norm = v.norm2().sqrt();
    if norm < 1e-7 {
        Vec3::new(1.0,0.0,0.0)
    } else {
        v / norm
    }
}

fn tri_height(base1: Vec3, base2: Vec3, opposite: Vec3) -> f32 {
    let b21 = base2 - base1;
    let bopp1 = opposite - base1;
    let normal = b21.cross(bopp1);
    let height_dir = normalize_v3(normal.cross(b21));
    let height = height_dir.dot(bopp1);
    height.abs()
}

fn argmax3(a: f32, b: f32, c: f32) -> u8 {
    if a < b {
        if b < c { 2 } else { 1 }
    } else {
        if a < c { 2 } else { 0 }
    }
}

fn max3(a: f32, b: f32, c: f32) -> f32 {
    if a < b {
        if b < c { c } else { b }
    } else {
        if a < c { c } else { a }
    }
}


impl VoxelSet {
    // Draw a line by interpolating between the endpoints,
    // using max(abs(a-b)) as the number of steps.
    pub fn line(self: &mut VoxelSet, a: Vec3, b: Vec3) {
        let ab = a - b;
        let ab_abs = Vec3::new(ab.x.abs(), ab.y.abs(), ab.z.abs());
        let max_delta = max3(ab_abs.x, ab_abs.y, ab_abs.z);
        let steps = (max_delta.floor() as u32) + 1;
        let step = ab / (steps as f32);
        println!("steps {} ({} -> {})", steps, a, b);
        for i in 0 ..= steps {
            let p = b + step * (i as f32);
            let x: i32 = p.x.round() as i32;
            let y: i32 = p.y.round() as i32;
            let z: i32 = p.z.round() as i32;
            if x >= 0 && y >= 0 && z >= 0 {
                self.set(x as u32, y as u32, z as u32, true);
            };
        }
    }
    pub fn fill_tri(self: &mut VoxelSet, a: Vec3, b: Vec3, c: Vec3) {
        let ab_h = tri_height(a,b,c);
        let bc_h = tri_height(c,a,b);
        let ca_h = tri_height(c,a,b);
        println!("fill_tri {} {} {} (height rel to ab={} bc={} ca={})",a,b,c,ab_h,bc_h,ca_h);
        let (origin, initial_delta, origin_change, height) = match argmax3(ab_h, bc_h, ca_h) {
            0 => (a, b-a, c-a, ab_h),
            1 => (b, c-b, a-b, bc_h),
            2 => (c, c-a, b-c, ca_h),
            _ => panic!("should not happen")
        };
        let n_steps = ((height.ceil() + 1.0) * 2.0) as u32;
        let n_steps_inv = 1.0 / (n_steps as f32);
        for i in 0 ..= n_steps {
            let frac = (i as f32) * n_steps_inv;
            let start = origin + origin_change * frac;
            let end = initial_delta * (1.0 - frac) + start;
            self.line(start, end);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micropatch_set_get() {
        let mut patch = MICROPATCH_EMPTY;
        assert_eq!(patch.set(3, 4, 5, true), false);
        assert_eq!(patch.get(3, 4, 5), true);
        assert_eq!(patch.set(3, 4, 5, false), true);
        assert_eq!(patch.get(3, 4, 5), false);
    }

    #[test]
    fn test_micropatch_all_zeros() {
        let mut patch = MICROPATCH_EMPTY;
        assert!(patch.is_all_zeros());
        
        patch.set(3, 4, 5, true);
        assert!(!patch.is_all_zeros());
        
        patch.set(3, 4, 5, false);
        assert!(patch.is_all_zeros());
    }

    #[test]
    fn test_micropatch_all_ones() {
        let mut patch = MICROPATCH_FULL;
        assert!(patch.is_all_ones());
        
        patch.set(3, 4, 5, false);
        assert!(!patch.is_all_ones());
        
        patch.set(3, 4, 5, true);
        assert!(patch.is_all_ones());
    }

    #[test]
    fn test_micropatch_index() {
        assert_eq!(micropatch_index(0), (0, 0));
        assert_eq!(micropatch_index(7), (0, 7));
        assert_eq!(micropatch_index(8), (1, 0));
        assert_eq!(micropatch_index(15), (1, 7));
        assert_eq!(micropatch_index(16), (2, 0));
    }

    #[test]
    fn test_round_up() {
        assert_eq!(round_up(0), 0);
        assert_eq!(round_up(1), 8);
        assert_eq!(round_up(8), 8);
        assert_eq!(round_up(9), 16);
        assert_eq!(round_up(15), 16);
        assert_eq!(round_up(16), 16);
    }

    #[test]
    fn test_voxel_set_new() {
        let vs = VoxelSet::new(10, 12, 15);
        assert_eq!(vs.size, (10, 12, 15));
        assert_eq!(vs.size_i, (16, 16, 16));
        assert_eq!(vs.grid.len(), 16 * 16 * 16);
        assert_eq!(vs.micropatches.len(), 0);
        assert_eq!(vs.free_micropatches.len(), 0);
    }

    #[test]
    fn test_voxel_set_get() {
        let vs = VoxelSet::new(10, 12, 15);
        
        // Out of bounds coordinates
        assert_eq!(vs.get(11, 5, 5), false);
        assert_eq!(vs.get(5, 13, 5), false);
        assert_eq!(vs.get(5, 5, 16), false);
        
        // In bounds but empty
        assert_eq!(vs.get(5, 5, 5), false);
    }

    #[test]
    fn test_voxel_set_set_get() {
        let mut vs = VoxelSet::new(10, 12, 15);
        
        // Set and get a cell
        assert_eq!(vs.set(5, 6, 7, true), false); // Returns previous value
        assert_eq!(vs.get(5, 6, 7), true);
        
        // Set to false and check
        assert_eq!(vs.set(5, 6, 7, false), true); // Returns previous value
        assert_eq!(vs.get(5, 6, 7), false);
        
        // Out of bounds setting should return false
        assert_eq!(vs.set(20, 6, 7, true), false);
    }

    #[test]
    fn test_force_micropatch() {
        let mut vs = VoxelSet::new(8, 8, 8);
        
        // Force first micropatch from Empty
        let idx = 0;
        let patch = vs._force_micropatch(idx);
        assert_eq!(*patch, MICROPATCH_EMPTY);
        assert_eq!(vs.micropatches.len(), 1);
        
        match vs.grid[idx] {
            MicropatchStatus::Subdivided(mp_idx) => {
                assert_eq!(mp_idx, 0);
            },
            _ => panic!("Expected Subdivided status")
        }
        
        // Set grid cell to Full and force micropatch
        vs.grid[1] = MicropatchStatus::Full;
        let patch = vs._force_micropatch(1);
        
        // Set and verify micropatch is changed
        patch.set(1, 1, 1, false);
        assert_eq!(patch.get(1, 1, 1), false);

        assert_eq!(vs.micropatches.len(), 2);
    }

    #[test]
    fn test_free_micropatch_reuse() {
        let mut vs = VoxelSet::new(16, 8, 8);
        
        // Force micropatch and fill it
        vs.set(0, 0, 0, true);
        let idx = 0;
        
        // Make it all ones
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    vs.set(x, y, z, true);
                }
            }
        }
        
        // Now check that it became Full
        match vs.grid[idx] {
            MicropatchStatus::Full => {},
            _ => panic!("Expected Full status")
        }
        
        // Force another micropatch and check free micropatch is reused
        let free_count = vs.free_micropatches.len();
        assert!(free_count > 0);
        
        // Add one cell outside first micropatch
        vs.set(8, 0, 0, true);
        
        // Ensure we used a free micropatch
        assert_eq!(vs.free_micropatches.len(), free_count - 1);
    }

    #[test]
    fn test_normalize_v3() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = normalize_v3(v);
        assert!((n.norm2() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!((n.y - 0.8).abs() < 1e-6);
        
        // Test very small vector
        let v_small = Vec3::new(1e-10, 0.0, 0.0);
        let n_small = normalize_v3(v_small);
        assert_eq!(n_small.x, 1.0);
        assert_eq!(n_small.y, 0.0);
        assert_eq!(n_small.z, 0.0);
    }

    #[test]
    fn test_tri_height() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 0.0, 0.0);
        let c = Vec3::new(5.0, 5.0, 0.0);
        
        let height = tri_height(a, b, c);
        assert!((height - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax3() {
        assert_eq!(argmax3(1.0, 2.0, 3.0), 2);
        assert_eq!(argmax3(3.0, 2.0, 1.0), 0);
        assert_eq!(argmax3(1.0, 3.0, 2.0), 1);
        assert_eq!(argmax3(3.0, 3.0, 1.0), 0); // First one wins in tie
    }

    #[test]
    fn test_max3() {
        assert!((max3(1.0, 2.0, 3.0) - 3.0).abs() < 1e-6);
        assert!((max3(3.0, 2.0, 1.0) - 3.0).abs() < 1e-6);
        assert!((max3(1.0, 3.0, 2.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_line() {
        let mut vs = VoxelSet::new(10, 10, 10);
        
        // Draw horizontal line from (1,1,1) to (5,1,1)
        vs.line(Vec3::new(1.0, 1.0, 1.0), Vec3::new(5.0, 1.0, 1.0));
        
        // Check points along the line
        assert!(vs.get(1, 1, 1));
        assert!(vs.get(2, 1, 1));
        assert!(vs.get(3, 1, 1));
        assert!(vs.get(4, 1, 1));
        assert!(vs.get(5, 1, 1));
        
        // Check off-line points
        assert!(!vs.get(3, 2, 1));
        assert!(!vs.get(3, 1, 2));
    }

    #[test]
    fn test_fill_tri() {
        let mut vs = VoxelSet::new(10, 10, 10);
        
        // Create a simple triangle in the xy plane
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(5.0, 1.0, 1.0);
        let c = Vec3::new(3.0, 5.0, 1.0);
        
        vs.fill_tri(a, b, c);
        
        // Check some points that should be in the triangle
        assert!(vs.get(3, 3, 1));
        assert!(vs.get(2, 2, 1));
        assert!(vs.get(4, 2, 1));
        
        // Check a point outside the triangle
        assert!(!vs.get(1, 5, 1));
        assert!(!vs.get(5, 5, 1));
    }

    #[test]
    fn test_voxel_set_clone() {
        let mut original = VoxelSet::new(10, 10, 10);
        original.set(3, 4, 5, true);
        original.set(6, 7, 8, true);
        
        let cloned = original.clone();
        
        // Check that the clone has the same values
        assert_eq!(cloned.get(3, 4, 5), true);
        assert_eq!(cloned.get(6, 7, 8), true);
        assert_eq!(cloned.get(1, 1, 1), false);
        
        // Check that the clone has independent data
        assert_eq!(cloned.size, original.size);
        assert_eq!(cloned.size_i, original.size_i);
        assert_eq!(cloned.micropatches.len(), original.micropatches.len());
    }

    #[test]
    fn test_union() {
        let mut vs1 = VoxelSet::new(10, 10, 10);
        let mut vs2 = VoxelSet::new(10, 10, 10);
        
        // Set different cells in each set
        vs1.set(1, 2, 3, true);
        vs1.set(4, 5, 6, true);
        vs2.set(4, 5, 6, true);  // Overlapping cell
        vs2.set(7, 8, 9, true);
        
        // Apply union
        vs1.union(&vs2);
        
        // Check that all cells are set
        assert!(vs1.get(1, 2, 3));
        assert!(vs1.get(4, 5, 6));
        assert!(vs1.get(7, 8, 9));
        assert!(!vs1.get(0, 0, 0));
    }

    #[test]
    fn test_intersection() {
        let mut vs1 = VoxelSet::new(10, 10, 10);
        let mut vs2 = VoxelSet::new(10, 10, 10);
        
        // Set cells in each set with one overlapping
        vs1.set(1, 2, 3, true);
        vs1.set(4, 5, 6, true);
        vs2.set(4, 5, 6, true);  // Only overlapping cell
        vs2.set(7, 8, 9, true);
        
        // Apply intersection
        vs1.intersection(&vs2);
        
        // Check that only the overlapping cell is set
        assert!(!vs1.get(1, 2, 3));
        assert!(vs1.get(4, 5, 6));
        assert!(!vs1.get(7, 8, 9));
    }

    #[test]
    fn test_difference() {
        let mut vs1 = VoxelSet::new(10, 10, 10);
        let mut vs2 = VoxelSet::new(10, 10, 10);
        
        // Set cells in each set with some overlapping
        vs1.set(1, 2, 3, true);
        vs1.set(4, 5, 6, true);
        vs2.set(4, 5, 6, true);
        vs2.set(7, 8, 9, true);
        
        // Apply difference (vs1 - vs2)
        vs1.difference(&vs2);
        
        // Check that only non-overlapping cells from vs1 remain
        assert!(vs1.get(1, 2, 3));
        assert!(!vs1.get(4, 5, 6));
        assert!(!vs1.get(7, 8, 9));
    }

    #[test]
    fn test_flood_fill() {
        let mut vs = VoxelSet::new(10, 10, 10);
        
        // Create a hollow cube
        for x in 1..6 {
            for y in 1..6 {
                for z in 1..6 {
                    if x == 1 || x == 5 || y == 1 || y == 5 || z == 1 || z == 5 {
                        vs.set(x, y, z, true);
                    }
                }
            }
        }
        
        // Make a door in the cube
        vs.set(1, 3, 3, false);
        
        // Flood fill from outside the cube
        let filled = vs.flood_fill(0, 0, 0);
        
        // Check that outside is filled
        assert!(filled.get(0, 0, 0));
        assert!(filled.get(1, 3, 3)); // The door
        
        // Check that inside the cube is filled (through the door)
        assert!(filled.get(2, 3, 3));
        assert!(filled.get(3, 3, 3));
        
        // Check that the walls are not filled
        assert!(!filled.get(1, 1, 1));
        assert!(!filled.get(5, 5, 5));
    }
}
