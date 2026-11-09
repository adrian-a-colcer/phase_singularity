
---------- TIP from wgpu docs -------------
A lot of data used in GPU programming comes in 2D and even 3D arrays. 
Because of this workgroup size using 3 dimensions instead of 1 to make 
writing multidimensional code more convenient.

For example, a blur on a 2D image would benefit from a 2D work group so 
each thread would match up to a pixel in the image. A marching cubes 
implementation would benefit from a 3D workgroup, so each thread handles 
the geometry for one voxel in the voxel grid.
