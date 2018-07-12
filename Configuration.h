#ifndef CONFIGURATION
#define CONFIGURATION

#define LINUX
//#define WINDOWS

//#define VERTICES_TRACK
//#define RENDER_PARTICLES
#define RENDER_MESH

//#define ENABLE_FRAME_CAPTURE

//#define OUTPUT_PARTICLE_OBJECT
#define OUTPUT_MESH_OBJECT

//#define SPLINE_KERNEL
//#define KERNEL

#define DF
//#define RENDER_BOUNDARY_PARTICLES
#define RENDER_OBJECT_PARTICLE
#define ENABLE_BOUNDARY_PARTICLE

#define DIVERGENCE_SOLVER
#define DENSITY_SOLVER  

#define VISUAL_OPTIMIZATION

//#define TIMER

//#define DOWNX .02f  
//#define UPX .98f  
//#define DOWNY .02f
//#define UPY .8f
//#define DOWNZ .02f
//#define UPZ .2f

#define DOWNX .02f
#define UPX .20f
#define DOWNY .02f
#define UPY .8f
#define DOWNZ .02f
#define UPZ .98f

#define WORLDSIZE_X 10.0f
#define WORLDSIZE_Y 4.0f
#define WORLDSIZE_Z 2.0f

//#define WORLDSIZE_X 2.0f
//#define WORLDSIZE_Y 2.0f
//#define WORLDSIZE_Z 2.0f

#define INTERVAL .5f
#define BOUNDARY_PARTICLE_INTERVAL .2f

#define MESH_RESOLUTION 50
#define IOSVALUE 0.4f

#define GRAVITY -9.8f

#endif // CONFIGURATION

// YOU NEED TO PRESS SPACE TO START THE SIMULATION

