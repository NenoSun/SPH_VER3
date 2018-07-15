#ifndef CONFIGURATION
#define CONFIGURATION

//#define LINUX
#define WINDOWS

//#define VERTICES_TRACK
#define RENDER_PARTICLES
//#define RENDER_MESH

//#define ENABLE_FRAME_CAPTURE

//#define OUTPUT_PARTICLE_OBJECT
//#define OUTPUT_MESH_OBJECT

//#define SPLINE_KERNEL
//#define KERNEL

//#define PARTICLE_VELOCITY_RAMP

#define DF
//#define RENDER_BOUNDARY_PARTICLES
//#define RENDER_OBJECT_PARTICLE
//#define ENABLE_BOUNDARY_PARTICLE

#define DIVERGENCE_SOLVER
#define DENSITY_SOLVER  

#define VISUAL_OPTIMIZATION


#define DOWNX .4f  
#define UPX .6f  
#define DOWNY .4f
#define UPY .6f
#define DOWNZ .4f
#define UPZ .6f

//#define DOWNX .04f
//#define UPX .20f
//#define DOWNY .02f
//#define UPY .3f
//#define DOWNZ .02f
//#define UPZ .98f

//#define WORLDSIZE_X 10.0f
//#define WORLDSIZE_Y 4.0f
//#define WORLDSIZE_Z 2.0f

#define WORLDSIZE_X 0.64f
#define WORLDSIZE_Y 0.64f
#define WORLDSIZE_Z 0.64f

#define INTERVAL .4f
//#define INTERVAL 2.0f
#define BOUNDARY_PARTICLE_INTERVAL .2f

#define MESH_RESOLUTION 50
#define IOSVALUE 0.4f

#define GRAVITY -9.8f


// TIMER CONTROL PANEL
//#define TIMER
//#define FRAME_TIMER
//#define DENSITY_SOLVER_TIMER
//#define DIVERGENCE_SOLVER_TIMER
//#define NEIGHBOR_SEARCHING_TIMER
//
//#define FRAME_TIMEDATA_FILEPATH "D:/frameTime.txt"
//#define DENSITYSOVLER_TIMEDATA_FILEAPTH "D:/DensitySolverTime.txt"
//#define DIVERGENCESOLVER_TIMEDATE_FILEPATH "D:/DivergenceSolverTime.txt"
//#define NEIGHBOR_SEARCHING_TIMEDATA_FILEPATH "D:/NeightborSearchingTime.txt"
// TIMER CONTROL PANEL

#define ITERATE_NEIGHBOR for(int x = -1; x <= 1; x++)\
for(int y = -1; y <= 1; y++)\
for(int z = -1; z <= 1; z++)

#endif // CONFIGURATION

// YOU NEED TO PRESS SPACE TO START THE SIMULATION

