#ifndef CONFIGURATION
#define CONFIGURATION

/* System */
//#define LINUX
#define WINDOWS


/* Particle model or surface model */
#define RENDER_PARTICLES
//#define RENDER_MESH
//#define RENDER_BOUNDARY_PARTICLES
//#define RENDER_OBJECT_PARTICLE
//#define PARTICLE_VELOCITY_RAMP


/* Output .obj file? */
//#define OUTPUT_PARTICLE_OBJECT
//#define OUTPUT_MESH_OBJECT

/* Allow frame capture? */
//#define ENABLE_FRAME_CAPTURE


/* Physical engine working principle */
//#define SPLINE_KERNEL
//#define KERNEL
#define DF
//#define CPU_DF
//#define DIVERGENCE_SOLVER
//#define DENSITY_SOLVER


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

#define WORLDSIZE_X 1.5f
#define WORLDSIZE_Y 1.5f
#define WORLDSIZE_Z 1.5f

//#define INTERVAL .49f
#define INTERVAL .5f
#define BOUNDARY_PARTICLE_INTERVAL .2f


/* Marching cube controller */
#define MESH_RESOLUTION 50
#define IOSVALUE 0.4f


/* Force controller */
#define GRAVITY -9.8f
//#define GRAVITY 0.0f
#define SURFACE_TENSION
#define SURFACE_TENSION_COEFFICIENT 1.0
#define VISCOUS_FORCE
#define VISCOUS_FROCE_COEFFICIENT 1.0 
#define PRESSURE_FORCE
#define PRESSURE_FORCE_COEFFICIENT 1.0

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


/* Additional options */
//#define VERTICES_TRACK
#define ENABLE_BOUNDARY_PARTICLE
#define VISUAL_OPTIMIZATION

