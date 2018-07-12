#ifndef _SPH_MAIN_CPP
#define _SPH_MAIN_CPP
#include "Configuration.h"
#include "sph_data.h"
#include "sph_timer.h"
#include "SPHSystem.h"
#include "Snapshot.h"
#include "Type.cuh"
#include "OBJGenerator.h"
#include <fstream>
#include <time.h>

#ifdef WINDOWS
#include <GL/glew.h>
#include <GL/glut.h>
#pragma comment(lib, "glew32.lib") 
#pragma comment(lib, "glut64.lib")
#endif


#ifdef LINUX
#include </usr/include/GL/glew.h>
#include </usr/include/GL/glut.h>
#include <cstring>
//Linux frame caputure
#include<cstdlib>
#include<cstdio>
#endif


SPHSystem *sph;

#ifdef TIMER
Timer *sph_timer;
#endif

char *window_title;

GLuint v;
GLuint f;
GLuint p;

OBJGenerator og("/home/neno/OBJ/");

void set_shaders()
{
	char *vs = NULL;
	char *fs = NULL;

	vs = (char *)malloc(sizeof(char) * 10000);
	fs = (char *)malloc(sizeof(char) * 10000);
	memset(vs, 0, sizeof(char) * 10000);
	memset(fs, 0, sizeof(char) * 10000);

	FILE *fp;
	char c;
	int count;

	fp = fopen("shader/shader.vs", "r");
	count = 0;
	while ((c = fgetc(fp)) != EOF)
	{
		vs[count] = c;
		count++;
	}
	fclose(fp);

	fp = fopen("shader/shader.fs", "r");
	count = 0;
	while ((c = fgetc(fp)) != EOF)
	{
		fs[count] = c;
		count++;
	}
	fclose(fp);

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	const char *vv;
	const char *ff;
	vv = vs;
	ff = fs;
	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	int success;

	glCompileShader(v);
	glGetShaderiv(v, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		char info_log[5000];
		glGetShaderInfoLog(v, 5000, NULL, info_log);
		printf("Error in vertex shader compilation!\n");
		printf("Info Log: %s\n", info_log);
	}

	glCompileShader(f);
	glGetShaderiv(f, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		char info_log[5000];
		glGetShaderInfoLog(f, 5000, NULL, info_log);
		printf("Error in fragment shader compilation!\n");
		printf("Info Log: %s\n", info_log);
	}

	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);
	glLinkProgram(p);
	glUseProgram(p);

	free(vs);
	free(fs);
}

void draw_box(float ox, float oy, float oz, float width, float height, float length)
{
	glLineWidth(1.0f);
	glColor3f(1.0f, 0.0f, 0.0f);

	glBegin(GL_LINES);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox + width, oy, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy + height, oz);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox + width, oy + height, oz + length);

	glVertex3f(ox + width, oy + height, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox + width, oy + height, oz + length);

	glEnd();
}

void init_sph_system()
{
	//real_world_side.x = 10.0f;
	//real_world_side.y = 10.0f;
	//real_world_side.z = 40.0f;

	//real_world_origin.x = -10.0f;
	//real_world_origin.y = -10.0f;
	//real_world_origin.z = -10.0f;

	real_world_side.x = WORLDSIZE_X * 5.0f;
	real_world_side.y = WORLDSIZE_Y * 5.0f;
	real_world_side.z = WORLDSIZE_Z * 5.0f;

	real_world_origin.x = -(real_world_side.x / 2.0f);
	real_world_origin.y = -(real_world_side.y / 2.0f);
	real_world_origin.z = -(real_world_side.z / 2.0f);

	sph = new SPHSystem();
	sph->generateParticles();

#ifdef TIMER
	sph_timer = new Timer();
#endif
	window_title = (char *)malloc(sizeof(char) * 50);
}

void init()
{
	glewInit();

	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)window_width / window_height, 10.0f, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SMOOTH);
}

void init_ratio()
{
	sim_ratio.x = real_world_side.x / sph->worldSize.x; 
	sim_ratio.y = real_world_side.y / sph->worldSize.y; 
	sim_ratio.z = real_world_side.z / sph->worldSize.z; 
	sph->sim_ratio.x = sim_ratio.x;
	sph->sim_ratio.y = sim_ratio.y;
	sph->sim_ratio.z = sim_ratio.z;
}


void render_particles()
{
	glPointSize(4.0f);
	glColor3f(0.2f, 0.2f, 1.0f);
	std::cout << sph->sys_running << std::endl;

	//float avg = 0.0f;
	//for(uint i = 0; i < sph->num_particles; i++){
	//	avg += sph->hParticles[i].vel_scalar;
	//}
	//avg = avg / sph->num_particles;

	//float standardDeviation = 0.0f;
	//for(int i = 0;i < sph->num_particles; i++){
	//	standardDeviation += (sph->hParticles[i].vel_scalar - avg) * (sph->hParticles[i].vel_scalar - avg);
	//}
	//standardDeviation = sqrt(standardDeviation / sph->num_particles);

	//for(int i = 0;i < sph->num_particles; i++){
	//	sph->hParticles[i].vel_scalar = (sph->hParticles[i].vel_scalar - avg)/standardDeviation;
	//}

	float minVel = 100000;
	float maxVel = -11;

	for (uint i = 0; i<sph->num_particles; i++)
	{
		float vel = sph->hParticles[i].vel_scalar;
		if(vel > maxVel)
			maxVel = vel;
		else if(vel < minVel)
			minVel = vel;
	}

	float vel_diff = maxVel - minVel;


	for (uint i = 0; i<sph->num_particles; i++)
	{
		if(sph->sys_running == 1)
#ifdef OUTPUT_PARTICLE_OBJ
			og.writeVertice(sph->hParticles[i].pos.x / WORLDSIZE_X - WORLDSIZE_X / 2.0,
		  		sph->hParticles[i].pos.y / WORLDSIZE_Y - WORLDSIZE_Y / 2.0,
		  		sph->hParticles[i].pos.z / WORLDSIZE_Z - WORLDSIZE_Z / 2.0);
#endif				  
		glColor3f((sph->hParticles[i].vel_scalar - minVel)/3.0, (sph->hParticles[i].vel_scalar - minVel)/3.0, 0.9f);
		glBegin(GL_POINTS);
		glVertex3f(sph->hParticles[i].pos.x*sim_ratio.x + real_world_origin.x,
			sph->hParticles[i].pos.y*sim_ratio.y + real_world_origin.y,
			sph->hParticles[i].pos.z*sim_ratio.z + real_world_origin.z);
		glEnd();
	}

#ifdef OUTPUT_PARTICLE_OBJ
	if(sph->sys_running == 1){
		og.finishThisObject();
	}
#endif;

#ifdef RENDER_BOUNDARY_PARTICLES
	for (uint i = 0; i<sph->num_boundary_p; i++)
	{
		glBegin(GL_POINTS);
		glVertex3f(sph->hBoundaryParticles[i].pos.x*sim_ratio.x + real_world_origin.x,
			sph->hBoundaryParticles[i].pos.y*sim_ratio.y + real_world_origin.y,
			sph->hBoundaryParticles[i].pos.z*sim_ratio.z + real_world_origin.z);
		glEnd();
	}
#endif

#ifdef RENDER_OBJECT_PARTICLE
	glPointSize(2.0f);
	glColor3f(1.0f, 0.2f, 0.2f);
	for (uint i = 0; i<sph->num_boundary_p; i++)
	{
		if(sph->hBoundaryParticles[i].isObject)
		glBegin(GL_POINTS);
		glVertex3f(sph->hBoundaryParticles[i].pos.x*sim_ratio.x + real_world_origin.x,
			sph->hBoundaryParticles[i].pos.y*sim_ratio.y + real_world_origin.y,
			sph->hBoundaryParticles[i].pos.z*sim_ratio.z + real_world_origin.z);
		glEnd();
	}
#endif

}


void drawTriangles(Float3* triangles,  Float3* hNorms, uint cube_num) {
	glColor3b(1.0f, 0.2f, 0.2f);
	long count = 0;

	for (int i = 0; i < 15 * cube_num; i = i + 3) {
		if (triangles[i].x == -999.0 || triangles[i].y == -999.0 || triangles[i].z == -999.0)
			continue;

		glBegin(GL_TRIANGLES);
		for (int j = 0; j < 3; j++) {
			glNormal3f(hNorms[i + j].x, hNorms[i + j].y, hNorms[i + j].z);
			glVertex3f(triangles[i + j].x*sph->sim_ratio.x + real_world_origin.x,
				triangles[i + j].y*sph->sim_ratio.y + real_world_origin.y,
				triangles[i + j].z*sph->sim_ratio.z + real_world_origin.z);
#ifdef OUTPUT_MESH_OBJECT
			if(sph->sys_running == 1){
				og.writeVertice(triangles[i + j].x, triangles[i + j].y, triangles[i + j].z);
			}
#endif
		}
		glEnd();
		count++;
	}

#ifdef OUTPUT_MESH_OBJECT
	for(int i = 0; i < count; i++){
		og.writeTriangle(i);
	}

	if(sph->sys_running == 1){
		og.finishThisObject();
	}
#endif
}


float light_ambient[] = { .0f, .0f, 1.0f, 1.0f };
float light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
float light_position[] = { -10.0f, -10.0f, -10.0f, 1.0f };

int cou = 0;
void display_func()
{
	glClearColor(0.75f, 0.75f, 0.75f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glPushMatrix();
	
	if (buttonState == 1)
	{
		xRot += (xRotLength - xRot)*0.1f;
		yRot += (yRotLength - yRot)*0.1f;
	}
	
	glTranslatef(xTrans, yTrans, zTrans);	
	glRotatef(xRot, 1.0f, 0.0f, 0.0f);
	glRotatef(yRot, 0.0f, 1.0f, 0.0f);

	// THE GAME IS ON!
	sph->animation();

#ifdef RENDER_MESH
	glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, light_position);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHTING);
	drawTriangles(sph->hTriangles, sph->hNorms, sph->cube_num);
#endif

#ifdef RENDER_PARTICLES
	glUseProgram(p);
	render_particles();
#endif


#ifdef RENDER_MESH
	glDisable(GL_LIGHTING);
#endif

	glUseProgram(0);
	glColor3f(1.0f, 0.0f, 0.0f);
	draw_box(real_world_origin.x, real_world_origin.y, real_world_origin.z, real_world_side.x, real_world_side.y, real_world_side.z);
	

 	glPopMatrix();
	
	glutSwapBuffers();
	
#ifdef TIMER
	sph_timer->update();
	memset(window_title, 0, 50);
	sprintf(window_title, "SPH System 3D. FPS: %f", sph_timer->get_fps());
	glutSetWindowTitle(window_title);
#endif

#ifdef ENABLE_FRAME_CAPTURE
#ifdef WINDOWS
	std::string k = "Screena.jpg";
	Snapshot::gdiscreen();
#endif


#ifdef LINUX
	char buffer[100];
	sprintf(buffer,"gnome-screenshot -f ./frame/%d.jpg",cou);
	std::system(buffer);
#endif	
#endif
	//printf("%d\n", cou);
	cou++;
}

void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
	window_width = width;
	window_height = height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)width / height, 0.001, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
	if (key == ' ')
	{
		sph->sys_running = 1 - sph->sys_running;
	}

	if (key == 'w')
	{
		zTrans += 0.3f;
	}

	if (key == 's')
	{
		zTrans -= 0.3f;
	}

	if (key == 'a')
	{
		xTrans -= 0.3f;
	}

	if (key == 'd')
	{
		xTrans += 0.3f;
	}

	if (key == 'q')
	{
		yTrans -= 0.3f;
	}

	if (key == 'e')
	{
		yTrans += 0.3f;
	}

	glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState = 1;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

void motion_func(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 1)
	{
		xRotLength += dy / 5.0f;
		yRotLength += dx / 5.0f;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

int main(int argc, char **argv)
{
	// Initialize displaying window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("SPH Fluid 3D GPU");


	init_sph_system();
	init();
	init_ratio();


#ifdef RENDER_PARTICLES
	set_shaders();
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
#endif

	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);

	glutMainLoop();

	return 0;
}


#endif
