#include "cuda_helper.cuh"
#include "constant.h"
#include "gl_helper.h"
#include "helper.h"

extern "C"
void launch_init_curand(dim3 grid, dim3 block);
extern "C"
void launch_free_device();
extern "C"
void launch_init_scene();
extern "C"
void launch_render(dim3 grid, dim3 block, float iter, float* dev_tex, interaction& param);
extern "C"
void launch_clear_buf();

// host
float iter;

// cuda
dim3 grid, block;

// opengl
GLuint vao, vbo, ebo;
glshader shader;

// opengl <-> cuda
float *dev_tex;
cudaGraphicsResource *cuda_tex_resrc;
GLuint gl_tex_id;

interaction param;

void init_gl()
{
	// CUDA 9
	//// init cuda
	//cudaDeviceProp prop;
	//int dev;
	//memset(&prop, 0, sizeof(cudaDeviceProp));
	//prop.major = 2;
	//prop.minor = 0;
	//HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	//HANDLE_ERROR(cudaGLSetGLDevice(dev));

	// init opengl
	int argc = 1;
	char *argv = (char*)"";
	glutInit(&argc, &argv);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitContextVersion(4, 2);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow("Viewer");

	// init glew
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << glewGetErrorString(err) << "\n";
	}
	if (glewIsSupported("GL_VERSION_4_2")) {
		std::cerr << "Ready for OpenGL 4.2\n";
	}
	else {
		std::cerr << "OpenGL 4.2 not supported\n";
		exit(1);
	}

	// CUDA 10
	// https://github.com/OpenGLInsights/OpenGLInsightsCode/blob/master/Chapter%2009%20Mixing%20Graphics%20and%20Compute%20with%20Multiple%20GPUs/main.cu#L579
	unsigned int cudaDeviceCount;
	int cudaDevices[1];
	HANDLE_ERROR(cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 1, cudaGLDeviceListAll));

	// init shader
	shader = glshader("shader.vs.glsl", "shader.fs.glsl");

	// init buffer
	GLfloat vertices[] = {
		 1.0f,  1.0f,
		 1.0f, -1.0f,
		-1.0f, -1.0f,
		-1.0f,  1.0f
	};
	GLuint indices[] = {
		1, 2, 3,
		0, 1, 3
	};

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// init texture
	glGenTextures(1, &gl_tex_id);
	glBindTexture(GL_TEXTURE_2D, gl_tex_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);

	shader.use();
	shader.setInt("ydim", HEIGHT);
}

void init_cuda()
{
	block = dim3(TILE, TILE);
	grid = dim3((WIDTH + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);

	launch_init_curand(grid, block);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	launch_init_scene();
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	// opengl <-> cuda
	HANDLE_ERROR(cudaGraphicsGLRegisterImage(&cuda_tex_resrc, gl_tex_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	HANDLE_ERROR(cudaMalloc((void**)&dev_tex, WIDTH * HEIGHT * 4 * sizeof(float)));
}

inline void run_cuda()
{
	if (iter > MAX_SAMPLES) {
		return;
	}
	else {
		iter += 1.f;
	}

	if (param.is_mouse || param.is_keyboard) {
		iter = 1.f;
		launch_clear_buf();
	}

	launch_render(grid, block, iter, dev_tex, param);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	// cuda <-> opengl
	cudaArray *mapped_tex_ptr;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_tex_resrc, 0));
	HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&mapped_tex_ptr, cuda_tex_resrc, 0, 0));
	// CUDA 9
	//HANDLE_ERROR(cudaMemcpyToArray(mapped_tex_ptr, 0, 0, dev_tex, WIDTH * HEIGHT * 4 * sizeof(float), cudaMemcpyDeviceToDevice));
	// CUDA 10
	// https://stackoverflow.com/questions/70268991/whats-pitch-in-cudamemcpy2dtoarray-and-cudamemcpy2dfromarray
	HANDLE_ERROR(cudaMemcpy2DToArray(mapped_tex_ptr, 0, 0, dev_tex, WIDTH * 4 * sizeof(float), WIDTH * 4 * sizeof(float), HEIGHT, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_tex_resrc, 0));
}

inline void run_opengl()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	shader.use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gl_tex_id);
	shader.setInt("tex", 0);
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int x, int y)
{
	vec3 forward = unit_vector(param.lookdir);
	vec3 right = unit_vector(cross(forward, vec3(0, 1, 0)));
	vec3 up = cross(forward, right);

	switch (key) {
	case GLUT_KEY_ESC:
		exit(0);
		break;
	case 'w':
		param.lookfrom += forward * param.KEYBOARD_SPEED;
		param.is_keyboard = true;
		break;
	case 's':
		param.lookfrom -= forward * param.KEYBOARD_SPEED;
		param.is_keyboard = true;
		break;
	case 'd':
		param.lookfrom += right * param.KEYBOARD_SPEED;
		param.is_keyboard = true;
		break;
	case 'a':
		param.lookfrom -= right * param.KEYBOARD_SPEED;
		param.is_keyboard = true;
		break;
	}
}
void onKeyboardUp(unsigned char key, int x, int y)
{
	param.is_keyboard = false;
}
void onMouse(int who, int state, int x, int y)
{
	switch (who) {
	case GLUT_LEFT_BUTTON:
		param.curr_x = x;
		param.curr_y = y;
		break;
	}

	switch (state) {
	case GLUT_DOWN:
		param.is_mouse = true;
		break;
	case GLUT_UP:
		param.is_mouse = false;
		break;
	}
}
void onMouseMotion(int x, int y)
{
	int diff_x = x - param.curr_x;
	int diff_y = y - param.curr_y;

	float offset_x = float(-diff_x) * param.MOUSE_SPEED;
	float offset_y = float(diff_y) * param.MOUSE_SPEED;

	param.curr_yaw += offset_x;
	param.curr_pitch += offset_y;

	if (param.curr_pitch > 89.f) {
		param.curr_pitch = 89.f;
	}
	if (param.curr_pitch < -89.f) {
		param.curr_pitch = -89.f;
	}

	param.lookdir = vec3(cosf(param.curr_yaw * param.DEG2RAD) * cosf(param.curr_pitch * param.DEG2RAD),
						 sinf(param.curr_pitch * param.DEG2RAD),
						 sinf(param.curr_yaw * param.DEG2RAD) * cosf(param.curr_pitch * param.DEG2RAD));
	param.lookdir.make_unit_vector();

	param.curr_x = x;
	param.curr_y = y;
}
void onIdle()
{
	glutPostRedisplay();
}
void onDisplay(void)
{
	run_cuda();
	run_opengl();
	printf("\rIter %5d\r", int(iter));
}

int main()
{
	init_gl();
	init_cuda();

	print_gpu_info();

	iter = 0.f;
	printf("\n");

	glViewport(0, 0, WIDTH, HEIGHT);
	glutKeyboardFunc(onKeyboard);
	glutIdleFunc(onIdle);
	glutDisplayFunc(onDisplay);
	glutMouseFunc(onMouse);
	glutMotionFunc(onMouseMotion);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMainLoop();

	launch_free_device();
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	return 0;
}

