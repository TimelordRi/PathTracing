#include <windows.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "bmp.h"
#include <time.h>

#include <iostream>
#include "Struct.h"
#include "Model.h"

// INVIDIA Discrete GPU
extern "C" {
    _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void ScreenShot(int width, int height, string file);
unsigned int load_img(string file);
unsigned int getTextureRGB32F(int width, int height);


int main()
{
    Model model;
    string path = "./scenes/";

    cout << "Please choose your model(1, 2, 3)." << endl;
    cout << "*****\t" << "1. veach-mis" << "\t*****" << endl;
    cout << "*****\t" << "2. cornell-box" << "\t*****" << endl;
    cout << "*****\t" << "3. bedroom" << "\t*****" << endl;
    string line(89, '-');
    cout << line << endl;
    int a = 0, count = 5;
    cin >> a;
    while (a != 1 && a != 2 && a != 3 && count != 0) {
        cout << "Choose Error." << endl;
        count--;
        cin >> a;
    }
    string model_n = a == 1 ? "veach-mis" : (a == 2 ? "cornell-box" : "bedroom");
    cout << line << endl;

    model.ReadOBJ(path + model_n + "/" + model_n + ".obj");
    model.ReadMTL(path + model_n + "/" + model_n + ".mtl");
    model.ReadXML(path + model_n + "/" + model_n + ".xml");

    //model.camera.width = model.camera.height = 512;
    const int SCR_WIDTH = model.camera.width;
    const int SCR_HEIGHT = model.camera.height;

    model.Pre2Shader();

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Path Tracing", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    //glEnable(GL_DEPTH_TEST);


    unsigned int Tris;
    unsigned int TBO0;
    glGenBuffers(1, &TBO0);
    glBindBuffer(GL_TEXTURE_BUFFER, TBO0);
    glBufferData(GL_TEXTURE_BUFFER, model.Tris2Shader.size() * sizeof(Triangle2Shader), &model.Tris2Shader[0], GL_STATIC_DRAW);
    glGenTextures(1, &Tris);
    glBindTexture(GL_TEXTURE_BUFFER, Tris);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, TBO0);

    unsigned int Tree;
    unsigned int TBO1;
    glGenBuffers(1, &TBO1);
    glBindBuffer(GL_TEXTURE_BUFFER, TBO1);
    glBufferData(GL_TEXTURE_BUFFER, model.Nodes2Shader.size() * sizeof(BVHNode2Shader), &model.Nodes2Shader[0], GL_STATIC_DRAW);
    glGenTextures(1, &Tree);
    glBindTexture(GL_TEXTURE_BUFFER, Tree);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, TBO1);

    unsigned int Lights;
    unsigned int TBO2;
    glGenBuffers(1, &TBO2);
    glBindBuffer(GL_TEXTURE_BUFFER, TBO2);
    glBufferData(GL_TEXTURE_BUFFER, model.Lights2Shader.size() * sizeof(Triangle2Shader), &model.Lights2Shader[0], GL_STATIC_DRAW);
    glGenTextures(1, &Lights);
    glBindTexture(GL_TEXTURE_BUFFER, Lights);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, TBO2);

    // Rendering Pipeline
    Shader shader1("./shader/vshader.vs", "./shader/fshader1.fs");
    Shader shader2("./shader/vshader.vs", "./shader/fshader2.fs");
    Shader shader3("./shader/vshader.vs", "./shader/fshader3.fs");


    float vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f
    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Position Attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    vector<unsigned int> textures;
    for (int i = 0; i < (int)model.Texture.size(); ++i) {
        unsigned int texture;
        texture = load_img(path + model_n + "/" + model.Texture[i]);
        textures.push_back(texture);
    }

    unsigned int update = getTextureRGB32F(SCR_WIDTH, SCR_HEIGHT);
    unsigned int lastframe = getTextureRGB32F(SCR_WIDTH, SCR_HEIGHT);
    
    unsigned int FBO1, FBO2;
    glGenFramebuffers(1, &FBO1);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO1);
    glBindTexture(GL_TEXTURE_2D, update);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, update, 0);
    glDrawBuffers(1, &FBO1);

    glGenFramebuffers(1, &FBO2);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO2);
    glBindTexture(GL_TEXTURE_2D, lastframe);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lastframe, 0);
    glDrawBuffers(1, &FBO2);

    glm::mat4 view = glm::lookAt(model.camera.eye + vec3(0.0,1.0,1.0),
        model.camera.lookat + vec3(0.0, 1.0, 1.0),
        model.camera.up);

    int frameCounter = 0;
    string oo(35, ' ');
    cout << oo << "RENDER START!" << oo << endl;
    cout << line << endl;
    while (!glfwWindowShouldClose(window))
    {
        cout << "Frame: " << frameCounter + 1 << endl;
        processInput(window);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(VAO);

        shader1.use();
        // Attachment
        glBindFramebuffer(GL_FRAMEBUFFER, FBO1);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, lastframe);
        shader1.setInt("lastframe", 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER, Tris);
        shader1.setInt("Tris", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER, Tree);
        shader1.setInt("Tree", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER, Lights);
        shader1.setInt("Lights", 3);

        shader1.setInt("frameCounter", frameCounter++);
        shader1.setInt("ntriangles", (int)model.Tris2Shader.size());
        shader1.setInt("nlights", (int)model.Lights2Shader.size());

        // Camera Params
        shader1.setMat4("view", inverse(view));
        shader1.setVec3("eye", model.camera.eye);
        shader1.setFloat("fovy", model.camera.fovy);
        shader1.setInt("width", SCR_WIDTH);
        shader1.setInt("height", SCR_HEIGHT);

        // Textures
        int tex_id = 10;
        for (int i = 0; i < (int)textures.size(); ++i, ++tex_id) {
            glActiveTexture(GL_TEXTURE0 + tex_id);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            string sss = "texture" + std::to_string(i);
            shader1.setInt(sss.c_str(), tex_id);
        }

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader2.use();
        glBindFramebuffer(GL_FRAMEBUFFER, FBO2);
        glActiveTexture(GL_TEXTURE20);
        glBindTexture(GL_TEXTURE_2D, update);
        shader2.setInt("update", 20);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader3.use();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glActiveTexture(GL_TEXTURE21);
        glBindTexture(GL_TEXTURE_2D, lastframe);
        shader3.setInt("lastframe", 21);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        /*if (frameCounter % 100 == 0 && frameCounter < 1050) {
            string file = "./results/" + model_n + "_frame" + std::to_string(frameCounter);
            ScreenShot(SCR_WIDTH, SCR_HEIGHT, file);
            cout << "Save result at" << file << "." << endl;
        }*/

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void ScreenShot(int width, int height, string file) {
    RGBColor* ColorBuffer = new RGBColor[width * height];
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, ColorBuffer);
    WriteBMP(file.c_str(), ColorBuffer, width, height);
    delete[] ColorBuffer;
}

unsigned int load_img(string file) {
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    string type(file.end() - 3, file.end());
    int width, height, nrChannels;

    //stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char* data = stbi_load(file.c_str(), &width, &height, &nrChannels, 0);

    if (data)
    {
		if (nrChannels == 3) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
		else if (nrChannels == 4) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

unsigned int getTextureRGB32F(int width, int height) {
    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}