import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt

xml_path = '2D_double_pendulum.xml' #xml file (assumes this is in the same folder as this file)
simend = 5 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

q0_init = -3 * np.pi / 2
q1_init = 0

q0_end = -np.pi / 2
q1_end = np.pi / 2

t_init = 0
t_end = 5

t = []
qact0 = []
qref0 = []
qact1 = []
qref1 = []

def generate_trajectory(t0, tf, q0, qf):
    tf_t0_3 = (tf - t0) ** 3

    a0 = qf*(t0 ** 2)*(3*tf - t0) + q0*(tf ** 2)*(tf - 3*t0)
    a0 /= tf_t0_3

    a1 = 6*t0*tf*(qf - q0)
    a1 /= tf_t0_3

    a2 = 3*(t0 + tf)*(qf - q0)
    a2 = a2 / tf_t0_3

    a3 = 2 * (q0 - qf)
    a3 = a3 / tf_t0_3

    return a0, a1, a2, a3

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    global a_jnt0, a_jnt1
    
    a_jnt0 = generate_trajectory(t_init, t_end, q0_init, q0_end)

    a_jnt1 = generate_trajectory(t_init, t_end, q1_init, q1_end)

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    global a_jnt0, a_jnt1

    time = data.time

    if time > t_end:
        time = t_end

    if time < t_init:
        time = t_init

    q_ref0 = a_jnt0[0] + a_jnt0[1] * time + a_jnt0[2] * time ** 2 + \
            a_jnt0[3] * time ** 3
    
    qdot_ref0 = a_jnt0[1] + 2 * a_jnt0[2] * time + 3 * a_jnt0[3] * time ** 2

    q_ref1 = a_jnt1[0] + a_jnt1[1] * time + a_jnt1[2] * time ** 2 + \
            a_jnt1[3] * time ** 3
    
    qdot_ref1 = a_jnt1[1] + 2 * a_jnt1[2] * time + 3 * a_jnt1[3] * time ** 2
    
    # # PD Control
    # data.ctrl[0] = -500 * (data.qpos[0] - q_ref0) - 50 * (data.qvel[0] - qdot_ref0)
    # data.ctrl[1] = -500 * (data.qpos[1] - q_ref1) - 50 * (data.qvel[1] - qdot_ref1)

    # Model based control (feedback linearization)
    # tau = M * (PD-control) + f
    M = np.zeros((2, 2))
    mj.mj_fullM(model, M, data.qM)
    f0 = data.qfrc_bias[0]
    f1 = data.qfrc_bias[1]
    f = np.array([f0, f1])

    kp = 500
    kd = 2 * np.sqrt(kp)
    pd_0 = -kp * (data.qpos[0] - q_ref0) - kd * (data.qvel[0] - qdot_ref0)
    pd_1 = -kp * (data.qpos[1] - q_ref1) - kd * (data.qvel[1] - qdot_ref1)
    pd_control = np.array([pd_0, pd_1])
    tau_M_pd_control = np.matmul(M, pd_control)
    tau = np.add(tau_M_pd_control, f)
    data.ctrl[0] = tau[0]
    data.ctrl[1] = tau[1]

    t.append(data.time)
    qact0.append(data.qpos[0])
    qref0.append(q_ref0)
    qact1.append(data.qpos[1])
    qref1.append(q_ref1)
    

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 88.64687499999998 ; cam.elevation = 3.4437499999998704 ; cam.distance =  6.2325934695742
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

data.qpos[0] = q0_init
data.qpos[1] = q1_init

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        plt.figure(1)

        plt.subplot(2, 1, 1)
        # plt.plot(t, qact0, "r-")
        # plt.plot(t, qref0, "k")

        plt.plot(t, np.subtract(qref0, qact0), "k")

        plt.subplot(2, 1, 2)
        # plt.plot(t, qact1, "r-")
        # plt.plot(t, qref1, "k")
        plt.plot(t, np.subtract(qref1, qact1), "k")

        plt.show(block=False)
        plt.pause(5)
        plt.close()
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
