import os, os.path
from pathlib import Path

import numpy as np
from PIL import ImageTk, Image
import cv2
import skimage.transform
from scipy.ndimage import gaussian_filter

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename

from boosted_depth.depth_util import create_depth_models, get_depth
from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan
from chrislib.normal_util import get_omni_normals

import intrinsic.model_util
import intrinsic.pipeline

import intrinsic_compositing.shading.pipeline
from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)
import intrinsic_compositing.albedo.pipeline

from omnidata_tools.model_util import load_omni_model

import utils
from shadows import calculate_screen_space_shadows, combine_depth

# ----------------------------------- #
# utils

def update_textarea(textarea, colorstr, text):
    textarea.configure(state="normal")
    textarea.delete(1.0, tk.END)
    textarea.insert(tk.END, text)
    textarea.config(fg=colorstr)
    textarea.configure(state="disabled")

# ----------------------------------- #
# custom interface

class Interface(tk.Tk):
    def __init__(self):
        super().__init__()

        self.DEFAULT_PANE_WIDTH = 256 + 128 - 6
        self.DEFAULT_PANE_HEIGHT = 512+256
        self.PADDING = 8
        
        self.pane_width = 512 + 128
        self.pane_height = self.DEFAULT_PANE_HEIGHT
        self.max_edge_size = 512+256+256

        self.light_vis_size = 144

        self.composite_x = 0.3
        self.composite_y = 0.3
        self.fg_scale_relative = 0.25

        self.bg_path = ""
        self.fg_path = ""

        self.bg_im   = None
        self.fg_im   = None
        self.fg_mask = None
        self.composited_im = np.zeros((self.pane_height, self.pane_width))

        self.bg_depth = None
        self.fg_depth = None
        self.composited_depth = np.zeros((self.pane_height, self.pane_width))

        self.bg_albedo = None
        self.fg_albedo = None
        self.composited_albedo = np.zeros((self.pane_height, self.pane_width))

        self.bg_inv_shading = None
        self.fg_inv_shading = None
        self.composited_inv_shading = np.zeros((self.pane_height, self.pane_width))

        self.bg_normals = None
        self.fg_normals = None
        self.composited_normals = np.zeros((self.pane_height, self.pane_width))

        self.composited_shadow = np.zeros((self.pane_height, self.pane_width))

        # shadow params
        self.light_coeffs = None
        self.shadow_num_steps = 256
        #self.shadow_opacity = 0.45
        #self.shadow_blur_px = 6
        # shadow compositing params
        self.shadow_bg_depth_multiplier = 1.0
        self.shadow_fg_squish    = 0.2
        self.shadow_fg_depth_pad = 0.2
        self.shadow_fg_distance  = 0.6 # can be negative

        self.depth_model     = None
        self.intrinsic_model = None
        self.normals_model   = None
        self.albedo_model    = None
        self.reshading_model = None

        self.init_interface()

        self._active_image_num = 0
        self.set_active_image_num(0)

        self.update_combobox_choices()

        print("starting interface")

    def init_interface(self):
        self.title("cast-shadow compositing demo")
        self.configure(background="black") 
        
        self.left_frame = tk.Frame(self, width=self.pane_width, height=self.pane_height, bg="black")
        self.left_frame.pack()
        self.left_frame.place(x=1*self.PADDING+0*self.pane_width, y=0)
        
        self.left_button_frame = tk.Frame(self.left_frame, width=self.pane_width, height=10, bg="black")
        self.left_button0 = tk.Button(self.left_button_frame, text="Composite [1]", command=lambda: self.set_active_image_num(0))
        self.left_button1 = tk.Button(self.left_button_frame, text="Depth [2]", command=lambda: self.set_active_image_num(1))
        self.left_button2 = tk.Button(self.left_button_frame, text="Inv Shading [3]", command=lambda: self.set_active_image_num(2))
        self.left_button3 = tk.Button(self.left_button_frame, text="Shadow [4]", command=lambda: self.set_active_image_num(3))
        self.left_button4 = tk.Button(self.left_button_frame, text="Harmonized [5]", command=lambda: self.set_active_image_num(4))
        self.left_button_frame.pack(anchor="w", pady=self.PADDING)
        self.left_button0.pack(side="left", padx=self.PADDING)
        self.left_button1.pack(side="left", padx=self.PADDING)
        self.left_button2.pack(side="left", padx=self.PADDING)
        self.left_button3.pack(side="left", padx=self.PADDING)
        self.left_button4.pack(side="left", padx=self.PADDING)
        
        self.left_im_label = ttk.Label(self.left_frame)

        self.right_frame = tk.Frame(self, width=self.pane_width+200, height=self.pane_height)
        self.right_frame.pack()
        self.right_frame.place(x=2*self.PADDING+1*self.pane_width, y=self.PADDING)

        if True:
            self.decompose_pass_label = tk.Label(self.right_frame, text="Pack Generation", background="white", justify="left", anchor="w")    
        
            self.bg_button_frame = tk.Frame(self.right_frame)
            self.bg_button       = tk.Button(self.bg_button_frame, text="Select BG", command=self.load_bg)
            self.bg_path         = tk.StringVar()
            self.bg_path_entry   = tk.Entry(self.bg_button_frame, textvariable=self.bg_path, background="white", justify="left", width=35)
            
            self.fg_button_frame = tk.Frame(self.right_frame)
            self.fg_button       = tk.Button(self.fg_button_frame, text="Select FG", command=self.load_fg)
            self.fg_path         = tk.StringVar()
            self.fg_path_entry   = tk.Entry(self.fg_button_frame, textvariable=self.fg_path, background="white", justify="left", width=35)
            
            self.decompose_pass_frame = tk.Frame(self.right_frame)
            self.decompose_pass_button = tk.Button(self.decompose_pass_frame, text="Generate Pack", command=self.decompose_pass)

            self.decompose_pass_result = tk.Text(self.right_frame, background="white", width=35, height=5, wrap=tk.WORD, borderwidth=0, highlightthickness=0)
            self.decompose_pass_result.configure(state="disabled")

            self.sep0 = ttk.Separator(self.right_frame, orient='horizontal')

            # ------------------------------- #
        
            self.decompose_pass_label.pack(side="top", anchor="w", padx=self.PADDING, pady=self.PADDING)
    
            self.bg_button_frame.pack(anchor="w")
            self.bg_button.pack(side="left", padx=self.PADDING)
            self.bg_path_entry.pack(side="left", padx=self.PADDING)

            self.fg_button_frame.pack(anchor="w")
            self.fg_button.pack(side="left", padx=self.PADDING)
            self.fg_path_entry.pack(side="left", padx=self.PADDING)

            self.decompose_pass_frame.pack(pady=self.PADDING)
            self.decompose_pass_button.pack()

            self.decompose_pass_result.pack(padx=self.PADDING)
            self.sep0.pack(fill='x', pady=self.PADDING)

        if True:
            self.pack_label = tk.Label(self.right_frame, text="Packs", background="white", justify="left", anchor="w")

            self.packs_frame = tk.Frame(self.right_frame)
            self.packs_label = tk.Label(self.packs_frame, text="Current Pack", justify="left", anchor="w") 
            self.packs_combobox_var = tk.StringVar()
            self.packs_combobox = ttk.Combobox(self.packs_frame, textvariable=self.packs_combobox_var)
            self.packs_combobox.state(["readonly"])
            self.packs_load_button   = tk.Button(self.packs_frame, text="Load", command=lambda: self.load_pack())
            self.packs_save_button   = tk.Button(self.packs_frame, text="Save", command=lambda: self.load_pack())
            
            self.packs_result = tk.Text(self.right_frame, background="white", width=35, height=2, wrap=tk.WORD, borderwidth=0, highlightthickness=0)
            self.packs_result.configure(state="disabled")

            self.sep1 = ttk.Separator(self.right_frame, orient='horizontal')

            # ------------------------------- #

            self.pack_label.pack(side="top", anchor="w", padx=self.PADDING)

            self.packs_frame.pack(anchor="w", padx=self.PADDING, pady=self.PADDING)
            self.packs_label.pack(side="left")
            self.packs_combobox.pack(side="left", padx=self.PADDING)
            self.packs_load_button.pack(side="left")
            self.packs_save_button.pack(side="left")

            self.packs_result.pack()
            self.sep1.pack(fill='x', pady=self.PADDING)

        if True:
            self.shadow_label = tk.Label(self.right_frame, text="Shadow", background="white", justify="left", anchor="w")

            self.light_im_label = ttk.Label(self.right_frame)

            self.opacity_var    = tk.DoubleVar()
            self.opacity_slider = tk.Scale(
                self.right_frame, 
                from_=0.0, to=1.0, 
                resolution=0.01,
                length=300, 
                orient=tk.HORIZONTAL, 
                variable=self.opacity_var, 
                label="opacity"
            )
            self.opacity_slider.set(0.5)
            self.blur_var       = tk.IntVar()
            self.blur_slider    = tk.Scale(
                self.right_frame, 
                from_=0, to=32, 
                resolution=1, 
                length=300, 
                orient=tk.HORIZONTAL, 
                variable=self.blur_var,
                label="blur"
            )
            self.blur_slider.set(4)

            self.sep2 = ttk.Separator(self.right_frame, orient='horizontal')
            
            # ------------------------------- #
            
            self.shadow_label.pack(side="top", anchor="w", padx=self.PADDING, pady=self.PADDING)

            self.light_im_label.pack()
            
            self.opacity_slider.pack()
            self.blur_slider.pack()

            self.sep2.pack(fill='x', pady=self.PADDING)

        if True:
            self.compositing_label = tk.Label(self.right_frame, text="Compositing", background="white", justify="left", anchor="w")

            # TODO: add the compositing parameters - depth, etc...

            self.sep3 = ttk.Separator(self.right_frame, orient='horizontal')

            # ------------------------------- #
            
            self.compositing_label.pack(side="top", anchor="w", padx=self.PADDING, pady=self.PADDING)
            
            self.sep3.pack(fill='x', pady=self.PADDING)

        if True:
            self.harmonize_pass_label = tk.Label(self.right_frame, text="Harmonize Pass", background="white", justify="left", anchor="w")

            self.output_frame        = tk.Frame(self.right_frame)
            self.output_button       = tk.Button(self.output_frame, text="Write Final Copy", command=self.harmonize_pass)
            self.output_path         = tk.StringVar()
            self.output_path_entry   = tk.Entry(self.output_frame, textvariable=self.output_path, background="white", justify="left")
            self.output_path_entry.insert(0, "./output.png")
            
            self.harmonize_pass_result = tk.Label(self.right_frame, text="", background="white")

            # ------------------------------- #

            self.harmonize_pass_label.pack(side="top", anchor="w", padx=self.PADDING, pady=self.PADDING)

            self.output_frame.pack()
            self.output_button.pack(side="top", pady=self.PADDING)
            self.output_path_entry.pack(side="top", pady=self.PADDING)

            self.harmonize_pass_result.pack()

        # ------------------------------- #

        window_width = self.pane_width * 1 + self.DEFAULT_PANE_WIDTH + self.PADDING * 7
        window_height = self.pane_height + self.PADDING * 2
        self.geometry(f"{window_width}x{window_height}")

        style = ttk.Style(self)
        style.configure("TFrame", background="white")

        self.bind('<Key>', self.key_pressed)
        self.bind('<B1-Motion>', self.click_motion)
        self.bind('<Button-4>', self.scrolled)
        self.bind('<Button-5>', self.scrolled)
        self.bind('<Button-1>', self.clicked)

    def update_combobox_choices(self):
        if os.path.isdir("packs"):
            # look through the packs folder
            packs = [pack for pack in os.listdir("packs")]
            self.packs_combobox['value'] = tuple(packs)

    def update_light_display(self):
        if self.light_coeffs is None:
            return

        self.light_display_im = utils.viz_coeffs(self.light_coeffs, self.light_vis_size)

        self.light_im_photo = ImageTk.PhotoImage(np_to_pil(self.light_display_im))
        self.light_im_label.configure(image=self.light_im_photo)
        self.light_im_label.image = self.light_im_photo
        self.light_im_label.pack()

        # do an image resize in case vertical height changes
        window_width = self.pane_width * 1 + self.DEFAULT_PANE_WIDTH + self.PADDING * 7 + 4
        window_height = max(
            self.pane_height + self.PADDING * 3 + self.left_button_frame.winfo_height(),
            self.PADDING * 2 + self.right_frame.winfo_height(),
        )
        self.geometry(f"{window_width}x{window_height}")

    def update_display_image(self, im):
        self.left_im_photo = ImageTk.PhotoImage(np_to_pil(im))
        self.left_im_label.configure(image=self.left_im_photo)
        self.left_im_label.image = self.left_im_photo
        self.left_im_label.pack()
        
        # resize the window & location of the main panes
        self.pane_width = im.shape[1]
        self.pane_height = im.shape[0]

        self.left_frame.place(x=1*self.PADDING+0*self.pane_width, y=0)
        self.right_frame.place(x=2*self.PADDING+1*self.pane_width, y=self.PADDING)

        window_width = self.pane_width * 1 + self.DEFAULT_PANE_WIDTH + self.PADDING * 7 + 4
        window_height = max(
            self.pane_height + self.PADDING * 3 + self.left_button_frame.winfo_height(),
            self.PADDING * 2 + self.right_frame.winfo_height(),
        )
        self.geometry(f"{window_width}x{window_height}")

    # ----------------------------------- #
    # inputs
        
    def scrolled(self, e):
        if e.num == 5 and self.fg_scale_relative > 0.05: # scroll down
            self.fg_scale_relative -= 0.01
        elif e.num == 4 and self.fg_scale_relative < 1.0: # scroll up
            self.fg_scale_relative += 0.01

        if not self.fg_im is None:
            self.update_active_rescaled_fg()
            self.update_active_image()
            self.update_display_image(self.get_active_image())

    def clicked(self, e):
        x, y = e.x, e.y
        radius = self.light_vis_size // 2

        if e.widget == self.light_im_label:
            # if the user clicked the light ball, compute the direction from mouse pos
            rel_x = (x - radius) / radius
            rel_y = (y - radius) / radius
            
            z = np.sqrt(1 - rel_x ** 2 - rel_y ** 2)
            
            # print('clicked the lighting viz:', rel_x, rel_y, z)

            # after converting the mouse pos to a normal direction on a unit sphere
            # we can create our 4D lighting coefficients using the slider values
            self.light_coeffs = np.array([0, 0, 0, self.light_coeffs[3]])
            direction_vector = np.array([rel_x, -rel_y, z]) * float(self.opacity_slider.get()) # light intensity
            self.light_coeffs[:3] = direction_vector

            self.update_light_display()

    def click_motion(self, e):
        x, y = e.x, e.y
        if e.widget == self.left_im_label:
            if (y <= self.bg_im.shape[0]) and (x <= self.bg_im.shape[1]):
                self.composite_y = y / self.bg_im.shape[0]
                self.composite_x = x / self.bg_im.shape[1]

                if not self.fg_im is None:
                    self.update_active_image()
                    self.update_display_image(self.get_active_image())

    def key_pressed(self, e):
        if e.char == "1":
            self.set_active_image_num(0)
        elif e.char == "2":
            self.set_active_image_num(1)
        elif e.char == "3":
            self.set_active_image_num(2)
        elif e.char == "4":
            self.set_active_image_num(3)
        elif e.char == "5":
            self.set_active_image_num(4)

        self.update_display_image(self.get_active_image())

    # ----------------------------------- #
    # panel composite update logic:
        
    def set_active_image_num(self, num):
        self._active_image_num = num
        if num == 0:
            self.left_button0['bg'] = "light goldenrod yellow"
            self.left_button1['bg'] = "#d9d9d9"
            self.left_button2['bg'] = "#d9d9d9"
            self.left_button3['bg'] = "#d9d9d9"
            self.left_button4['bg'] = "#d9d9d9"
        elif num == 1:
            self.left_button0['bg'] = "#d9d9d9"
            self.left_button1['bg'] = "light goldenrod yellow"
            self.left_button2['bg'] = "#d9d9d9"
            self.left_button3['bg'] = "#d9d9d9"
            self.left_button4['bg'] = "#d9d9d9"
        elif num == 2:
            self.left_button0['bg'] = "#d9d9d9"
            self.left_button1['bg'] = "#d9d9d9"
            self.left_button2['bg'] = "light goldenrod yellow"
            self.left_button3['bg'] = "#d9d9d9"
            self.left_button4['bg'] = "#d9d9d9"
        elif num == 3:
            self.left_button0['bg'] = "#d9d9d9"
            self.left_button1['bg'] = "#d9d9d9"
            self.left_button2['bg'] = "#d9d9d9"
            self.left_button3['bg'] = "light goldenrod yellow"
            self.left_button4['bg'] = "#d9d9d9"
        elif num == 4:
            self.left_button0['bg'] = "#d9d9d9"
            self.left_button1['bg'] = "#d9d9d9"
            self.left_button2['bg'] = "#d9d9d9"
            self.left_button3['bg'] = "#d9d9d9"
            self.left_button4['bg'] = "light goldenrod yellow"
        else:
            raise Exception(f"invalid num {num}")
        
        if not self.fg_im is None:
            self.update_active_rescaled_fg()
            self.update_active_image()
        self.update_display_image(self.get_active_image())
    
    def get_active_image(self):
        if self._active_image_num == 0:
            return self.composited_im
        elif self._active_image_num == 1:
            return self.composited_depth
        elif self._active_image_num == 2:
            return self.composited_inv_shading
        elif self._active_image_num == 3:
            return self.composited_shadow
        elif self._active_image_num == 4:
            return np.zeros_like(self.composited_im)

    def update_active_image(self):
        if self._active_image_num == 0:
            self.update_composited_image()
        elif self._active_image_num == 1:
            self.update_composited_depth()
        elif self._active_image_num == 2:
            self.update_composited_inv_shading()
        elif self._active_image_num == 3:
            self.update_composited_shadow()
        elif self._active_image_num == 4:
            pass

    def update_active_rescaled_fg(self):
        fg_scaled_height = int(self.fg_im.shape[0] * self.fg_scale_relative)
        fg_scaled_width  = int(self.fg_im.shape[1] * self.fg_scale_relative)
        
        if self._active_image_num == 0:
            if self.fg_im is None: return
            self.fg_im_rescaled   = cv2.resize(self.fg_im, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)
            self.fg_mask_rescaled = cv2.resize(self.fg_mask, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)
        
        elif self._active_image_num == 1:
            if self.fg_depth is None: return
            self.fg_depth_rescaled = cv2.resize(self.fg_depth, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)
            self.fg_mask_rescaled  = cv2.resize(self.fg_mask, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)

        elif self._active_image_num == 2:
            if self.fg_inv_shading is None: return
            self.fg_inv_shading_rescaled = cv2.resize(self.fg_inv_shading, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)
            self.fg_mask_rescaled        = cv2.resize(self.fg_mask, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)

        elif self._active_image_num == 3:
            if self.fg_mask is None: return
            self.fg_mask_rescaled = cv2.resize(self.fg_mask, (fg_scaled_width, fg_scaled_height), interpolation=cv2.INTER_AREA)

        elif self._active_image_num == 4:
            pass

    def update_composited_image(self):
        if self.bg_im is None:
            return np.zeros_like(self.composited_im)

        top  = int(self.composite_y * self.bg_im.shape[0])
        left = int(self.composite_x * self.bg_im.shape[1])

        self.composited_im = utils.composite_crop(
            self.bg_im,
            (top, left),
            self.fg_im_rescaled,
            self.fg_mask_rescaled,
        )

    def update_composited_depth(self):
        if self.bg_depth is None:
            return np.zeros_like(self.composited_im)
        
        top  = int(self.composite_y * self.bg_im.shape[0])
        left = int(self.composite_x * self.bg_im.shape[1])
        
        # TODO: add user controls to this
        self.composited_depth = utils.composite_depth(
            self.bg_depth,
            (top, left),
            self.fg_depth_rescaled,
            self.fg_mask_rescaled,
        )

    def update_composited_inv_shading(self):
        if self.bg_inv_shading is None:
            return np.zeros_like(self.composited_im)
        
        top  = int(self.composite_y * self.bg_im.shape[0])
        left = int(self.composite_x * self.bg_im.shape[1])
        
        self.composited_inv_shading = utils.composite_crop(
            self.bg_inv_shading[:, :, 0],
            (top, left),
            self.fg_inv_shading_rescaled,
            self.fg_mask_rescaled,
        )

    # actually generate the shadows
    def update_composited_shadow(self):
        if self.composited_depth is None or self.light_coeffs is None:
            print("invalid conditions")
            return np.zeros_like(self.composited_im)
            
        top  = int(self.composite_y * self.bg_im.shape[0])
        left = int(self.composite_x * self.bg_im.shape[1])

        # generate fg_full_mask given current top,left of composite
        fg_full_mask = np.zeros((self.bg_im.shape[0], self.bg_im.shape[1]), dtype=np.float32)
        fg_full_mask[
            top : top + self.fg_im_rescaled.shape[0], 
            left : left + self.fg_im_rescaled.shape[1],
        ] = self.fg_mask_rescaled

        # NOTE: includes self-shading
        all_shaded_mask = calculate_screen_space_shadows(
            self.light_coeffs[:3],
            self.composited_depth,
            fg_full_mask,
            depth_cutoff=0.0,
            max_steps=128,
        )

        # remove self-shading
        shaded_mask = all_shaded_mask.copy()
        shaded_mask[fg_full_mask > 0.0] = 0

        # get to-composite shadow
        blurred_shadow_mask = np.zeros((self.composited_depth.shape[0], self.composited_depth.shape[1]))
        blurred_shadow_mask[:, :] = shaded_mask * float(self.opacity_slider.get())
        blurred_shadow_mask[:, :] = gaussian_filter(blurred_shadow_mask[:, :], sigma=int(self.blur_slider.get()))

        self.composited_shadow = blurred_shadow_mask
        
    # ----------------------------------- #

    def load_bg(self):
        self.bg_path = askopenfilename()

        if self.bg_path == () or not os.path.isfile(self.bg_path):
            update_textarea(self.decompose_pass_result, "OrangeRed2", "error: invalid bg_path. not a file")
            self.bg_path = ""
            return

        try:
            print("\n2.1 load & resize bg_im")
            
            self.bg_im_original = load_image(self.bg_path)
            print("\tloaded image")

            max_dim = max(self.bg_im_original.shape[0], self.bg_im_original.shape[1])
            scale = self.max_edge_size / max_dim
            bg_height = int(self.bg_im_original.shape[0] * scale)
            bg_width  = int(self.bg_im_original.shape[1] * scale)
            self.bg_im = cv2.resize(self.bg_im_original, (bg_width, bg_height), interpolation=cv2.INTER_AREA)
            print("\tresized image")

            if self.fg_im is None:
                self.composited_im = self.bg_im
            else:
                self.set_active_image_num(0)
            
            self.composited_depth       = np.zeros_like(self.composited_im)
            self.composited_inv_shading = np.zeros_like(self.composited_im)
            self.composited_albedo      = np.zeros_like(self.composited_im)
            self.composited_normals     = np.zeros_like(self.composited_im)
            self.composited_shadow      = np.zeros_like(self.composited_im)

            self.update_display_image(self.composited_im)

            self.bg_path_entry.delete(0, "end")
            self.bg_path_entry.insert(0, self.bg_path)

            update_textarea(self.decompose_pass_result, "cornflower blue", f"loaded bg im")

        except Exception as e:
            update_textarea(self.decompose_pass_result, "OrangeRed2", f"error: {e}. try again")
            self.bg_path = ""
            raise e

    def load_fg(self):
        if self.bg_im is None:
            update_textarea(self.decompose_pass_result, "OrangeRed2", "error: bg_path is None. please load bg_im first")
            return
        
        self.fg_path = askopenfilename()
        if self.fg_path == () or not os.path.isfile(self.fg_path):
            update_textarea(self.decompose_pass_result, "OrangeRed2", "error: invalid fg_path. not a file")
            self.fg_path = ""
            return

        try:
            print("\n2.2 load, resize, and apply mask to fg_im")

            fg_im_with_alpha = load_image(self.fg_path)
            self.fg_im_original = fg_im_with_alpha[:, :, :3]
            self.fg_mask_original = fg_im_with_alpha[:, :, 3]
            print(f"\tfg_im shape: {self.fg_im_original.shape}")
            print(f"\tfg_mask shape: {self.fg_mask_original.shape}")

            bb = utils.get_bbox(self.fg_mask_original)
            fg_im = self.fg_im_original[bb[0]:bb[1], bb[2]:bb[3], :].copy()
            fg_mask = self.fg_mask_original[bb[0]:bb[1], bb[2]:bb[3]].copy()
            max_dim = max(fg_im.shape[0], fg_im.shape[1])
            fg_scale = self.max_edge_size / max_dim
            cropped_height = int(fg_im.shape[0] * fg_scale)
            cropped_width  = int(fg_im.shape[1] * fg_scale)
            self.fg_im = skimage.transform.resize(fg_im, (cropped_height, cropped_width))
            self.fg_mask = skimage.transform.resize(fg_mask, (cropped_height, cropped_width))
            print(f"\tfg_im shape: {self.fg_im.shape}")
            print(f"\tfg_mask shape: {self.fg_mask.shape}")

            self.set_active_image_num(0)

            self.fg_path_entry.delete(0, "end")
            self.fg_path_entry.insert(0, self.fg_path)

            update_textarea(self.decompose_pass_result, "cornflower blue", f"loaded fg im")

        except Exception as e:
            update_textarea(self.decompose_pass_result, "OrangeRed2", f"error: {e}. try again")
            self.fg_path = ""

    def decompose_pass(self):
        if self.depth_model is None:
            print("\n1.1 loading depth model")
            self.depth_model = create_depth_models()

        if self.intrinsic_model is None:
            print("\n1.2 loading intrinsic decomposition model")
            self.intrinsic_model = intrinsic.model_util.load_models('paper_weights')

        if self.normals_model is None:
            print("\n1.3 loading normals model")
            self.normals_model = load_omni_model()
        
        original_image_num = self._active_image_num

        print("\ncomputing depth...")
        self.bg_depth = get_depth(self.bg_im, self.depth_model)
        self.fg_depth = get_depth(self.fg_im, self.depth_model)
        self.set_active_image_num(1)
        
        print("computing albedo & inv shading...")
        bg_result = intrinsic.pipeline.run_pipeline(
            self.intrinsic_model,
            self.bg_im ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True,
        )
        fg_result = intrinsic.pipeline.run_pipeline(
            self.intrinsic_model,
            self.fg_im ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True,
        )
        self.bg_albedo      = bg_result["albedo"]
        self.bg_inv_shading = bg_result["inv_shading"][:, :, np.newaxis]
        self.fg_albedo      = fg_result["albedo"]
        self.fg_inv_shading = fg_result["inv_shading"][:, :, np.newaxis]
        self.set_active_image_num(2)

        print("compute normals...")
        self.bg_normals = get_omni_normals(self.normals_model, self.bg_im)
        self.fg_normals = get_omni_normals(self.normals_model, self.fg_im)

        self.set_active_image_num(original_image_num)

        print("solving for light coefficients...")
        self.solve_for_light_coef()
        self.update_light_display()

        update_textarea(self.decompose_pass_result, "SpringGreen3", f"success")
        self.update()

        print("writing pack...")
        self.write_pack()

        print("done!")

    def solve_for_light_coef(self):
        if (self.bg_im is None) or (self.bg_inv_shading is None):
            return
        
        if self.normals_model is None:
            print("\n* loading normals model")
            self.normals_model = load_omni_model()

        # to ensure that normals are globally accurate we compute them at
        # a resolution of 512 pixels, so resize our shading and image to compute 
        # rescaled normals, then run the lighting model optimization
        max_dim = max(self.bg_im.shape[0], self.bg_im.shape[1])
        small_height = int(self.bg_im.shape[0] * (512.0 / max_dim))
        small_width = int(self.bg_im.shape[1] * (512.0 / max_dim))
        # TODO: speed up this resizing
        small_bg_im = skimage.transform.resize(self.bg_im, (small_height, small_width), anti_aliasing=True)
        small_bg_normals = get_omni_normals(self.normals_model, small_bg_im)
        # TODO: speed up this resizing
        small_bg_inv_shading = skimage.transform.resize(self.bg_inv_shading, (small_height, small_width), anti_aliasing=True)

        print("solve system...")
        coeffs, light_vis = intrinsic_compositing.shading.pipeline.get_light_coeffs(
            small_bg_inv_shading[:, :, 0], 
            small_bg_normals,
            small_bg_im,
        )

        self.light_coeffs = coeffs
        print(list(self.light_coeffs))

    def shadow_pass(self):
        pass

    def harmonize_pass(self):
        self.harmonize_pass_result["text"] = "success"
        self.harmonize_pass_result.config(fg="SpringGreen3")
        pass

    # ----------------------------------- #

    def write_pack(self):
        bg_im_name = os.path.basename(self.bg_path).split(".")[0]
        fg_im_name = os.path.basename(self.fg_path).split(".")[0]
        pack_name = f"{bg_im_name}_{fg_im_name}"
        
        print(f"writing pack {pack_name}...")

        output_files = {
            "bg_im.png": self.bg_im,
            "fg_im.png": self.fg_im,
            "fg_mask.png": self.fg_mask,

            #"bg_im_original.png": self.bg_im_original,
            #"fg_im_original.png": self.fg_im_original,
            #"fg_mask_original.png": self.fg_mask_original,

            "bg_depth.png": self.bg_depth,
            "fg_depth.png": self.fg_depth,
            "bg_inv_shading.png": self.bg_inv_shading,
            "fg_inv_shading.png": self.fg_inv_shading,
            "bg_albedo.png": self.bg_albedo,
            "fg_albedo.png": self.fg_albedo,
            "bg_normals.png": self.bg_normals,
            "fg_normals.png": self.fg_normals,
            
            "data.txt": f"x={self.composite_x}\n"
                      + f"y={self.composite_y}\n"
                      + f"fg_scale_relative={self.fg_scale_relative}\n"
                      + f"light_coeffs={list(self.light_coeffs)}\n",
                    
                    # TODO: add more parameters
        }

        Path(f"packs/{pack_name}").mkdir(parents=True, exist_ok=True)

        # write all the output files
        for file_name, contents in output_files.items():
            if type(contents) is str:
                with open(f"packs/{pack_name}/{file_name}", "w") as f:
                    f.write(contents)
            elif len(contents.shape) == 3 and contents.shape[2] == 1:
                np_to_pil(contents[:, :, 0]).save(f"packs/{pack_name}/{file_name}")
            else:
                np_to_pil(contents).save(f"packs/{pack_name}/{file_name}")
        
        self.update_combobox_choices()
        self.packs_combobox.set(pack_name)

        update_textarea(self.decompose_pass_result, "SpringGreen3", f"successfully wrote files for {pack_name}")

    def load_pack(self):
        pack_name = self.packs_combobox.get()

        print(f"loading images from pack {pack_name}")

        self.bg_im = load_image(f"packs/{pack_name}/bg_im.png")
        self.fg_im = load_image(f"packs/{pack_name}/fg_im.png")
        self.fg_mask = load_image(f"packs/{pack_name}/fg_mask.png")

        #"bg_im_original.png": self.bg_im_original,
        #"fg_im_original.png": self.fg_im_original,
        #"fg_mask_original.png": self.fg_mask_original,

        self.bg_depth = load_image(f"packs/{pack_name}/bg_depth.png")
        self.fg_depth = load_image(f"packs/{pack_name}/fg_depth.png")
        self.bg_inv_shading = load_image(f"packs/{pack_name}/bg_inv_shading.png")[:, :, np.newaxis]
        self.fg_inv_shading = load_image(f"packs/{pack_name}/fg_inv_shading.png")[:, :, np.newaxis]

        self.bg_albedo  = load_image(f"packs/{pack_name}/bg_albedo.png")
        self.fg_albedo  = load_image(f"packs/{pack_name}/fg_albedo.png")
        self.bg_normals = load_image(f"packs/{pack_name}/bg_normals.png")
        self.fg_normals = load_image(f"packs/{pack_name}/fg_normals.png")

        with open(f"packs/{pack_name}/data.txt", "r") as f:
            lines = f.readlines()
            self.composite_x       = float(lines[0].split("=")[1])
            self.composite_y       = float(lines[1].split("=")[1])
            self.fg_scale_relative = float(lines[2].split("=")[1])
            self.light_coeffs      = np.asarray(
                [
                    float(lines[3].replace(" ", "").replace("[", "").replace("]", "").split("=")[1].split(",")[0]),
                    float(lines[3].replace(" ", "").replace("[", "").replace("]", "").split("=")[1].split(",")[1]),
                    float(lines[3].replace(" ", "").replace("[", "").replace("]", "").split("=")[1].split(",")[2]),
                    float(lines[3].replace(" ", "").replace("[", "").replace("]", "").split("=")[1].split(",")[3]),
                ]
            )

        original_image_num = self._active_image_num
        self.set_active_image_num(0)
        self.set_active_image_num(1)
        self.set_active_image_num(2)
        self.set_active_image_num(original_image_num)

        self.update_light_display()

        update_textarea(self.packs_result, "SpringGreen3", f"loaded pack {pack_name}")

if __name__ == "__main__":
    # for wsl using xlaunch
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # make directory in case
    Path("packs").mkdir(parents=True, exist_ok=True)

    # run gui
    app = Interface()
    app.mainloop()
