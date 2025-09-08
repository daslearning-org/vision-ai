# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread

from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.uix.image import Image

from kivymd.app import MDApp
from kivymd.uix.navigationdrawer import MDNavigationDrawerMenu
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

# Import your local screen classes & modules
from screens.img_obj_detect import ImgObjDetBox

## Global definitions
__version__ = "0.0.1"
# Determine the base path for your application's resources
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
kv_file_path = os.path.join(base_path, 'main_layout.kv')


## define custom kivymd classes
class ContentNavigationDrawer(MDNavigationDrawerMenu):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()

## kivymd app class
class VisionAiApp(MDApp):
    is_onnx_running = ObjectProperty()
    image_path = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        self.top_menu_items = {
            "Documentation": {
                "icon": "file-document-check",
                "action": "web",
                "url": "https://blog.daslearning.in/llm_ai/genai/image-to-animation.html",
            },
            "Contact Us": {
                "icon": "card-account-phone",
                "action": "web",
                "url": "https://daslearning.in/contact/",
            },
            "Check for update": {
                "icon": "github",
                "action": "update",
                "url": "",
            }
        }
        return Builder.load_file(kv_file_path)

    def on_start(self):
        # paths setup
        if platform == "android":
            from android.permissions import request_permissions, Permission
            from jnius import autoclass, PythonJavaClass, java_method
            request_permissions([
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE
            ])
            context = autoclass('org.kivy.android.PythonActivity').mActivity
            android_path = context.getExternalFilesDir(None).getAbsolutePath()
            self.model_dir = os.path.join(android_path, 'model_files')
            self.op_dir = os.path.join(android_path, 'outputs')
            image_dir = os.path.join(android_path, 'images')
            os.makedirs(image_dir, exist_ok=True)
            self.internal_storage = android_path
            try:
                Environment = autoclass("android.os.Environment")
                self.external_storage = Environment.getExternalStorageDirectory().getAbsolutePath()
            except Exception:
                self.external_storage = os.path.abspath("/storage/emulated/0/")
        else:
            self.internal_storage = os.path.abspath("/")
            self.external_storage = os.path.abspath("/")
            self.model_dir = os.path.join(self.user_data_dir, 'model_files')
            self.op_dir = os.path.join(android_path, 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.op_dir, exist_ok=True)

        # file managers
        self.is_img_manager_open = False
        self.img_file_manager = MDFileManager(
            exit_manager=self.img_file_exit_manager,
            select_path=self.select_img_path,
            ext=[".png", ".jpg", ".jpeg"],  # Restrict to image files
            selector="file",  # Restrict to selecting files only
            preview=True,
            #show_hidden_files=True,
        )
        # debug
        print(self.root.ids)

    def show_toast_msg(self, message, is_error=False):
        from kivymd.uix.snackbar import MDSnackbar
        bg_color = (0.2, 0.6, 0.2, 1) if not is_error else (0.8, 0.2, 0.2, 1)
        MDSnackbar(
            MDLabel(
                text = message,
                font_style = "Subtitle1"
            ),
            md_bg_color=bg_color,
            y=dp(24),
            pos_hint={"center_x": 0.5},
            duration=3
        ).open()

    def open_img_file_manager(self):
        """Open the file manager to select an image file. On android use Downloads or Pictures folders only"""
        if self.is_onnx_running:
            self.show_toast_msg("Please wait for the current operation to finish", is_error=True)
            return
        try:
            self.img_file_manager.show(self.internal_storage)  # native app specific path
            self.is_img_manager_open = True
        except Exception as e:
            self.show_toast_msg(f"Error: {e}", is_error=True)

    def select_img_path(self, path: str):
        self.image_path = path
        uploaded_image_box = self.root.ids.img_detect_box.ids.uploaded_image
        uploaded_image_box.clear_widgets()
        fitImage = Image(
            source = path,
            fit_mode = "contain"
        )
        uploaded_image_box.add_widget(fitImage)
        self.img_file_exit_manager()

    def img_file_exit_manager(self, *args):
        """Closes the file manager for image upload"""
        self.is_img_manager_open = False
        self.img_file_manager.close()

if __name__ == '__main__':
    VisionAiApp().run()
