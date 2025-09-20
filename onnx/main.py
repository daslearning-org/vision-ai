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
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

# Import your local screen classes & modules
from screens.img_obj_detect import ImgObjDetBox, TempSpinWait
from screens.setting import SettingsBox
from onnx_vision import OnnxDetect

## Global definitions
__version__ = "0.0.2"
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
    image_path = StringProperty("")

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
            self.op_dir = os.path.join(self.user_data_dir, 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.op_dir, exist_ok=True)

        # create onnx objects
        self.onnx_detect = OnnxDetect(
            save_dir=self.op_dir,
            model_dir=self.model_dir,
        )

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

    def show_text_dialog(self, title, text="", buttons=[]):
        self.txt_dialog = MDDialog(
            title=title,
            text=text,
            buttons=buttons
        )
        self.txt_dialog.open()

    def txt_dialog_closer(self, instance):
        if self.txt_dialog:
            self.txt_dialog.dismiss()

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
        result_box = self.root.ids.img_detect_box.ids.result_image
        result_box.clear_widgets()
        self.img_file_exit_manager()

    def img_file_exit_manager(self, *args):
        """Closes the file manager for image upload"""
        self.is_img_manager_open = False
        self.img_file_manager.close()

    def submit_onnx_detect(self):
        if self.image_path == "":
            self.show_toast_msg("No image is selected!", is_error=True)
            return
        if self.is_onnx_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        onnx_thread = Thread(target=self.onnx_detect.run_detect, args=(self.image_path, self.onnx_detect_callback), daemon=True)
        onnx_thread.start()
        self.is_onnx_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.img_detect_box.ids.result_image
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def onnx_detect_callback(self, onnx_resp):
        status = onnx_resp["status"]
        message = onnx_resp["message"]
        self.is_onnx_running = False
        if status is True:
            self.show_toast_msg(f"Output generated at: {message}")
            result_box = self.root.ids.img_detect_box.ids.result_image
            result_box.clear_widgets()
            fitImage = Image(
                source = message,
                fit_mode = "contain"
            )
            result_box.add_widget(fitImage)
        else:
            self.show_toast_msg(message, is_error=True)

    def reset_object_detect(self):
        self.image_path = ""
        uploaded_image_box = self.root.ids.img_detect_box.ids.uploaded_image
        uploaded_image_box.clear_widgets()
        result_box = self.root.ids.img_detect_box.ids.result_image
        result_box.clear_widgets()

    def open_link(self, instance, url):
        import webbrowser
        webbrowser.open(url)

    def update_link_open(self, instance):
        self.txt_dialog_closer(instance)
        self.open_link(instance=instance, url="https://github.com/daslearning-org/vision-ai/releases")

    def update_checker(self, instance):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Releases",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.update_link_open
            ),
        ]
        self.show_text_dialog(
            "Check for update",
            f"Your version: {__version__}",
            buttons
        )

    def show_delete_alert(self):
        op_img_count = 0
        for filename in os.listdir(self.op_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                op_img_count += 1
        self.show_text_dialog(
            title="Delete all output files?",
            text=f"There are total: {op_img_count} image files. This action cannot be undone!",
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.txt_dialog_closer
                ),
                MDFlatButton(
                    text="DELETE",
                    theme_text_color="Custom",
                    text_color="red",
                    on_release=self.delete_op_action
                ),
            ],
        )

    def delete_op_action(self, instance):
        # Custom function called when DISCARD is clicked
        for filename in os.listdir(self.op_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                file_path = os.path.join(self.op_dir, filename)
                try:
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Could not delete the audion files, error: {e}")
        self.show_toast_msg("Executed the audio cleanup!")
        self.txt_dialog_closer(instance)

    def events(self, instance, keyboard, keycode, text, modifiers):
        """Handle mobile device button presses (e.g., Android back button)."""
        if keyboard in (1001, 27):  # Android back button or equivalent
            if self.is_img_manager_open:
                # Check if we are at the root of the directory tree
                if self.img_file_manager.current_path == self.external_storage:
                    self.show_toast_msg(f"Closing file manager from main storage")
                    self.img_file_exit_manager()
                else:
                    self.img_file_manager.back()  # Navigate back within file manager
                return True  # Consume the event to prevent app exit
        return False

if __name__ == '__main__':
    VisionAiApp().run()
