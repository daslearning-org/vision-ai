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
from kivy.uix.camera import Camera

from kivymd.app import MDApp
from kivymd.uix.navigationdrawer import MDNavigationDrawerMenu
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDFloatingActionButton

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

# Import your local screen classes & modules
from screens.img_obj_detect import ImgObjDetBox, TempSpinWait
from screens.cam_obj_detect import CamObjDetBox
from screens.setting import SettingsBox
from onnx_vision import OnnxDetect

## Global definitions
__version__ = "0.0.4"
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
    op_img_path = StringProperty("")
    onnx_detect = ObjectProperty(None)
    cam_found = ObjectProperty(None)
    camera = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        return Builder.load_file(kv_file_path)

    def on_start(self):
        # paths setup
        if platform == "android":
            from android.permissions import request_permissions, Permission
            from jnius import autoclass, PythonJavaClass, java_method
            sdk_version = 28
            try:
                VERSION = autoclass('android.os.Build$VERSION')
                sdk_version = VERSION.SDK_INT
                print(f"Android SDK: {sdk_version}")
                #self.show_toast_msg(f"Android SDK: {sdk_version}")
            except Exception as e:
                print(f"Could not check the android SDK version: {e}")
                #self.show_toast_msg(f"Error checking SDK: {e}", is_error=True)
            permissions = [Permission.CAMERA]
            if sdk_version >= 33:  # Android 13+
                permissions.append(Permission.READ_MEDIA_IMAGES)
            else:  # Android 10–12
                permissions.extend([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
            request_permissions(permissions)
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
            self.internal_storage = os.path.expanduser("~")
            self.external_storage = os.path.expanduser("~")
            os.makedirs(os.path.join(self.internal_storage, 'images'), exist_ok=True)
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
        self.is_op_file_mgr_open = False
        self.op_file_manager = MDFileManager(
            exit_manager=self.op_file_exit_manager,
            select_path=self.select_op_path,
            selector="folder",  # Restrict to selecting directories only
        )
        print("Initialisation is successfull")

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

    def on_img_obj_detect(self):
        self.show_toast_msg("You can detect objects on your image!")

    def on_cam_obj_detect(self):
        self.show_toast_msg("Capture an image & detect objects!")
        self.cam_uix = self.root.ids.cam_detect_box.ids.capture_image
        self.cam_uix.clear_widgets()
        if platform == "android":
            cam_indx = 0
            resolution = (960, 720) # will fallback to 480 if fails again
        else:
            resolution = (640, 480)
            import cv2
            available_cameras = []
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Camera found at index: {i}")
                    available_cameras.append(i)
                    cap.release()
            if len(available_cameras) >= 1:
                cam_indx = available_cameras[0]
            else:
                self.show_toast_msg(f"No camera found on {platform}!", is_error=True)
                return
        try:
            self.camera = Camera(
                index = cam_indx,
                resolution = resolution,
                fit_mode = "contain",
                play = True
            )
            self.cam_uix.add_widget(self.camera)
            self.cam_found = True
        except Exception as e:
            print(f"Error setting up the camera: {e}")
            self.show_toast_msg(f"Error setting up the camera: {e}", is_error=True)

    def on_cam_obj_dt_leave(self):
        if self.cam_found:
            self.camera.play = False
            self.cam_uix.clear_widgets()

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

    def open_op_file_manager(self, instance):
        """Open the file manager to select destination folder. On android use Downloads or Pictures folders only"""
        try:
            self.op_file_manager.show(self.external_storage)
            self.is_op_file_mgr_open = True
        except Exception as e:
            self.show_toast_msg(f"Error: {e}", is_error=True)

    def download_n_remove_file(self, source, dest):
        import shutil
        try:
            shutil.copyfile(source, dest)
            print(f"File successfully download to: {dest}")
            self.show_toast_msg(f"File download to: {dest}")
            os.remove(source)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            self.show_toast_msg(f"Error saving file: {e}", is_error=True)
            return False

    def select_op_path(self, path: str):
        """
        Called when a directory is selected. Save the Output file.
        """
        filename = os.path.basename(self.op_img_path)
        chosen_path = os.path.join(path, filename) # destination path
        download_stat = self.download_n_remove_file(self.op_img_path, chosen_path)
        if download_stat:
            self.op_img_path = ""
            if self.root.ids.screen_manager.current == "imgObjDetect":
                result_box = self.root.ids.img_detect_box.ids.result_image
            else:
                result_box = self.root.ids.cam_detect_box.ids.cam_result_image
            result_box.clear_widgets()
        self.op_file_exit_manager()

    def op_file_exit_manager(self, *args):
        """Called when the user reaches the root of the directory tree."""
        self.is_op_file_mgr_open = False
        self.op_file_manager.close()

    def submit_onnx_detect(self):
        if self.image_path == "":
            self.show_toast_msg("No image is selected!", is_error=True)
            return
        if self.is_onnx_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        onnx_thread = Thread(target=self.onnx_detect.run_detect, args=(self.image_path, self.onnx_detect_callback, "imgObjDetect"), daemon=True)
        onnx_thread.start()
        self.is_onnx_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.img_detect_box.ids.result_image
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def capture_n_onnx_detect(self):
        if not self.cam_found:
            self.show_toast_msg("Camera could not be loaded!", is_error=True)
            return
        if self.is_onnx_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        self.image_path = ""
        import datetime
        now = datetime.datetime.now()
        current_time = str(now.strftime("%H%M%S"))
        current_date = str(now.strftime("%Y%m%d"))
        internal_dir = os.path.join(self.internal_storage, 'images')
        capture_file = f"aivision-{current_date}-{current_time}.png"
        self.image_path = os.path.join(internal_dir, capture_file)
        self.camera.export_to_png(self.image_path)
        onnx_thread = Thread(target=self.onnx_detect.run_detect, args=(self.image_path, self.onnx_detect_callback, "camObjDetect"), daemon=True)
        onnx_thread.start()
        self.is_onnx_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.cam_detect_box.ids.cam_result_image
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def onnx_detect_callback(self, onnx_resp):
        status = onnx_resp["status"]
        message = onnx_resp["message"]
        caller = onnx_resp["caller"]
        self.is_onnx_running = False
        if caller == "camObjDetect":
            result_box = self.root.ids.cam_detect_box.ids.cam_result_image
        else:
            result_box = self.root.ids.img_detect_box.ids.result_image
        if status is True:
            self.show_toast_msg(f"Output generated at: {message}")
            self.op_img_path = message
            result_box.clear_widgets()
            fitImage = Image(
                source = message,
                fit_mode = "contain"
            )
            result_box.add_widget(fitImage)
            down_btn = MDFloatingActionButton(
                icon="download",
                type="small",
                theme_icon_color="Custom",
                md_bg_color='#e9dff7',
                icon_color='#211c29',
            )
            down_btn.bind(on_release=self.open_op_file_manager)
            result_box.add_widget(down_btn)
        else:
            self.show_toast_msg(message, is_error=True)

    def reset_object_detect(self):
        self.image_path = ""
        uploaded_image_box = self.root.ids.img_detect_box.ids.uploaded_image
        uploaded_image_box.clear_widgets()
        result_box = self.root.ids.img_detect_box.ids.result_image
        result_box.clear_widgets()

    def reset_cam_object_detect(self):
        self.image_path = ""
        result_box = self.root.ids.cam_detect_box.ids.cam_result_image
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
            if self.is_op_file_mgr_open:
                # Check if we are at the root of the directory tree
                if self.op_file_manager.current_path == self.external_storage:
                    self.show_toast_msg(f"Closing file manager from main storage")
                    self.op_file_exit_manager()
                else:
                    self.op_file_manager.back()  # Navigate back within file manager
                return True
        return False

if __name__ == '__main__':
    VisionAiApp().run()
