# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread
import requests

from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.clock import Clock

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
from screens.img_obj_classify import ImgClassifytBox
from screens.img_species import ImgSpeciesBox
from screens.setting import SettingsBox
from onnx_detect import OnnxDetect
from onnx_classify import OnnxClassify
from onnx_species import OnnxSpecies

## Global definitions
__version__ = "0.2.0" # The APP version

detect_model_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
classify_model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
species_model_url = "https://github.com/daslearning-org/vision-ai/releases/download/vOnnxModels/spicesNet_v401a.onnx"
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
    is_detect_running = ObjectProperty(None)
    is_downloading = ObjectProperty(None)
    is_classify_running = ObjectProperty(None)
    is_species_running = ObjectProperty(None)
    image_path = StringProperty("")
    op_img_path = StringProperty("")
    onnx_detect = ObjectProperty(None)
    onnx_detect_sess = ObjectProperty(None)
    onnx_classify = ObjectProperty(None)
    onnx_classify_sess = ObjectProperty(None)
    onnx_species = ObjectProperty(None)
    onnx_species_sess = ObjectProperty(None)
    cam_found = ObjectProperty(None)
    camera = ObjectProperty(None)
    detect_model_path = StringProperty("")

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
        self.detect_model_path = os.path.join(self.model_dir, "ssd_mobilenet_v1_10.onnx")
        self.classify_model_path = os.path.join(self.model_dir, "resnet18-v1-7.onnx")
        self.species_model_path = os.path.join(self.model_dir, "spicesNet_v401a.onnx")

        # file managers
        self.img_preview = False
        self.is_img_manager_open = False
        self.img_file_manager = MDFileManager(
            exit_manager=self.img_file_exit_manager,
            select_path=self.select_img_path,
            ext=[".png", ".jpg", ".jpeg", "webp"],  # Restrict to image files
            selector="file",  # Restrict to selecting files only
            preview=False,
            #show_hidden_files=True,
        )
        self.is_op_file_mgr_open = False
        self.op_file_manager = MDFileManager(
            exit_manager=self.op_file_exit_manager,
            select_path=self.select_op_path,
            selector="folder",  # Restrict to selecting directories only
        )

        if not os.path.exists(self.detect_model_path):
            self.popup_detect_model()

        # create onnx objects
        self.onnx_detect = OnnxDetect(
            save_dir=self.op_dir,
            model_dir=self.model_dir,
        )

        print("Initialisation is successfull")

    def update_download_progress(self, downloaded, total_size):
        if total_size > 0:
            percentage = (downloaded / total_size) * 100
            self.download_progress.text = f"Progress: {percentage:.1f}%"
        else:
            self.download_progress.text = f"Progress: {downloaded} bytes"

    def download_file(self, download_url, download_path):
        filename = download_url.split("/")[-1]
        try:
            self.is_downloading = filename
            with requests.get(download_url, stream=True) as req:
                req.raise_for_status()
                total_size = int(req.headers.get('content-length', 0))
                downloaded = 0
                with open(download_path, 'wb') as f:
                    for chunk in req.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            Clock.schedule_once(lambda dt: self.update_download_progress(downloaded, total_size))
            if os.path.exists(download_path):
                Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download complete: {download_path}"))
            else:
                Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
            self.is_downloading = False
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the onnx file: {e} 😞")
            Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
            self.is_downloading = False

    def download_model_file(self, model_url, download_path, instance=None):
        self.txt_dialog_closer(instance)
        filename = download_path.split("/")[-1]
        print(f"Starting the download for: {filename}")
        if self.root.ids.screen_manager.current == "imgObjDetect":
            result_box = self.root.ids.img_detect_box.ids.result_image
        elif self.root.ids.screen_manager.current == "imgClassify":
            result_box = self.root.ids.img_classify_box.ids.result_label
        elif self.root.ids.screen_manager.current == "imgSpecies":
            result_box = self.root.ids.img_species_box.ids.result_label
        else:
            result_box = self.root.ids.cam_detect_box.ids.cam_result_image
        result_box.clear_widgets()
        self.download_progress = MDLabel(
            text="Progress: 0%",
            halign="center"
        )
        result_box.add_widget(self.download_progress)
        Thread(target=self.download_file, args=(model_url, download_path), daemon=True).start()

    def download_detect_model(self, instance):
        self.download_model_file(detect_model_url, self.detect_model_path, instance)

    def download_classify_model(self, instance):
        self.download_model_file(classify_model_url, self.classify_model_path, instance)

    def download_species_model(self, instance):
        self.download_model_file(species_model_url, self.species_model_path, instance)

    def popup_detect_model(self):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Ok",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.download_detect_model
            ),
        ]
        self.show_text_dialog(
            "Downlaod the model file",
            f"You need to downlaod the file for the first time (~30MB)",
            buttons
        )

    def popup_classify_model(self):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Ok",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.download_classify_model
            ),
        ]
        self.show_text_dialog(
            "Downlaod the model file",
            f"You need to downlaod the file for the first time (~45MB)",
            buttons
        )

    def popup_species_model(self):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Ok",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.download_species_model
            ),
        ]
        self.show_text_dialog(
            "Downlaod the model file",
            f"You need to downlaod the file for the first time (~215MB)",
            buttons
        )

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
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.popup_detect_model()
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

    def on_img_classify(self):
        if not os.path.exists(self.classify_model_path) and self.is_downloading != "resnet18-v1-7.onnx":
            self.popup_classify_model()
        self.show_toast_msg("Select an image & get top 5 predictions")
        if not self.onnx_classify:
            self.onnx_classify = OnnxClassify(
                save_dir=self.op_dir,
                model_dir=self.model_dir,
            )

    def on_img_species(self):
        if not os.path.exists(self.species_model_path) and self.is_downloading != "spicesNet_v401a.onnx":
            self.popup_species_model()
        self.show_toast_msg("Select an image & identify the Species from more than 2000 species list")
        if not self.onnx_species:
            self.onnx_species = OnnxSpecies(
                save_dir=self.op_dir,
                model_dir=self.model_dir,
            )

    def on_cam_obj_dt_leave(self):
        if self.cam_found:
            self.camera.play = False
            self.cam_uix.clear_widgets()

    def open_img_file_manager(self):
        """Open the file manager to select an image file. On android use Downloads or Pictures folders only"""
        if self.is_downloading == "ssd_mobilenet_v1_10.onnx":
            self.show_toast_msg("Please wait for the model download to finish!", is_error=True)
            return
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.onnx_detect_sess = False
            self.popup_detect_model()
            return
        if not self.onnx_detect_sess:
            self.onnx_detect_sess = self.onnx_detect.start_detect_session()
        if self.is_detect_running:
            self.show_toast_msg("Please wait for the current operation to finish", is_error=True)
            return
        try:
            self.img_file_manager.show(self.external_storage)  # native app specific path
            self.is_img_manager_open = True
        except Exception as e:
            self.show_toast_msg(f"Error: {e}", is_error=True)

    def open_clsfy_img_file(self):
        """Open the file manager to select an image file. On android use Downloads or Pictures folders only"""
        if self.is_downloading == "resnet18-v1-7.onnx":
            self.show_toast_msg("Please wait for the model download to finish!", is_error=True)
            return
        if not os.path.exists(self.classify_model_path) and self.is_downloading != "resnet18-v1-7.onnx":
            self.onnx_classify_sess = False
            self.popup_classify_model()
            return
        if not self.onnx_classify_sess: # update it
            self.onnx_classify_sess = self.onnx_classify.start_classify_session()
        if self.is_classify_running:
            self.show_toast_msg("Please wait for the current operation to finish", is_error=True)
            return
        try:
            self.img_file_manager.show(self.external_storage)
            self.is_img_manager_open = True
        except Exception as e:
            self.show_toast_msg(f"Error: {e}", is_error=True)

    def open_spcnt_img_file(self):
        """Open the file manager to select an image file. On android use Downloads or Pictures folders only"""
        if self.is_downloading == "spicesNet_v401a.onnx":
            self.show_toast_msg("Please wait for the model download to finish!", is_error=True)
            return
        if not os.path.exists(self.species_model_path) and self.is_downloading != "spicesNet_v401a.onnx":
            self.onnx_species_sess = False
            self.popup_species_model()
            return
        if not self.onnx_species_sess: # update it
            self.onnx_species_sess = self.onnx_species.start_species_session()
        if self.is_species_running:
            self.show_toast_msg("Please wait for the current operation to finish", is_error=True)
            return
        try:
            self.img_file_manager.show(self.external_storage)
            self.is_img_manager_open = True
        except Exception as e:
            self.show_toast_msg(f"Error: {e}", is_error=True)

    def select_img_path(self, path: str):
        self.image_path = path
        if self.root.ids.screen_manager.current == "imgObjDetect":
            uploaded_image_box = self.root.ids.img_detect_box.ids.uploaded_image
        elif self.root.ids.screen_manager.current == "imgSpecies":
            uploaded_image_box = self.root.ids.img_species_box.ids.uploaded_image
        else:
            uploaded_image_box = self.root.ids.img_classify_box.ids.uploaded_image
        uploaded_image_box.clear_widgets()
        fitImage = Image(
            source = path,
            fit_mode = "contain"
        )
        uploaded_image_box.add_widget(fitImage)
        if self.root.ids.screen_manager.current == "imgObjDetect":
            result_box = self.root.ids.img_detect_box.ids.result_image
        elif self.root.ids.screen_manager.current == "imgSpecies":
            result_box = self.root.ids.img_species_box.ids.result_label
        else:
            result_box = self.root.ids.img_classify_box.ids.result_label
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
        if not self.onnx_detect_sess:
            self.onnx_detect_sess = self.onnx_detect.start_detect_session()
            self.submit_onnx_detect()
        if self.is_detect_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        onnx_thread = Thread(target=self.onnx_detect.run_detect, args=(self.image_path, self.onnx_detect_callback, "imgObjDetect"), daemon=True)
        onnx_thread.start()
        self.is_detect_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.img_detect_box.ids.result_image
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def capture_n_onnx_detect(self):
        if not self.cam_found:
            self.show_toast_msg("Camera could not be loaded!", is_error=True)
            return
        if self.is_downloading == "ssd_mobilenet_v1_10.onnx":
            self.show_toast_msg("Please wait for the model download to finish!", is_error=True)
            return
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.onnx_detect_sess = False
            self.popup_detect_model()
            return
        if not self.onnx_detect_sess:
            self.onnx_detect_sess = self.onnx_detect.start_detect_session()
            self.capture_n_onnx_detect()
        if self.is_detect_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        self.image_path = ""
        import datetime
        now = datetime.datetime.now()
        current_time = str(now.strftime("%H%M%S"))
        current_date = str(now.strftime("%Y%m%d"))
        capture_file = f"cam-{current_date}-{current_time}.png"
        self.image_path = os.path.join(self.op_dir, capture_file)
        self.camera.export_to_png(self.image_path)
        onnx_thread = Thread(target=self.onnx_detect.run_detect, args=(self.image_path, self.onnx_detect_callback, "camObjDetect"), daemon=True)
        onnx_thread.start()
        self.is_detect_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.cam_detect_box.ids.cam_result_image
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def submit_onnx_classify(self):
        if self.image_path == "":
            self.show_toast_msg("No image is selected!", is_error=True)
            return
        if not self.onnx_classify_sess:
            self.onnx_classify_sess = self.onnx_classify.start_classify_session()
            self.submit_onnx_classify()
        if self.is_classify_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        onnx_thread = Thread(target=self.onnx_classify.run_classify, args=(self.image_path, self.onnx_classify_callback, "imgClassify"), daemon=True)
        onnx_thread.start()
        self.is_classify_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.img_classify_box.ids.result_label
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def submit_onnx_species(self):
        if self.image_path == "":
            self.show_toast_msg("No image is selected!", is_error=True)
            return
        if not self.onnx_classify_sess:
            self.onnx_classify_sess = self.onnx_classify.start_classify_session()
            self.submit_onnx_species()
        if self.is_species_running:
            self.show_toast_msg("Please wait for the previous request to finish", is_error=True)
            return
        onnx_thread = Thread(target=self.onnx_species.run_species, args=(self.image_path, self.onnx_classify_callback, "imgSpecies"), daemon=True)
        onnx_thread.start()
        self.is_species_running = True
        tmp_spin = TempSpinWait()
        result_box = self.root.ids.img_species_box.ids.result_label
        result_box.clear_widgets()
        result_box.add_widget(tmp_spin)

    def onnx_detect_callback(self, onnx_resp):
        status = onnx_resp["status"]
        message = onnx_resp["message"]
        caller = onnx_resp["caller"]
        self.is_detect_running = False
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

    def onnx_classify_callback(self, onnx_resp):
        status = onnx_resp["status"]
        message = onnx_resp["message"]
        caller = onnx_resp["caller"]
        self.is_classify_running = False
        result_box = self.root.ids.img_classify_box.ids.result_label
        result_box.clear_widgets()
        if status is True:
            classify_label = MDLabel(
                text=message,
                halign="left",
                valign="top",
                markup=True
            )
            result_box.add_widget(classify_label)
        else:
            self.show_toast_msg(message, is_error=True)

    def onnx_species_callback(self, onnx_resp):
        status = onnx_resp["status"]
        message = onnx_resp["message"]
        caller = onnx_resp["caller"]
        self.is_species_running = False
        result_box = self.root.ids.img_species_box.ids.result_label
        result_box.clear_widgets()
        if status is True:
            classify_label = MDLabel(
                text=message,
                halign="left",
                valign="top",
                markup=True
            )
            result_box.add_widget(classify_label)
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

    def reset_classify(self):
        self.image_path = ""
        uploaded_image_box = self.root.ids.img_classify_box.ids.uploaded_image
        uploaded_image_box.clear_widgets()
        result_box = self.root.ids.img_classify_box.ids.result_label
        result_box.clear_widgets()

    def reset_species(self):
        self.image_path = ""
        uploaded_image_box = self.root.ids.img_species_box.ids.uploaded_image
        uploaded_image_box.clear_widgets()
        result_box = self.root.ids.img_species_box.ids.result_label
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

    def img_preview_on(self):
        print("Switch pressed...")
        img_preview_sw = self.root.ids.settings_box.ids.img_preview_switch
        if self.img_preview:
            img_preview_sw.icon = "toggle-switch-off"
            img_preview_sw.text_color = "gray"
            self.img_file_manager.preview = False
            self.img_preview = False
        else:
            img_preview_sw.icon = "toggle-switch"
            img_preview_sw.text_color = "green"
            self.img_file_manager.preview = True
            self.img_preview = True

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
