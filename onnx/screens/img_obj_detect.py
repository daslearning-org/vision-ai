from kivy.lang import Builder
from kivy.metrics import dp, sp

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDFillRoundFlatIconButton


Builder.load_string('''
<TempSpinWait>:
    id: temp_spin
    orientation: 'horizontal'
    adaptive_height: True
    padding: dp(8)

    MDLabel:
        text: "Please wait..."
        font_style: "Subtitle1"
        adaptive_width: True

    MDSpinner:
        size_hint: None, None
        size: dp(14), dp(14)
        active: True


<ImgObjDetBox@MDBoxLayout>:
    orientation: 'vertical'
    spacing: dp(4)

    BoxLayout: # original image
        size_hint_y: 0.4
        id: uploaded_image
        #adaptive_height: True
        # add fit image here

    MDGridLayout: # buttons
        cols: 3
        size_hint_y: 0.1
        spacing: dp(4)
        padding: 14, 0, 14, 0 # left, top, right, bottom

        MDFillRoundFlatIconButton:
            id: btn_upload
            text: "Image"
            icon: "upload"
            font_size: sp(18)
            #md_bg_color: '#333036'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.2
            on_release: app.open_img_file_manager()

        MDFillRoundFlatIconButton:
            id: btn_submit
            text: "Submit"
            icon: "send"
            font_size: sp(18)
            md_bg_color: 'orange'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.6
            on_release: app.submit_onnx_detect()

        MDFillRoundFlatIconButton:
            id: btn_reset
            text: "Reset"
            icon: "undo-variant"
            font_size: sp(18)
            md_bg_color: '#333036'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.2
            on_release: app.reset_object_detect()

    BoxLayout: # converted image
        size_hint_y: 0.5
        id: result_image
        # add result here


''')

class TempSpinWait(MDBoxLayout):
    pass

class ImgObjDetBox(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "img_detect_bx"
