from kivy.lang import Builder
from kivy.metrics import dp, sp

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDFillRoundFlatIconButton


Builder.load_string('''

<ImgSpeciesBox@MDBoxLayout>:
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
            id: btn_spc_upload
            text: "Image"
            icon: "upload"
            font_size: sp(18)
            #md_bg_color: '#333036'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.2
            on_release: app.open_spcnt_img_file()

        MDFillRoundFlatIconButton:
            id: btn_submit
            text: "Identify"
            icon: "send"
            font_size: sp(18)
            md_bg_color: 'orange'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.6
            on_release: app.submit_onnx_species()

        MDFillRoundFlatIconButton:
            id: btn_reset
            text: "Reset"
            icon: "undo-variant"
            font_size: sp(18)
            md_bg_color: '#333036'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.2
            on_release: app.reset_species()

    BoxLayout: # converted image
        size_hint_y: 0.5
        id: result_label
        padding: dp(20)
        # add result here

''')

class ImgSpeciesBox(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "img_species_bx"
