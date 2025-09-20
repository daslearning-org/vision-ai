from pythonforandroid.recipe import Recipe
from os.path import join, exists, dirname, realpath
import shutil
from pythonforandroid.logger import info
import os


class OnnxruntimeRecipe(Recipe):
    name = 'onnxruntime'
    version = '1.22.1'
    url = None
    depends = ['python3', 'numpy']

    def get_recipe_dir(self):
        return dirname(realpath(__file__))

    def build_arch(self, arch):
        super().build_arch(arch)

        recipe_dir = self.get_recipe_dir()
        src = join(recipe_dir, 'onnxruntime_termux')
        dst = join(self.ctx.get_python_install_dir(arch.arch), 'onnxruntime')

        # copy onnxruntime packages
        info(f"Copying onnxruntime package from {src} to {dst}")
        if exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        # Now copy libprotobuf-lite.so
        recipe_libs_dir = join(self.get_recipe_dir(), 'binaries')
        libs_dir = join(self.ctx.get_libs_dir(arch.arch))
        for loop_so_file in os.listdir(recipe_libs_dir):
            loop_src = join(recipe_libs_dir, loop_so_file)
            loop_dst = join(libs_dir, loop_so_file)
            info(f"Copying {loop_so_file} to {libs_dir}")
            shutil.copy2(loop_src, loop_dst)

        from glob import glob
        import subprocess
        # Patch .so file
        so_file = join(dst, 'capi', 'onnxruntime_pybind11_state.so')
        if exists(so_file):
            info(f"Patching {so_file}: libpython3.11.so.1.0 → libpython3.11.so")
            subprocess.run(['patchelf', '--replace-needed', 'libpython3.11.so.1.0', 'libpython3.11.so', so_file], check=True)
        else:
            info(f"{so_file} not found for patching")


recipe = OnnxruntimeRecipe()
