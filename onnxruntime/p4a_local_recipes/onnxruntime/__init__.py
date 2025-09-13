from pythonforandroid.recipe import Recipe
from pythonforandroid.util import current_directory
from pythonforandroid.logger import shprint, info
from os.path import join
import sh
import glob

class OnnxruntimeRecipe(Recipe):
    version = '1.21.1'
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{version}.zip"
    name = 'onnxruntime'
    depends = ['numpy', 'setuptools', 'packaging', 'wheel', 'pybind11', 'protobuf']  # Note: You may need to create recipes for 'abseil-cpp' and 're2' if they don't exist in p4a. Use the Termux packages as a guide.
    conflicts = []  # If any conflicts

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['CPPFLAGS'] += ' -Wno-unused-variable'
        # Add any other necessary environment variables here
        return env

    def build_arch(self, arch):
        env = self.get_recipe_env(arch)
        build_dir = self.get_build_dir(arch.arch)
        info(f'Building onnxruntime for {arch}')

        with current_directory(build_dir):
            python_site_packages = self.ctx.get_site_packages_dir(arch)
            python_include_numpy = join(python_site_packages,
                                        'numpy', 'core', 'include')
            # Setup CMake arguments based on Termux script and Android requirements
            cmake_args = [
                '.',  # Build in current dir
                f'-DCMAKE_TOOLCHAIN_FILE={self.ctx.ndk.cmake_toolchain_path}',
                f'-DANDROID_ABI={arch.arch}',
                f'-DANDROID_PLATFORM=android-{self.ctx.android_api}',
                '-DANDROID_STL=c++_shared',
                '-DANDROID_CPP_FEATURES=exceptions rtti',
                '-DCMAKE_POLICY_DEFAULT_CMP0057=NEW',  # If needed for policy
                '-Donnxruntime_ENABLE_PYTHON=ON',
                '-Donnxruntime_BUILD_SHARED_LIB=OFF',
                '-DPYBIND11_USE_CROSSCOMPILING=TRUE',
                '-Donnxruntime_USE_NNAPI_BUILTIN=ON',
                '-Donnxruntime_USE_XNNPACK=ON',
                f'-DPYTHON_EXECUTABLE={self.ctx.hostpython}',
                # Path to protoc from protobuf recipe - adjust if necessary
                f'-DONNX_CUSTOM_PROTOC_EXECUTABLE={join(self.get_recipe("protobuf", self.ctx).get_build_dir(arch.arch), "protoc")}',
                # Numpy include dir
                f'-DPython_NumPy_INCLUDE_DIR={python_include_numpy}',
                '-G', 'Unix Makefiles'  # Use Makefiles; if you want Ninja, ensure ninja is available and change to '-G Ninja'
            ]

            # Configure CMake with source in 'cmake' subdir
            cmake_source_dir = join(build_dir, 'cmake')
            shprint(sh.cmake, *cmake_args, _cwd=build_dir, _env=env, _tail=20, _critical=True)

            # Build the native parts
            shprint(sh.make, '-j', str(self.ctx.concurrent_make_level()), _env=env)  # Or sh.ninja if using Ninja

            # Build the Python wheel
            shprint(sh.python, '-m', 'build', '--wheel', '--no-isolation', _env=env)

        # Find the built wheel (adjust the pattern if necessary)
        pyver = self.ctx.python_recipe.version[0:4].replace(".", "")
        whl_pattern = f"onnxruntime-{self.version}-cp{pyver}-*.whl"
        wheels = glob.glob(whl_pattern)
        if not wheels:
            raise Exception('No wheel found matching pattern')
        wheel_path = wheels[0]

        # Install the wheel to site's packages
        site_packages_dir = self.ctx.get_site_packages_dir(arch)
        hostpython = sh.Command(self.ctx.hostpython)
        shprint(hostpython, '-m', 'pip', 'install', '--target', site_packages_dir, '--no-deps', wheel_path, _env=env)

recipe = OnnxruntimeRecipe()