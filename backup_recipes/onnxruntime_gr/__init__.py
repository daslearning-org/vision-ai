from pythonforandroid.recipe import Recipe
from pythonforandroid.util import current_directory
from pythonforandroid.logger import shprint, info
from os.path import join
import sh
import glob

class OnnxruntimeRecipe(Recipe):
    version = '1.17.0'
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{version}.zip"
    name = 'onnxruntime'
    depends = ['numpy', 'setuptools', 'packaging', 'wheel', 'pybind11', 'protobuf', 'abseil-cpp', 're2']  # Note: You may need to create recipes for 'abseil-cpp' and 're2' if they don't exist in p4a. Use the Termux packages as a guide.
    conflicts = []  # If any conflicts

    def post_download(self):
        super().post_download()
        # Checkout the specific version tag
        with current_directory(self.get_build_dir('any')):
            shprint(sh.git, 'checkout', f'v{self.version}')

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
                f'-DPython_NumPy_INCLUDE_DIR={self.get_recipe("numpy", self.ctx).get_numpy_include_dir()}',
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
        py_ver = self.ctx.python_recipe.major_minor_version_string.replace('.', '')
        arch_tag = arch.arch.replace('arm64-v8a', 'aarch64').replace('armeabi-v7a', 'armv7l')  # Map to Linux wheel tags
        wheel_pattern = join(build_dir, 'dist', f'onnxruntime-{self.version}-cp{py_ver}-cp{py_ver}-linux_{arch_tag}.whl')
        wheels = glob.glob(wheel_pattern)
        if not wheels:
            raise Exception('No wheel found matching pattern')
        wheel_path = wheels[0]

        # Install the wheel to site's packages
        site_packages_dir = self.ctx.get_site_packages_dir(arch)
        hostpython = sh.Command(self.ctx.hostpython)
        shprint(hostpython, '-m', 'pip', 'install', '--target', site_packages_dir, '--no-deps', wheel_path, _env=env)

    def get_numpy_include_dir(self, arch):
        # If numpy recipe doesn't have this method, implement it here or adjust
        return join(self.get_recipe('numpy', self.ctx).get_build_dir(arch.arch), 'numpy', 'core', 'include')

recipe = OnnxruntimeRecipe()