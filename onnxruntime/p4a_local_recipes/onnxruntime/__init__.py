from pythonforandroid.recipe import Recipe, CythonRecipe
from pythonforandroid.util import current_directory
from pythonforandroid.logger import shprint, info
from os.path import join
import shutil
import glob

class OnnxruntimeRecipe(CythonRecipe):
    version = '1.21.1'
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{version}.zip"
    name = 'onnxruntime'
    depends = ["setuptools", "wheel", "numpy", "protobuf", "libc++"]
    python_depends = ["numpy"]
    build_in_src = True

    def get_recipe_env(self, arch=None, with_flags_in_cc=True):
        env = super().get_recipe_env(arch, with_flags_in_cc)
        env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

        # Make sure cross python/protoc are visible
        env["PYTHON_EXECUTABLE"] = sh.which("python3")
        env["ONNX_CUSTOM_PROTOC_EXECUTABLE"] = sh.which("protoc")
        return env

    def build_arch(self, arch):
        super().build_arch(arch)
        env = self.get_recipe_env(arch)
        build_dir = self.get_build_dir(arch.arch)
        info(f'Building onnxruntime for {arch}')

        with current_directory(build_dir):
            # Setup CMake arguments based on Termux script and Android requirements
            python_site_packages = self.ctx.get_site_packages_dir(arch)
            python_include_numpy = join(python_site_packages,
                                        'numpy', 'core', 'include')

            cmake_args = [
            "cmake",
            "..",
            f"-DCMAKE_TOOLCHAIN_FILE={self.ctx.ndk_dir}/build/cmake/android.toolchain.cmake",
            f"-DANDROID_ABI={arch.arch}",
            f"-DANDROID_NATIVE_API_LEVEL={self.ctx.ndk_api}",
            "-Donnxruntime_ENABLE_PYTHON=ON",
            "-Donnxruntime_BUILD_SHARED_LIB=OFF",
            "-DPYBIND11_USE_CROSSCOMPILING=TRUE",
            "-Donnxruntime_USE_NNAPI_BUILTIN=ON",
            "-Donnxruntime_USE_XNNPACK=ON",
            f"-DPython_NumPy_INCLUDE_DIR={python_include_numpy}",
        ]

            # Configure CMake with source in 'cmake' subdir
            cmake_source_dir = join(build_dir, 'cmake')
            shprint(shutil.cmake, *cmake_args, _cwd=build_dir, _env=env, _tail=20, _critical=True)

            # Build the native parts
            shprint(shutil.make, '-j', str(self.ctx.concurrent_make_level()), _env=env)  # Or shutil.ninja if using Ninja

            # Build the Python wheel
            shprint(shutil.python, '-m', 'build', '--wheel', '--no-isolation', _env=env)

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
        hostpython = shutil.Command(self.ctx.hostpython)
        shprint(hostpython, '-m', 'pip', 'install', '--target', site_packages_dir, '--no-deps', wheel_path, _env=env)

recipe = OnnxruntimeRecipe()