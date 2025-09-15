from pythonforandroid.recipe import CythonRecipe, Recipe
from pythonforandroid.toolchain import current_directory, shprint
from os.path import join
import sh


class OnnxRuntimeRecipe(Recipe):
    version = "1.22.1"
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{version}.tar.gz"

    depends = ["setuptools", "wheel", "numpy", "protobuf"]

    # Build in source like your Termux build
    build_in_src = True

    def get_recipe_env(self, arch=None, with_flags_in_cc=True):
        env = super().get_recipe_env(arch, with_flags_in_cc)
        env['CPPFLAGS'] += ' -Wno-unused-variable'

        # Make sure cross python/protoc are visible
        print(f"Host Python: {self.ctx.hostpython}")
        print(f"NDK Dir: {self.ctx.ndk_dir}")
        return env

    def build_arch(self, arch):
        super().build_arch(arch)

        env = self.get_recipe_env(arch)

        build_dir = self.get_build_dir(arch.arch)
        with_build = join(build_dir, "build")
        print(f"Build dir: {build_dir}")
        cmake_dir = join(build_dir, "cmake")
        capi_dir = join(build_dir, "onnxruntime", "capi")
        dist_dir = join(build_dir, "dist")
        numpy_build_dir = Recipe.get_recipe("numpy", self.ctx).get_build_dir(arch.arch)
        print(f"Numpy build dir: {numpy_build_dir}")
        #python_include_numpy = join(numpy_site_packages, 'numpy', 'core', 'include')
        python_site_packages = self.ctx.get_site_packages_dir(arch)
        python_include_numpy = join(python_site_packages,
                                        'numpy', 'core', 'include')
        print(f"Python include numpy: {python_include_numpy}")
        toolchain_file = join(self.ctx.ndk_dir,
                                'build/cmake/android.toolchain.cmake')
        protoc_path = sh.which("protoc")
        python_path = self.ctx.hostpython
        shprint(sh.mkdir, "-p", with_build)
        shprint(sh.mkdir, "-p", capi_dir)
        shprint(sh.mkdir, "-p", dist_dir)

        cmake_args = [
            "cmake",
            cmake_dir,
            f"-DCMAKE_INSTALL_PREFIX={build_dir}",
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
            f"-DANDROID_ABI={arch.arch}",
            #"-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-Donnxruntime_ENABLE_PYTHON=ON",
            "-Donnxruntime_BUILD_SHARED_LIB=OFF",
            "-DPYBIND11_USE_CROSSCOMPILING=TRUE",
            "-Donnxruntime_USE_NNAPI_BUILTIN=ON",
            "-Donnxruntime_USE_XNNPACK=ON",
            f"-DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc",
            f"-DPython_NumPy_INCLUDE_DIR={python_include_numpy}",
            #f"-DPYTHON_NUMPY_INCLUDE_DIRS={python_include_numpy}",
            f"-DPYTHON_EXECUTABLE={python_path}",
            #"-DPython_NumPy_FOUND=TRUE",
            "-DCMAKE_BUILD_TYPE=RELEASE",
        ]


        with current_directory(build_dir):
            shprint(sh.Command("cmake"), *cmake_args, _env=env)
            #shprint(sh.make, '-j' + str(cpu_count()), _env=env)
            #shprint(sh.make, 'install', _env=env)
            shprint(sh.Command("cmake"), "--build", ".", _env=env)
            shprint(sh.Command("cmake"), "--install", ".", _env=env)

            print("=== Installed capi contents ===")
            shprint(sh.ls, "-R", capi_dir)
            # Build wheel
            hostpython = sh.Command(self.ctx.hostpython)
            print(f"Host Python: {self.ctx.hostpython}")
            hostpython = sh.Command(self.hostpython_location)
            print(f"Host Python: {self.hostpython_location}")
            #shprint(python_path, "-m", "build", "--wheel", "--no-isolation", _env=env)
            shprint("tree", build_dir)
            # Install wheel into target python site-packages


recipe = OnnxRuntimeRecipe()
