from pythonforandroid.recipe import CythonRecipe
from pythonforandroid.toolchain import current_directory, shprint
from os.path import join
import sh


class OnnxRuntimeRecipe(CythonRecipe):
    version = "1.22.1"
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{version}.tar.gz"

    depends = ["setuptools", "wheel", "numpy", "protobuf"]
    #python_depends = ["numpy"]

    # Build in source like your Termux build
    build_in_src = True

    def get_recipe_env(self, arch=None, with_flags_in_cc=True):
        env = super().get_recipe_env(arch, with_flags_in_cc)
        env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
        env['CPPFLAGS'] += ' -Wno-unused-variable'

        # Make sure cross python/protoc are visible
        env["PYTHON_EXECUTABLE"] = sh.which("python3")
        env["ONNX_CUSTOM_PROTOC_EXECUTABLE"] = sh.which("protoc")
        print(f"Host Python: {self.ctx.hostpython}")
        print(f"NDK Dir: {self.ctx.ndk_dir}")
        return env

    def build_arch(self, arch):
        super().build_arch(arch)

        env = self.get_recipe_env(arch)

        build_dir = self.get_build_dir(arch.arch)
        cmake_dir = join(build_dir, "cmake")
        capi_dir = join(build_dir, "capi")
        dist_dir = join(build_dir, "dist")
        python_site_packages = self.ctx.get_site_packages_dir(arch)
        python_include_numpy = join(python_site_packages, 'numpy', 'core', 'include')
        protoc_path = sh.which("protoc")
        python_path = sh.which("python")
        shprint(sh.mkdir, "-p", capi_dir)

        cmake_args = [
            "cmake",
            cmake_dir,
            f"-DCMAKE_INSTALL_PREFIX={capi_dir}",
            f"-DCMAKE_TOOLCHAIN_FILE={self.ctx.ndk_dir}/build/cmake/android.toolchain.cmake",
            f"-DANDROID_ABI={arch.arch}",
            f"-DANDROID_NATIVE_API_LEVEL={self.ctx.ndk_api}",
            "-Donnxruntime_ENABLE_PYTHON=ON",
            "-Donnxruntime_BUILD_SHARED_LIB=OFF",
            "-DPYBIND11_USE_CROSSCOMPILING=TRUE",
            "-Donnxruntime_USE_NNAPI_BUILTIN=ON",
            "-Donnxruntime_USE_XNNPACK=ON",
            f"-DPYTHON_EXECUTABLE={python_path}",
            f"-DONNX_CUSTOM_PROTOC_EXECUTABLE={protoc_path}",
            f"-DPython_NumPy_INCLUDE_DIR={python_include_numpy}",
            "-DCMAKE_BUILD_TYPE=RELEASE"
        ]


        with current_directory(build_dir):
            shprint(sh.Command("cmake"), *cmake_args, _env=env)
            shprint(sh.Command("cmake"), "--build", ".", "--target", "generate_build_and_package_info", _env=env)
            shprint(sh.Command("cmake"), "--build", ".", _env=env)
            shprint(sh.Command("cmake"), "--install", ".", _env=env)
            shprint("tree", build_dir)

            print("Listing the cmake directory:")
            shprint(sh.ls, "-R", cmake_dir)
            print("Listing the build directory:")
            shprint(sh.ls, "-R", build_dir)
            # Build wheel
            hostpython = sh.Command(self.ctx.hostpython)
            print(f"Host Python: {self.ctx.hostpython}")
            hostpython = sh.Command(self.hostpython_location)
            print(f"Host Python: {self.hostpython_location}")
            shprint(python_path, "-m", "build", "--wheel", "--no-isolation", _env=env)

            # Install wheel into target python site-packages
            pyver = self.ctx.python_recipe.version[0:4].replace(".", "")
            whl_pattern = f"onnxruntime-{self.version}-cp{pyver}-*.whl"
            wheels = glob.glob(whl_pattern)
            if not wheels:
                raise Exception('No wheel found matching pattern')

            shprint(
                sh.Command(python_path),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--prefix",
                '--target',
                self.ctx.get_python_install_dir(arch.arch),
                join(dist_dir, whl_pattern),
                _env=env,
            )


recipe = OnnxRuntimeRecipe()
