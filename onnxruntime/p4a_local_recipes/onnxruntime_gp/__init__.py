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

        # Make sure cross python/protoc are visible
        env["PYTHON_EXECUTABLE"] = self.ctx.hostpython
        env["ONNX_CUSTOM_PROTOC_EXECUTABLE"] = sh.which("protoc")
        print(f"Host Python: {self.ctx.hostpython}")
        return env

    def build_arch(self, arch):
        super().build_arch(arch)

        env = self.get_recipe_env(arch)

        build_dir = self.get_build_dir(arch.arch)
        cmake_dir = join(self.get_build_dir(arch.arch), "cmake")
        python_site_packages = self.ctx.get_site_packages_dir(arch)
        python_include_numpy = join(python_site_packages,
                                    'numpy', 'core', 'include')
        self.apply_patches(arch)

        cmake_args = [
            "cmake",
            cmake_dir,
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

        shprint(sh.mkdir, "-p", build_dir)
        with current_directory(build_dir):
            shprint(sh.Command("cmake"), *cmake_args, _env=env)
            shprint(sh.Command("cmake"), "--build", ".", _env=env)

            print("Listing the cmake directory:")
            shprint(sh.ls, "-R", cmake_dir)
            print("Listing the build directory:")
            shprint(sh.ls, "-R", build_dir)
            # Build wheel
            hostpython = sh.Command(self.ctx.hostpython)
            print(f"Host Python: {self.ctx.hostpython}")
            hostpython = sh.Command(self.hostpython_location)
            print(f"Host Python: {self.hostpython_location}")
            shprint(hostpython, "-m", "build", "--wheel", "--no-isolation", _env=env)

            # Install wheel into target python site-packages
            pyver = self.ctx.python_recipe.version[0:4].replace(".", "")
            whl_pattern = f"onnxruntime-{self.version}-cp{pyver}-*.whl"

            shprint(
                sh.Command(hostpython),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--prefix",
                '--target',
                self.ctx.get_python_install_dir(arch.arch),
                join("dist", whl_pattern),
                _env=env,
            )


recipe = OnnxRuntimeRecipe()
