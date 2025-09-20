from pythonforandroid.recipe import CythonRecipe, Recipe
from pythonforandroid.toolchain import current_directory, shprint
from os.path import join, exists
import sh, sys
import shutil
from multiprocessing import cpu_count

class OnnxRuntimeRecipe(Recipe):
    version = "1.22.1"
    url = "https://github.com/microsoft/onnxruntime/archive/refs/tags/v{version}.tar.gz"

    depends = ["setuptools", "wheel", "numpy", "protobuf", "pybind11"]
    patches = [
        'patches/onnx_numpy.patch',
        'patches/mlasi_bfloat.patch',
    ]
    # Build in source like your Termux build
    build_in_src = True

    def get_recipe_env(self, arch=None, with_flags_in_cc=True):
        env = super().get_recipe_env(arch, with_flags_in_cc)
        python_include_dir = self.ctx.python_recipe.include_root(arch.arch)
        print(f"Python include dir: {python_include_dir}")
        env['CPPFLAGS'] += f' -Wno-unused-variable -I{python_include_dir}'
        env['CXXFLAGS'] += f' -I{python_include_dir}'
        env['CFLAGS'] += f' -I{python_include_dir}'
        env["Python_INCLUDE_DIRS"] = python_include_dir

        # Make sure cross python/protoc are visible
        print(f"Host Python: {self.ctx.hostpython}")
        print(f"NDK Dir: {self.ctx.ndk_dir}")
        return env

    def build_arch(self, arch):
        super().build_arch(arch)

        env = self.get_recipe_env(arch)
        ANDROID_PLATFORM = str(self.ctx.ndk_api)

        build_dir = self.get_build_dir(arch.arch)
        with_build = join(build_dir, "build")
        print(f"Build dir: {build_dir}")
        cmake_dir = join(build_dir, "cmake")
        capi_dir = join(build_dir, "onnxruntime", "capi")
        dist_dir = join(build_dir, "dist")
        onnx_pybind_dir = join(build_dir, "_deps", "pybind11_project-src", "include", "pybind11", "detail")
        py_build_dir = Recipe.get_recipe("hostpython3", self.ctx).get_build_dir(arch.arch)
        print(f"Python build dir: {py_build_dir}")
        #python_include_dir = join(py_build_dir, 'Include') # from build dir
        python_include_dir = self.ctx.python_recipe.include_root(arch.arch)
        pybind11_recipe = self.get_recipe('pybind11', self.ctx)
        pybind11_include_dir = pybind11_recipe.get_include_dir(arch)
        print(f"Python include dir: {python_include_dir}")
        print(f"Does Python.h exist? {exists(join(python_include_dir, 'Python.h'))}")
        python_link_root = self.ctx.python_recipe.link_root(arch.arch)
        python_link_version = self.ctx.python_recipe.link_version
        python_library = join(python_link_root,
                              'libpython{}.so'.format(python_link_version))
        python_site_packages = self.ctx.get_site_packages_dir(arch)
        python_include_numpy = join(python_site_packages,
                                        'numpy', 'core', 'include') # from python-installs dir
        print(f"Python include numpy: {python_include_numpy}")
        toolchain_file = join(self.ctx.ndk_dir,
                                'build/cmake/android.toolchain.cmake')
        protoc_path = sh.which("protoc")
        python_path = self.ctx.hostpython
        #shprint(sh.mkdir, "-p", onnx_pybind_dir)
        shprint(sh.mkdir, "-p", capi_dir)
        shprint(sh.mkdir, "-p", dist_dir)

        cmake_args = [
            "cmake",
            cmake_dir,
            #f"-DCMAKE_INSTALL_PREFIX={build_dir}",
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
            f"-DANDROID_ABI={arch.arch}",
            f"-DANDROID_PLATFORM={ANDROID_PLATFORM}",
            #"-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-Donnxruntime_ENABLE_PYTHON=ON",
            "-Donnxruntime_BUILD_SHARED_LIB=OFF",
            "-DPYBIND11_USE_CROSSCOMPILING=TRUE",
            "-Donnxruntime_USE_NNAPI_BUILTIN=ON",
            "-Donnxruntime_USE_XNNPACK=ON",
            f"-DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc",
            f"-DPython_NumPy_INCLUDE_DIR={python_include_numpy}",
            f"-DPython_EXECUTABLE={python_path}",
            f"-Dpybind11_INCLUDE_DIRS={pybind11_include_dir};{python_include_dir};{python_include_numpy}",
            #f"-DPython_INCLUDE_DIR={python_include_dir}",
            #f"-DPython_INCLUDE_DIRS={python_include_dir}",
            f"-DPython_LIBRARY={python_library}",
            f"-DPython_LIBRARIES={python_library}",
            "-DCMAKE_BUILD_TYPE=RELEASE",
            "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
        ]

        with current_directory(build_dir):
            shprint(sh.Command("cmake"), *cmake_args, _env=env)
            shprint(sh.make, '-j' + str(cpu_count()), _env=env)
            #shprint(sh.make, 'install', _env=env)
            #shprint(sh.Command("cmake"), "--build", ".", _env=env)
            #shprint(sh.Command("cmake"), "--install", ".", _env=env)

            print("=== Installed capi contents ===")
            shprint(sh.ls, "-R", capi_dir)
            # Build wheel
            #shprint(python_path, "-m", "build", "--wheel", "--no-isolation", _env=env)

            # Install wheel into target python site-packages

recipe = OnnxRuntimeRecipe()
