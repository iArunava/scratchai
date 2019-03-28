from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("edit_distance", ["basic_module.cpp", "edit_distance.cpp"])

setup(name="edit_distance", ext_modules=[extension_mod])

print ("dfdf")
edit_distance("dfdf", "dfdfd")
