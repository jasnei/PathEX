import argparse
import ctypes
import glob
import os
import platform


def get_dir_filename(filename):
        dir, filename = os.path.split(filename)
        return dir, filename


def get_specified_files(folder_path: str,
                        suffixes: list=['.jpg', '.png', '.jpeg', '.bmp'],
                        recursive: bool=False):
        """
        Description:
            - Get all the suffixes files from folder path, or you could re-write it so that you could pass suffixes needed

        Parameters:
            - folder_path: str, The folder you want to get the files
            - suffixes   : list, list of all suffixes you want to get
            - recurssive : bool, Which means if get from the folder of the folder_path or not. default is False

        Return:
            -  List of the files founded
        """

        files = []
        for suffix in suffixes:
            if recursive:
                path = os.path.join(folder_path, "**", "*" + suffix)
            else:
                path = os.path.join(folder_path, "*" + suffix)
            files.extend(glob.glob(path, recursive=recursive))
        return files

def supress_ctypes_warnings():
    # for windows, justload "libtiff-5.dll" 
    # linux, because the version might be different, the lib folder name
    # would be different, some is "/usr/lib/x86_64-linux-gnu", other is
    # "/usr/lib/x86_64-redhat-linux6E", or search your folder for libtiff.so.5
    if platform.system().lower() == "windows":
        libtiff = ctypes.CDLL("libtiff-5.dll")
    elif platform.system().lower() == "linux":
        lib_default = '/usr/lib/'
        folders = os.listdir(lib_default)
        lib_path = [folder for folder in folders if 'x86' in folder]
        if lib_path:
            lib_path = os.path.join(lib_default, lib_path[0])
        else:
            lib_path = '/usr/lib64/'
        try:
            libopenslide = ctypes.CDLL(os.path.join(lib_path, 'libopenslide.so.0'))
            libopenslide.TIFFSetWarningHandler.argtypes = [ctypes.c_void_p]
            libopenslide.TIFFSetWarningHandler.restype = None
            libopenslide.TIFFSetWarningHandler(None)
            # print(f'linux --> lib openslide set up done!')
        except:
            pass
        try:
            libtiff = ctypes.CDLL(os.path.join(lib_path, 'libtiff.so.5'))
        except:
            pass

    libtiff.TIFFSetWarningHandler.argtypes = [ctypes.c_void_p]
    libtiff.TIFFSetWarningHandler.restype = ctypes.c_void_p
    libtiff.TIFFSetWarningHandler(None)
    # print(f'both --> lib TIFF set up done!')


def format_arg(args):
    msg = "\n".join("--%s=%s \\" % (k, str(v)) for k, v in dict(vars(args)).items())
    return "\n" + msg


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
