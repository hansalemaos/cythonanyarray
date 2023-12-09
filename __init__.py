import numexpr
import os
import subprocess
import sys
import numpy as np


def _dummyimport():
    import Cython


try:
    from .cythonanyarray import create_product_ordered, create_product_unordered

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp /O2
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
np.import_array()

ctypedef fused indextypes:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t



cpdef void create_product_ordered(cython.Py_ssize_t liste, indextypes[:] indlist, indextypes[:] ctypesframedataresults,):
    cdef cython.Py_ssize_t geht,bleibt,zahlx,q,geht1,zahl
    cdef cython.Py_ssize_t indlistlen =indlist.shape[0]
    cdef cython.Py_ssize_t cou=0
    with nogil:
        for q in range(liste):
            geht = q
            bleibt = 0
            for zahlx in range(indlistlen):
                zahl=indlist[zahlx]
                geht1 = geht // zahl
                bleibt = geht % zahl
                geht = geht1
                ctypesframedataresults[q*indlistlen+zahlx+cou] = bleibt

cpdef void create_product_unordered(cython.Py_ssize_t liste, indextypes[:] indlist, indextypes[:] ctypesframedataresults,):
    cdef cython.Py_ssize_t geht,bleibt,zahlx,q,geht1,zahl
    cdef cython.Py_ssize_t indlistlen =indlist.shape[0]
    cdef cython.Py_ssize_t cou=0
    for q in prange(liste,nogil=True):
        geht = q
        bleibt = 0
        for zahlx in range(indlistlen):
            zahl=indlist[zahlx]
            geht1 = geht // zahl
            bleibt = geht % zahl
            geht = geht1
            ctypesframedataresults[q*indlistlen+zahlx+cou] = bleibt         



"""
    pyxfile = f"cythonanyarray.pyx"
    pyxfilesetup = f"cythonanyarraycompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythonanyarray', 'sources': ['cythonanyarray.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythonanyarray',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .cythonanyarray import create_product_ordered, create_product_unordered

    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def get_pointer_array(original):
    dty = np.ctypeslib.as_ctypes_type(original.dtype)

    b = original.ctypes.data
    buff = (dty * original.size).from_address(b)

    aflat = np.frombuffer(buff, dtype=original.dtype)
    return aflat


def get_iterarray(a, dtype=np.int64, unordered=True):
    indlist = np.array(a.shape, dtype=dtype)
    listexx = int(np.product(indlist))
    ctypesframedataresults2 = np.zeros((len(indlist) * listexx), dtype=dtype)
    if unordered:
        create_product_unordered(listexx, indlist, ctypesframedataresults2)
    else:
        create_product_ordered(listexx, indlist, ctypesframedataresults2)

    soa = ctypesframedataresults2.reshape(((np.product(a.shape), -1)))
    stra = tuple(a.strides)
    axa = numexpr.evaluate('bhx*xstra', global_dict={}, local_dict={'bhx': soa[..., 0], 'xstra': stra[0]})
    for qq in range(len(stra)):
        if qq == 0: continue
        numexpr.evaluate('axa+(hb*hq)', global_dict={}, local_dict={'axa': axa, 'hb': soa[..., qq], 'hq': stra[qq]},
                         out=axa)
    axar = np.ascontiguousarray(axa.reshape((-1, 1)))
    numexpr.evaluate('bhx/xstra', global_dict={}, local_dict={'bhx': axa.reshape((-1, 1)), 'xstra': a.itemsize},
                     truediv=False, casting='unsafe', out=axar)

    return np.hstack([axar, soa], dtype=dtype)


def get_flat_iter_for_cython(a, dtype=np.int64, unordered=True):
    return get_iterarray(a, dtype=dtype, unordered=unordered), get_pointer_array(a)


def get_iterarray_shape(iterray, last_dim):
    for di in range(iterray.shape[1]):
        if di >= last_dim:
            iterray = iterray[numexpr.evaluate('bhx==0', global_dict={}, local_dict={'bhx': iterray[..., di]})]
    return iterray[..., :last_dim]


