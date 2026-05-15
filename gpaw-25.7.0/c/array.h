// In the code, one utilizes calls equvalent to PyArray API,
// except instead of PyArray_BYTES one uses Array_BYTES.
// Then, if GPAW is built with GPAW_GPU_AWARE_MPI define, these macros are rewritten with wrappers.
#ifndef GPAW_ARRAY_ALLOW_CUPY

#ifdef GPAW_ARRAY_DISABLE_NUMPY
#error "No CPAW_ARRAY_ALLOW_CUPY and GPAW_ARRAY_DISABLE_NUMPY is set. No array interfaces remain."
#endif

// Check that array is well-behaved and contains data that can be sent.
#define CHK_ARRAY(a) if ((a) == NULL || !PyArray_Check(a)                   \
			 || !PyArray_ISCARRAY((PyArrayObject*)a)            \
                         || !PyArray_ISNUMBER((PyArrayObject*)a)) {         \
    PyErr_SetString(PyExc_TypeError,                                        \
		    "Not a proper NumPy array for MPI communication.");     \
    return NULL; } else

// Check that array is well-behaved, read-only  and contains data that
// can be sent.
#define CHK_ARRAY_RO(a) if ((a) == NULL || !PyArray_Check(a)                \
			 || !PyArray_ISCARRAY_RO((PyArrayObject*)a)         \
			 || !PyArray_ISNUMBER((PyArrayObject*)a)) {         \
    PyErr_SetString(PyExc_TypeError,                                        \
		    "Not a proper NumPy array for MPI communication.");     \
    return NULL; } else

// Check that two arrays have the same type, and the size of the
// second is a given multiple of the size of the first
#define CHK_ARRAYS(a,b,n)                                                             \
  if ((PyArray_TYPE((PyArrayObject*)a) != PyArray_TYPE((PyArrayObject*)b))            \
      || (PyArray_SIZE((PyArrayObject*)b) != PyArray_SIZE((PyArrayObject*)a) * n)) {   \
    PyErr_SetString(PyExc_ValueError,                                                 \
		    "Incompatible array types or sizes.");                            \
      return NULL; } else


#define Array_NDIM(a) PyArray_NDIM((PyArrayObject*)a)
#define Array_DIM(a,d)  PyArray_DIM((PyArrayObject*)a,d)
#define Array_ITEMSIZE(a) PyArray_ITEMSIZE((PyArrayObject*)a)
#define Array_BYTES(a) PyArray_BYTES((PyArrayObject*)a)
#define Array_DATA(a) PyArray_DATA((PyArrayObject*)a)
#define Array_SIZE(a) PyArray_SIZE((PyArrayObject*)a)
#define Array_TYPE(a) PyArray_TYPE((PyArrayObject*)a)
#define Array_NBYTES(a) PyArray_NBYTES((PyArrayObject*)a)
#define Array_ISCOMPLEX(a) PyArray_ISCOMPLEX((PyArrayObject*)a)

#else // GPAW_ARRAY_ALLOW_CUPY

#define CHK_ARRAY(a) // TODO
#define CHK_ARRAY_RO(a) // TODO
#define CHK_ARRAYS(a,b,n) // TODO

#include <stdio.h>

static inline int Array_NDIM(PyObject* obj)
{
    #ifndef GPAW_ARRAY_DISABLE_NUMPY
    if (PyArray_Check(obj))
    {
	return PyArray_NDIM((PyArrayObject*)obj);
    }
    #endif

    // return len(obj.shape)
    PyObject* shape = PyObject_GetAttrString(obj, "shape");
    if (shape == NULL) return -1;
    Py_DECREF(shape);
    return PyTuple_Size(shape);
}

static inline int Array_DIM(PyObject* obj, int dim)
{
    #ifndef GPAW_ARRAY_DISABLE_NUMPY
    if (PyArray_Check(obj))
    {
	return PyArray_DIM((PyArrayObject*)obj, dim);
    }
    #endif
    PyObject* shape = PyObject_GetAttrString(obj, "shape");

    if (shape == NULL) return -1;
    PyObject* pydim = PyTuple_GetItem(shape, dim);
    Py_DECREF(shape);
    if (pydim == NULL) return -1;
    int value = (int) PyLong_AS_LONG(pydim);
    return value;
}

static inline char* Array_BYTES(PyObject* obj)
{
    #ifndef GPAW_ARRAY_DISABLE_NUMPY
    if (PyArray_Check(obj))
    {
	return PyArray_BYTES((PyArrayObject*)obj);
    }
    #endif
    // Equivalent to obj.data.ptr
    PyObject* ndarray_data = PyObject_GetAttrString(obj, "data");
    if (ndarray_data == NULL) return NULL;
    PyObject* ptr_data = PyObject_GetAttrString(ndarray_data, "ptr");
    if (ptr_data == NULL) return NULL;
    char* ptr = (char*) PyLong_AS_LONG(ptr_data);
    Py_DECREF(ptr_data);
    Py_DECREF(ndarray_data);
    return ptr;
}

#define Array_DATA(a) ((void*) Array_BYTES(a))

static inline int Array_SIZE(PyObject* obj)
{
    PyObject* size = PyObject_GetAttrString(obj, "size");
    int arraysize = (int) PyLong_AS_LONG(size);
    Py_DECREF(size);
    return arraysize;
}

static inline int Array_TYPE(PyObject* obj)
{
    #ifndef GPAW_ARRAY_DISABLE_NUMPY
    if (PyArray_Check(obj))
    {
	return PyArray_TYPE((PyArrayObject*)obj);
    }
    #endif
    PyObject* dtype = PyObject_GetAttrString(obj, "dtype");

    if (dtype == NULL) return -1;

    PyObject* num = PyObject_GetAttrString(dtype, "num");
    Py_DECREF(dtype);
    if (num == NULL) return -1;

    int value =  (int) PyLong_AS_LONG(num);
    Py_DECREF(num);
    return value;
}

static inline int Array_ITEMSIZE(PyObject* obj)
{
    #ifndef GPAW_ARRAY_DISABLE_NUMPY
    if (PyArray_Check(obj))
    {
	return PyArray_ITEMSIZE((PyArrayObject*)obj);
    }
    #endif
    PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
    if (dtype == NULL) return -1;
    PyObject* itemsize_obj = PyObject_GetAttrString(dtype, "itemsize");
    if (itemsize_obj == NULL) return -1;
    int itemsize = (int) PyLong_AS_LONG(itemsize_obj);
    Py_DECREF(itemsize_obj);
    Py_DECREF(dtype);
    return itemsize;
}


static inline long Array_NBYTES(PyObject* obj)
{
    #ifndef GPAW_ARRAY_DISABLE_NUMPY
    if (PyArray_Check(obj))
    {
	return PyArray_NBYTES((PyArrayObject*)obj);
    }
    #endif
    PyObject* nbytes = PyObject_GetAttrString(obj, "nbytes");
    long nbytesvalue = PyLong_AS_LONG(nbytes);
    Py_DECREF(nbytes);
    return nbytesvalue;
}

static inline int Array_ISCOMPLEX(PyObject* obj)
{
    int result = PyTypeNum_ISCOMPLEX(Array_TYPE(obj));
    return result;
}

static inline void print_array_info(PyObject* obj)
{
    if (PyArray_Check(obj))
    {
        printf("numpy ");
    }
    if (Array_ISCOMPLEX(obj))
    {
        printf("complex ");
    }
    printf("itemsize: %d", Array_ITEMSIZE(obj));
    printf("typenum %d", Array_TYPE(obj));
    printf("shape: [");
    for (int i=0; i<Array_NDIM(obj); i++)
    {
        printf("%d", Array_DIM(obj, i));
        if (i != Array_NDIM(obj) - 1)
        {
            printf(", ");
        }
        else
        {
            printf("]");
        }
    printf("\n");
    }
}


#endif

