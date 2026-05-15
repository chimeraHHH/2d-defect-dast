/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2005-2009  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef PARALLEL

#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <mpi.h>
#include "extensions.h"
#include <structmember.h>
#include "mympi.h"
#ifdef __bgp__
#include <mpix.h>
#endif

// Wrappers to support GPU_AWARE_MPI
#ifdef GPAW_GPU_AWARE_MPI
#define GPAW_ARRAY_ALLOW_CUPY
#endif
#include "array.h"

#ifdef GPAW_MPI2
#ifndef GPAW_MPI_INPLACE
#error "Deprecated: Define or undefine GPAW_MPI_INPLACE, instead of using GPAW_MPI2."
#endif
#endif

void gpawDeviceSynchronize();

// Check that a processor number is valid
#define CHK_PROC(n) if (n < 0 || n >= self->size) {\
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");     \
    return NULL; } else

// Check that a processor number is valid or is -1
#define CHK_PROC_DEF(n) if (n < -1 || n >= self->size) {\
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");     \
    return NULL; } else

// Check that a processor number is valid and is not this processor
#define CHK_OTHER_PROC(n) if (n < 0 || n >= self->size || n == self->rank) { \
    PyErr_SetString(PyExc_ValueError, "Invalid processor number.");     \
    return NULL; } else

// MPI request object, so we can store a reference to the buffer,
// preventing its early deallocation.
typedef struct {
  PyObject_HEAD
  MPI_Request rq;
  PyObject *buffer;
  int status;
} GPAW_MPI_Request;

static void maybeSynchronize(PyObject* a)
{
#ifdef GPAW_GPU_AWARE_MPI
    if (!PyArray_Check(a))
    {
        gpawDeviceSynchronize();
    }
#endif
}

static PyObject *mpi_request_wait(GPAW_MPI_Request *self, PyObject *noargs)
{

  if (self->status == 0)
    {
      // Calling wait multiple times is allowed but meaningless (as in the MPI standard)
      Py_RETURN_NONE;
    }
  int ret = MPI_Wait(&(self->rq), MPI_STATUS_IGNORE);
  if (ret != MPI_SUCCESS)
    {
      PyErr_SetString(PyExc_RuntimeError, "MPI_Wait error occurred.");
      return NULL;
    }
  Py_DECREF(self->buffer);
  self->status = 0;

  Py_RETURN_NONE;
}

static PyObject *mpi_request_test(GPAW_MPI_Request *self, PyObject *noargs)
{

  if (self->status == 0)
    {
      Py_RETURN_TRUE;  // Already completed
    }
  int flag;
  int ret = MPI_Test(&(self->rq), &flag, MPI_STATUS_IGNORE); // Can this change the Python string?
  if (ret != MPI_SUCCESS)
    {
      PyErr_SetString(PyExc_RuntimeError, "MPI_Test error occurred.");
      return NULL;
    }
  if (flag)
    {
      Py_DECREF(self->buffer);
      self->status = 0;
      Py_RETURN_TRUE;
    }
  else
    {
      Py_RETURN_FALSE;
    }
}

static void mpi_request_dealloc(GPAW_MPI_Request *self)
{
  if (self->status)
    {
      PyObject *none = mpi_request_wait(self, NULL);
      Py_DECREF(none);
    }
  PyObject_Del(self);
}

static PyMemberDef mpi_request_members[] = {
    {"status", T_INT, offsetof(GPAW_MPI_Request, status), READONLY,
	"status of the request, non-zero if communication is pending."},
    {NULL}
};

static PyMethodDef mpi_request_methods[] = {
    {"wait", (PyCFunction) mpi_request_wait, METH_NOARGS,
	"Wait for the communication to complete."
    },
    {"test", (PyCFunction) mpi_request_test, METH_NOARGS,
	"Test if the communication has completed."
    },
    {NULL}
};

PyTypeObject GPAW_MPI_Request_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MPI_Request",             /*tp_name*/
    sizeof(GPAW_MPI_Request),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)mpi_request_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MPI request object",           /* tp_doc */
    0,                   /* tp_traverse */
    0,                   /* tp_clear */
    0,                   /* tp_richcompare */
    0,                   /* tp_weaklistoffset */
    0,                   /* tp_iter */
    0,                   /* tp_iternext */
    mpi_request_methods,             /* tp_methods */
    mpi_request_members,
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

static GPAW_MPI_Request *NewMPIRequest(void)
{
  GPAW_MPI_Request *self;

  self = PyObject_NEW(GPAW_MPI_Request, &GPAW_MPI_Request_type);
  if (self == NULL) return NULL;
  memset(&(self->rq), 0, sizeof(MPI_Request));
  self->buffer = NULL;
  self->status = 1;  // Active

  return self;
}


static void mpi_ensure_finalized(void)
{
    int already_finalized = 1;
    int ierr = MPI_SUCCESS;

    MPI_Finalized(&already_finalized);
    if (!already_finalized)
    {
	ierr = MPI_Finalize();
    }
    if (ierr != MPI_SUCCESS)
	PyErr_SetString(PyExc_RuntimeError, "MPI_Finalize error occurred");
}


// MPI initialization
static void mpi_ensure_initialized(void)
{
    int already_initialized = 1;
    int ierr = MPI_SUCCESS;

    // Check whether MPI is already initialized
    MPI_Initialized(&already_initialized);
    if (!already_initialized)
    {
        // if not, let's initialize it
        int use_threads = 0;
#ifdef GPAW_GPU
        use_threads = 1;
#endif
#ifdef _OPENMP
        use_threads = 1;
#endif
        if (!use_threads) {
            ierr = MPI_Init(NULL, NULL);
            if (ierr == MPI_SUCCESS)
            {
                // No problem: register finalization when at Python exit
                Py_AtExit(*mpi_ensure_finalized);
            }
            else
            {
                // We have a problem: raise an exception
                char err[MPI_MAX_ERROR_STRING];
                int resultlen;
                MPI_Error_string(ierr, err, &resultlen);
                PyErr_SetString(PyExc_RuntimeError, err);
            }
        } else {
            int granted;
            ierr = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &granted);
            if (ierr == MPI_SUCCESS && granted == MPI_THREAD_MULTIPLE)
            {
                // No problem: register finalization when at Python exit
                Py_AtExit(*mpi_ensure_finalized);
            }
            else if (granted != MPI_THREAD_MULTIPLE)
            {
                // We have a problem: raise an exception
                char err[MPI_MAX_ERROR_STRING] = "MPI_THREAD_MULTIPLE is not supported";
                PyErr_SetString(PyExc_RuntimeError, err);
            }
            else
            {
                // We have a problem: raise an exception
                char err[MPI_MAX_ERROR_STRING];
                int resultlen;
                MPI_Error_string(ierr, err, &resultlen);
                PyErr_SetString(PyExc_RuntimeError, err);
            }
        }
    }
}


static void mpi_dealloc(MPIObject *obj)
{
    if (obj->comm != MPI_COMM_WORLD)
	MPI_Comm_free(&(obj->comm));
    Py_XDECREF(obj->parent);
    free(obj->members);
    PyObject_DEL(obj);
}

static PyObject * mpi_sendreceive(MPIObject *self, PyObject *args,
				  PyObject *kwargs)
{
    PyObject* a;
    PyObject* b;
    int dest, src;
    int sendtag = 123;
    int recvtag = 123;
    static char *kwlist[] = {"a", "dest", "b", "src", "sendtag", "recvtag",
			     NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiOi|ii:sendreceive",
				     kwlist,
				     &a, &dest, &b, &src, &sendtag, &recvtag))
	return NULL;
    CHK_ARRAY(a);
    CHK_OTHER_PROC(dest);
    CHK_ARRAY(b);
    CHK_OTHER_PROC(src);
    int nsend = Array_ITEMSIZE(a);
    for (int d = 0; d < Array_NDIM(a); d++)
	nsend *= Array_DIM(a,d);
    int nrecv = Array_ITEMSIZE(b);
    for (int d = 0; d < Array_NDIM(b); d++)
	nrecv *= Array_DIM(b,d);
    maybeSynchronize(a);
    int ret = MPI_Sendrecv(Array_BYTES(a), nsend, MPI_BYTE, dest, sendtag,
			   Array_BYTES(b), nrecv, MPI_BYTE, src, recvtag,
			   self->comm, MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS) {
	PyErr_SetString(PyExc_RuntimeError, "MPI_Sendrecv error occurred.");
	return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject * mpi_receive(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int src;
  int tag = 123;
  int block = 1;
  static char *kwlist[] = {"a", "src", "tag", "block", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii:receive", kwlist,
				   &a, &src, &tag, &block))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(src);
  int n = Array_ITEMSIZE(a);
  for (int d = 0; d < Array_NDIM(a); d++)
    n *= Array_DIM(a, d);
  if (block)
    {
      maybeSynchronize(a);
      int ret = MPI_Recv(Array_BYTES(a), n, MPI_BYTE, src, tag, self->comm,
			 MPI_STATUS_IGNORE);
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Recv error occurred.");
	  return NULL;
	}
      Py_RETURN_NONE;
    }
  else
    {
      GPAW_MPI_Request *req = NewMPIRequest();
      if (req == NULL) return NULL;
      req->buffer = (PyObject*)a;
      Py_INCREF(req->buffer);
      maybeSynchronize(a);
      int ret = MPI_Irecv(Array_BYTES(a), n, MPI_BYTE, src, tag, self->comm,
			  &(req->rq));
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Irecv error occurred.");
	  return NULL;
	}
      return (PyObject *) req;
    }
}

static PyObject * mpi_send(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int dest;
  int tag = 123;
  int block = 1;
  static char *kwlist[] = {"a", "dest", "tag", "block", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii:send", kwlist,
				   &a, &dest, &tag, &block))
    return NULL;
  CHK_ARRAY(a);
  CHK_OTHER_PROC(dest);
  int n = Array_ITEMSIZE(a);
  for (int d = 0; d < Array_NDIM(a); d++)
    n *= Array_DIM(a,d);
  if (block)
    {
      maybeSynchronize(a);
      int ret = MPI_Send(Array_BYTES(a), n, MPI_BYTE, dest, tag, self->comm);
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Send error occurred.");
	  return NULL;
	}
      Py_RETURN_NONE;
    }
  else
    {
      GPAW_MPI_Request *req = NewMPIRequest();
      req->buffer = (PyObject*)a;
      Py_INCREF(a);
      maybeSynchronize(a);
      int ret = MPI_Isend(Array_BYTES(a), n, MPI_BYTE, dest, tag, self->comm,
			  &(req->rq));
      if (ret != MPI_SUCCESS)
	{
	  PyErr_SetString(PyExc_RuntimeError, "MPI_Isend error occurred.");
	  return NULL;
	}
      return (PyObject *)req;
    }
}


static PyObject * mpi_ssend(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject* a;
  int dest;
  int tag = 123;
  static char *kwlist[] = {"a", "dest", "tag", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i:send", kwlist,
				   &a, &dest, &tag))
    return NULL;
  CHK_ARRAY_RO(a);
  CHK_OTHER_PROC(dest);
  int n = Array_ITEMSIZE(a);
  for (int d = 0; d < Array_NDIM(a); d++)
    n *= Array_DIM(a,d);
  maybeSynchronize(a);
  MPI_Ssend(Array_BYTES(a), n, MPI_BYTE, dest, tag, self->comm);
  Py_RETURN_NONE;
}


static PyObject * mpi_name(MPIObject *self, PyObject* Py_UNUSED(noargs))
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlen;
  MPI_Get_processor_name(name, &resultlen);
  return Py_BuildValue("s#", name, (Py_ssize_t)resultlen);
}


static PyObject * mpi_abort(MPIObject *self, PyObject *args)
{
  int errcode;
  if (!PyArg_ParseTuple(args, "i:abort", &errcode))
    return NULL;
  MPI_Abort(self->comm, errcode);
  Py_RETURN_NONE;
}

static PyObject * mpi_barrier(MPIObject *self, PyObject* noargs)
{
  MPI_Barrier(self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_test(MPIObject *self, PyObject *args)
{
  GPAW_MPI_Request* s;
  if (!PyArg_ParseTuple(args, "O!:wait", &GPAW_MPI_Request_type, &s))
	return NULL;
  return mpi_request_test(s, NULL);
}

static PyObject * mpi_testall(MPIObject *self, PyObject *requests)
{
  int n;   // Number of requests
  MPI_Request *rqs = NULL;
  int flag = 0;
  if (!PySequence_Check(requests))
    {
      PyErr_SetString(PyExc_TypeError, "mpi.testall: argument must be a sequence");
      return NULL;
    }
  // Extract the request objects
  n = PySequence_Size(requests);
  assert(n >= 0);  // This cannot fail.
  rqs = GPAW_MALLOC(MPI_Request, n);
  assert(rqs != NULL);
  for (int i = 0; i < n; i++)
    {
      PyObject *o = PySequence_GetItem(requests, i);
      if (o == NULL)
	return NULL;
      if (Py_TYPE(o) != &GPAW_MPI_Request_type)
	{
	  Py_DECREF(o);
	  free(rqs);
	  PyErr_SetString(PyExc_TypeError, "mpi.testall: argument must be a sequence of MPI requests");
	  return NULL;
	}
      GPAW_MPI_Request *s = (GPAW_MPI_Request *)o;
      rqs[i] = s->rq;
      Py_DECREF(o);
    }
  // Do the actual test.
  int ret = MPI_Testall(n, rqs, &flag, MPI_STATUSES_IGNORE);
  if (ret != MPI_SUCCESS)
    {
      // We do not dare to release the buffers now!
      PyErr_SetString(PyExc_RuntimeError, "MPI_Testall error occurred.");
      return NULL;
    }
  // Unlike MPI_Test, if flag outcome is non-zero, MPI_Testall will deallocate
  // all requests which were allocated by nonblocking communication calls, so
  // we must free these buffers. Otherwise, none of the requests are modified.
  if (flag != 0)
    {
      // Release the buffers used by the MPI communication
      for (int i = 0; i < n; i++)
      {
	GPAW_MPI_Request *o = (GPAW_MPI_Request *) PySequence_GetItem(requests, i);
	if (o->status)
	{
	  assert(o->buffer != NULL);
	  Py_DECREF(o->buffer);
	}
	o->status = 0;
	Py_DECREF(o);
      }
    }
  // Release internal data and return.
  free(rqs);
  return Py_BuildValue("i", flag);
}

static PyObject * mpi_wait(MPIObject *self, PyObject *args)
{
  GPAW_MPI_Request* s;
  if (!PyArg_ParseTuple(args, "O!:wait", &GPAW_MPI_Request_type, &s))
    return NULL;
  return mpi_request_wait(s, NULL);
}

static PyObject * mpi_waitall(MPIObject *self, PyObject *requests)
{
  int n;   // Number of requests
  MPI_Request *rqs = NULL;
  if (!PySequence_Check(requests))
    {
      PyErr_SetString(PyExc_TypeError, "mpi.waitall: argument must be a sequence");
      return NULL;
    }
  // Extract the request objects
  n = PySequence_Size(requests);
  assert(n >= 0);  // This cannot fail.
  rqs = GPAW_MALLOC(MPI_Request, n);
  for (int i = 0; i < n; i++)
    {
      PyObject *o = PySequence_GetItem(requests, i);
      if (o == NULL)
	return NULL;
      if (Py_TYPE(o) != &GPAW_MPI_Request_type)
	{
	  Py_DECREF(o);
	  free(rqs);
	  PyErr_SetString(PyExc_TypeError, "mpi.waitall: argument must be a sequence of MPI requests");
	  return NULL;
	}
      GPAW_MPI_Request *s = (GPAW_MPI_Request *)o;
      rqs[i] = s->rq;
      Py_DECREF(o);
    }
  int ret = MPI_Waitall(n, rqs, MPI_STATUSES_IGNORE);
  if (ret != MPI_SUCCESS)
    {
      // We do not dare to release the buffers now!
      PyErr_SetString(PyExc_RuntimeError, "MPI_Waitall error occurred.");
      return NULL;
    }
  // Release the buffers used by the MPI communication
  for (int i = 0; i < n; i++)
   {
     GPAW_MPI_Request *o = (GPAW_MPI_Request *) PySequence_GetItem(requests, i);
     if (o->status)
     {
       assert(o->buffer != NULL);
       Py_DECREF(o->buffer);
     }
     o->status = 0;
     Py_DECREF(o);
   }
  // Release internal data and return.
  free(rqs);
  Py_RETURN_NONE;
}


static MPI_Datatype get_mpi_datatype(PyObject *a)
{
  int n = Array_ITEMSIZE(a);
  if (Array_ISCOMPLEX(a))
    n = n / 2;
  int array_type = Array_TYPE(a);
  switch(array_type)
    {
      // Floating point numbers including complex numbers
    case NPY_DOUBLE:
    case NPY_CDOUBLE:
      assert(sizeof(double) == n);
      return MPI_DOUBLE;
    case NPY_FLOAT:
    case NPY_CFLOAT:
      assert(sizeof(float) == n);
      return MPI_FLOAT;
    case NPY_LONGDOUBLE:
    case NPY_CLONGDOUBLE:
       assert(sizeof(long double) == n);
      return MPI_LONG_DOUBLE;
      // Signed integer types
    case NPY_BYTE:
      assert(sizeof(char) == n);
      return MPI_CHAR;
    case NPY_SHORT:
      assert(sizeof(short) == n);
      return MPI_SHORT;
    case NPY_INT:
      assert(sizeof(int) == n);
      return MPI_INT;
    case NPY_LONG:
      assert(sizeof(long) == n);
      return MPI_LONG;
      // Unsigned integer types
    case NPY_BOOL:
    case NPY_UBYTE:
      assert(sizeof(unsigned char) == n);
      return MPI_UNSIGNED_CHAR;
    case NPY_USHORT:
      assert(sizeof(unsigned short) == n);
      return MPI_UNSIGNED_SHORT;
    case NPY_UINT:
      assert(sizeof(unsigned) == n);
      return MPI_UNSIGNED;
    case NPY_ULONG:
      assert(sizeof(unsigned long) == n);
      return MPI_UNSIGNED_LONG;
    }

  // If we reach this point none of the cases worked out.
  PyErr_SetString(PyExc_ValueError, "Cannot communicate data of this type.");
  return 0;
}

static PyObject * mpi_reduce(MPIObject *self, PyObject *args, PyObject *kwargs,
			     MPI_Op operation, int allowcomplex)
{
#ifdef GPAW_MPI_DEBUG
  MPI_Barrier(self->comm);
#endif
  PyObject* obj;
  int root = -1;
  static char *kwlist[] = {"a", "root", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:reduce", kwlist,
				   &obj, &root))
    return NULL;
  CHK_PROC_DEF(root);
  if (PyFloat_Check(obj))
    {
      double din = PyFloat_AS_DOUBLE(obj);
      double dout;
      if (root == -1)
	MPI_Allreduce(&din, &dout, 1, MPI_DOUBLE, operation, self->comm);
      else
	MPI_Reduce(&din, &dout, 1, MPI_DOUBLE, operation, root, self->comm);
      return PyFloat_FromDouble(dout);
    }
  if (PyLong_Check(obj))
    {
      long din = PyLong_AS_LONG(obj);
      long dout;
      if (root == -1)
	MPI_Allreduce(&din, &dout, 1, MPI_LONG, operation, self->comm);
      else
	MPI_Reduce(&din, &dout, 1, MPI_LONG, operation, root, self->comm);
      return PyLong_FromLong(dout);
    }
  else if (PyComplex_Check(obj) && allowcomplex)
    {
      double din[2];
      double dout[2];
      din[0] = PyComplex_RealAsDouble(obj);
      din[1] = PyComplex_ImagAsDouble(obj);
      if (root == -1)
	MPI_Allreduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, self->comm);
      else
	MPI_Reduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, root, self->comm);
      return PyComplex_FromDoubles(dout[0], dout[1]);
    }
  else if (PyComplex_Check(obj))
    {
      PyErr_SetString(PyExc_ValueError,
		      "Operation not allowed on complex numbers");
      return NULL;
    }
  else   // It should be an array
    {
      int n;
      int elemsize;
      MPI_Datatype datatype;
      PyObject* aobj = obj;
      CHK_ARRAY(aobj);
      datatype = get_mpi_datatype(aobj);
      if (datatype == 0)
	return NULL;
      n = Array_SIZE(aobj);
      elemsize = Array_ITEMSIZE(aobj);
      if (Array_ISCOMPLEX(aobj))
	{
	  if (allowcomplex)
	    {
	      n *= 2;
	      elemsize /= 2;
	    }
	  else
	    {
	      PyErr_SetString(PyExc_ValueError,
			      "Operation not allowed on complex numbers");
	      return NULL;
	    }
	}
      if (root == -1)
	{
          maybeSynchronize(aobj);
#ifdef GPAW_MPI_INPLACE
	  MPI_Allreduce(MPI_IN_PLACE, Array_BYTES(aobj), n, datatype,
			operation, self->comm);
#else
	  char* b = GPAW_MALLOC(char, n * elemsize);
      MPI_Allreduce(Array_BYTES(aobj), b, n, datatype, operation,
                    self->comm);
      assert(Array_NBYTES(aobj) == n * elemsize);
      memcpy(Array_BYTES(aobj), b, n * elemsize);
      free(b);
#endif
	}
      else
	{
	  int rank;
	  MPI_Comm_rank(self->comm, &rank);
	  char* b = 0;
	  if (rank == root)
	    {
              maybeSynchronize(aobj);
#ifdef GPAW_MPI_INPLACE
	      MPI_Reduce(MPI_IN_PLACE, Array_BYTES(aobj), n,
			 datatype, operation, root, self->comm);
#else
	      b = GPAW_MALLOC(char, n * elemsize);
	      MPI_Reduce(Array_BYTES(aobj), b, n, datatype,
			 operation, root, self->comm);
	      assert(Array_NBYTES(aobj) == n * elemsize);
	      memcpy(Array_BYTES(aobj), b, n * elemsize);
	      free(b);
#endif
	    }
	  else
	    {
              maybeSynchronize(aobj);
	      MPI_Reduce(Array_BYTES(aobj), b, n, datatype,
			 operation, root, self->comm);
	    }
	}
      Py_RETURN_NONE;
    }
}

static PyObject * mpi_reduce_scalar(MPIObject *self, PyObject *args, PyObject *kwargs,
                                    MPI_Op operation, int allowcomplex)
{
#ifdef GPAW_MPI_DEBUG
  MPI_Barrier(self->comm);
#endif
  PyObject* obj;
  int root = -1;
  static char *kwlist[] = {"a", "root", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:reduce", kwlist,
				   &obj, &root))
    return NULL;
  CHK_PROC_DEF(root);
  if (PyFloat_Check(obj))
    {
      double din = PyFloat_AS_DOUBLE(obj);
      double dout;
      if (root == -1)
	MPI_Allreduce(&din, &dout, 1, MPI_DOUBLE, operation, self->comm);
      else
	MPI_Reduce(&din, &dout, 1, MPI_DOUBLE, operation, root, self->comm);
      return PyFloat_FromDouble(dout);
    }
  if (PyLong_Check(obj))
    {
      long din = PyLong_AS_LONG(obj);
      long dout;
      if (root == -1)
	MPI_Allreduce(&din, &dout, 1, MPI_LONG, operation, self->comm);
      else
	MPI_Reduce(&din, &dout, 1, MPI_LONG, operation, root, self->comm);
      return PyLong_FromLong(dout);
    }
  else if (PyComplex_Check(obj) && allowcomplex)
    {
      double din[2];
      double dout[2];
      din[0] = PyComplex_RealAsDouble(obj);
      din[1] = PyComplex_ImagAsDouble(obj);
      if (root == -1)
	MPI_Allreduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, self->comm);
      else
	MPI_Reduce(&din, &dout, 2, MPI_DOUBLE, MPI_SUM, root, self->comm);
      return PyComplex_FromDoubles(dout[0], dout[1]);
    }
  else if (PyComplex_Check(obj))
    {
      PyErr_SetString(PyExc_ValueError,
		      "Operation not allowed on complex numbers");
      return NULL;
    }
  else   // It should be an array
    {
       PyErr_SetString(PyExc_ValueError,
           "Operation not allowed for this datatype for mpi_sum_scalar.");
	      return NULL;
    }
}

static PyObject * mpi_sum(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args, kwargs, MPI_SUM, 1);
}

static PyObject * mpi_sum_scalar(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce_scalar(self, args, kwargs, MPI_SUM, 1);
}

static PyObject * mpi_product(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  // No complex numbers as that would give separate products of
  // real and imaginary parts.
  return mpi_reduce(self, args, kwargs,  MPI_PROD, 0);
}

static PyObject * mpi_max(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args,  kwargs, MPI_MAX, 0);
}

static PyObject * mpi_max_scalar(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce_scalar(self, args,  kwargs, MPI_MAX, 0);
}

static PyObject * mpi_min(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce(self, args,  kwargs, MPI_MIN, 0);
}

static PyObject * mpi_min_scalar(MPIObject *self, PyObject *args, PyObject *kwargs)
{
  return mpi_reduce_scalar(self, args,  kwargs, MPI_MIN, 0);
}

static PyObject * mpi_scatter(MPIObject *self, PyObject *args)
{
  PyObject* sendobj;
  PyObject* recvobj;
  int root;
  if (!PyArg_ParseTuple(args, "OOi:scatter", &sendobj, &recvobj, &root))
    return NULL;
  CHK_ARRAY(recvobj);
  CHK_PROC(root);
  char* source = 0;
  if (self->rank == root) {
    CHK_ARRAY(sendobj);
    CHK_ARRAYS(recvobj, sendobj, self->size); // size(send) = size(recv)*Ncpu
    source = Array_BYTES(sendobj);
  }
  int n = Array_ITEMSIZE(recvobj);
  for (int d = 0; d < Array_NDIM(recvobj); d++)
    n *= Array_DIM(recvobj,d);
  maybeSynchronize(recvobj);
  MPI_Scatter(source, n, MPI_BYTE, Array_BYTES(recvobj),
	      n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}



static PyObject * mpi_allgather(MPIObject *self, PyObject *args)
{
  PyObject* a;
  PyObject* b;
  if (!PyArg_ParseTuple(args, "OO:allgather", &a, &b))
    return NULL;
  CHK_ARRAY(a);
  CHK_ARRAY(b);
  CHK_ARRAYS(a, b, self->size);
  int n = Array_ITEMSIZE(a);
  for (int d = 0; d < Array_NDIM(a); d++)
    n *= Array_DIM(a,d);
  // What about endianness????
  maybeSynchronize(a);
  MPI_Allgather(Array_BYTES(a), n, MPI_BYTE, Array_BYTES(b), n,
		MPI_BYTE, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_gather(MPIObject *self, PyObject *args)
{
  PyObject* a;
  int root;
  PyObject* b = 0;
  if (!PyArg_ParseTuple(args, "Oi|O", &a, &root, &b))
    return NULL;
  CHK_ARRAY(a);
  CHK_PROC(root);
  if (root == self->rank)
    {
      CHK_ARRAY(b);
      CHK_ARRAYS(a, b, self->size);
    }
  else if ((PyObject*)b != Py_None && b != NULL)
    {
      fprintf(stderr, "******** Root=%d\n", root);
      PyErr_SetString(PyExc_ValueError,
		      "mpi_gather: b array should not be given on non-root processors.");
      return NULL;
    }
  int n = Array_ITEMSIZE(a);
  for (int d = 0; d < Array_NDIM(a); d++)
    n *= Array_DIM(a,d);
  maybeSynchronize(a);
  if (root != self->rank)
    MPI_Gather(Array_BYTES(a), n, MPI_BYTE, 0, n, MPI_BYTE, root, self->comm);
  else
    MPI_Gather(Array_BYTES(a), n, MPI_BYTE, Array_BYTES(b), n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}

static PyObject * mpi_broadcast(MPIObject *self, PyObject *args)
{
#ifdef GPAW_MPI_DEBUG
  MPI_Barrier(self->comm);
#endif
  PyObject* buf;
  int root;
  if (!PyArg_ParseTuple(args, "Oi:broadcast", &buf, &root))
    return NULL;
  if (root == self->rank)
      CHK_ARRAY_RO(buf);
  else
      CHK_ARRAY(buf);

  CHK_PROC(root);
  int n = Array_ITEMSIZE(buf);
  for (int d = 0; d < Array_NDIM(buf); d++)
    n *= Array_DIM(buf,d);
  maybeSynchronize(buf);
  MPI_Bcast(Array_BYTES(buf), n, MPI_BYTE, root, self->comm);
  Py_RETURN_NONE;
}

static PyObject *mpi_compare(MPIObject *self, PyObject *args)
{
  MPIObject* other;
  int result;
  char* pyresult;
  if (!PyArg_ParseTuple(args, "O", &other))
    return NULL;

  MPI_Comm_compare(self->comm, other->comm, &result);
  if(result == MPI_IDENT) pyresult = "ident";
  else if (result == MPI_CONGRUENT) pyresult = "congruent";
  else if (result == MPI_SIMILAR) pyresult = "similar";
  else if (result == MPI_UNEQUAL) pyresult = "unequal";
  else return NULL;
  return Py_BuildValue("s", pyresult);
}

static PyObject *mpi_translate_ranks(MPIObject *self, PyObject *args)
{
  PyObject* myranks_anytype; // Conversion to numpy array below
  MPIObject* other;

  if (!PyArg_ParseTuple(args, "OO", &other, &myranks_anytype))
    return NULL;

  // XXXXXX This uses NPY_LONG and NPY_INT.  On some computers the
  // returned array is int32 while np.array(..., dtype=int) returns
  // int64.  This should very probably be changed so it always
  // corresponds to the default int of numpy.

  // This handling of arrays of ranks is taken from the MPICommunicator
  // creation method.  See that method for explanation of casting, datatypes
  // etc.
  PyArrayObject *myranks_long = (PyArrayObject*)PyArray_ContiguousFromAny(
					    myranks_anytype, NPY_LONG, 1, 1);
  if(myranks_long == NULL)
    return NULL;

  int nranks = PyArray_DIM(myranks_long, 0);

  PyArrayObject *myranks;
  myranks = (PyArrayObject*)PyArray_Cast(myranks_long, NPY_INT);

  npy_intp rankshape[1];
  rankshape[0] = PyArray_SIZE(myranks);
  PyArrayObject* other_ranks = (PyArrayObject*)PyArray_SimpleNew(1, rankshape,
								 NPY_INT);

  MPI_Group mygroup, othergroup;
  MPI_Comm_group(self->comm, &mygroup);
  MPI_Comm_group(other->comm, &othergroup);

  int* rankdata = (int*)PyArray_BYTES(myranks);
  int* otherrankdata = (int*)PyArray_BYTES(other_ranks);
  MPI_Group_translate_ranks(mygroup, nranks, rankdata, othergroup,
			    otherrankdata);

  // Return something with a definite value to Python.
  for(int i=0; i < nranks; i++) {
      if(otherrankdata[i] == MPI_UNDEFINED) {
	  otherrankdata[i] = -1;
      }
  }
  PyObject* other_ranks_anytype = PyArray_Cast(other_ranks,
      PyArray_TYPE((PyArrayObject*)myranks_anytype));

  Py_DECREF(myranks_long);
  Py_DECREF(myranks);
  Py_DECREF(other_ranks);
  return (PyObject*)other_ranks_anytype;
}

static PyObject * mpi_alltoallv(MPIObject *self, PyObject *args)
{
  PyObject* send_obj;
  PyObject* send_cnts;
  PyObject* send_displs;
  PyObject* recv_obj;
  PyObject* recv_cnts;
  PyObject* recv_displs;

  if (!PyArg_ParseTuple(args, "OOOOOO:alltoallv", &send_obj, &send_cnts,
						  &send_displs, &recv_obj,
						  &recv_cnts, &recv_displs))
    return NULL;
  CHK_ARRAY(send_obj);
  CHK_ARRAY(send_cnts);
  CHK_ARRAY(send_displs);
  CHK_ARRAY(recv_obj);
  CHK_ARRAY(recv_cnts);
  CHK_ARRAY(recv_displs);

  int *s_cnts = GPAW_MALLOC(int, self->size);
  int *s_displs = GPAW_MALLOC(int, self->size);
  int *r_cnts = GPAW_MALLOC(int, self->size);
  int *r_displs = GPAW_MALLOC(int, self->size);

  /* Create count and displacement arrays in units of bytes */
  int elem_size = Array_ITEMSIZE(send_obj);

  long* tmp1 = Array_DATA(send_cnts);
  long* tmp2 = Array_DATA(send_displs);
  long* tmp3 = Array_DATA(recv_cnts);
  long* tmp4 = Array_DATA(recv_displs);
  for (int i=0; i < self->size; i++) {
      s_cnts[i] = tmp1[i] * elem_size;
      s_displs[i] = tmp2[i] * elem_size;
      r_cnts[i] = tmp3[i] * elem_size;
      r_displs[i] = tmp4[i] * elem_size;
  }
  maybeSynchronize(send_obj);

  MPI_Alltoallv(Array_BYTES(send_obj),
		s_cnts, s_displs,
		MPI_BYTE, Array_BYTES(recv_obj), r_cnts,
		r_displs, MPI_BYTE, self->comm);

  free(s_cnts);
  free(s_displs);
  free(r_cnts);
  free(r_displs);

  Py_RETURN_NONE;
}

static PyObject * get_members(MPIObject *self, PyObject *args)
{
  PyArrayObject *ranks;
  npy_intp ranks_dims[1] = {self->size};
  ranks = (PyArrayObject *) PyArray_SimpleNew(1, ranks_dims, NPY_INT);
  if (ranks == NULL)
    return NULL;
  memcpy(INTP(ranks), self->members, self->size*sizeof(int));
  PyObject* values = Py_BuildValue("O", ranks);
  Py_DECREF(ranks);
  return values;
}

// See the documentation for corresponding function in debug wrapper
// for the purpose of this function (gpaw/mpi/__init__.py)
static PyObject * get_c_object(MPIObject *self, PyObject *args)
{
  return Py_BuildValue("O", self);
}

// Forward declaration of MPI_Communicator because it needs MPIType
// that needs MPI_getattr that needs MPI_Methods that need
// MPI_Communicator that need ...
static PyObject * MPICommunicator(MPIObject *self, PyObject *args);

static PyMethodDef mpi_methods[] = {
    {"sendreceive",          (PyCFunction)mpi_sendreceive,
     METH_VARARGS|METH_KEYWORDS,
     "sendreceive(a, dest, b, src, desttag=123, srctag=123) sends an array a to dest and receives an array b from src."},
    {"receive",          (PyCFunction)mpi_receive,
     METH_VARARGS|METH_KEYWORDS,
     "receive(a, src, tag=123, block=1) receives array a from src."},
    {"send",             (PyCFunction)mpi_send,
     METH_VARARGS|METH_KEYWORDS,
     "send(a, dest, tag=123, block=1) sends array a to dest."},
    {"ssend",             (PyCFunction)mpi_ssend,
     METH_VARARGS|METH_KEYWORDS,
     "ssend(a, dest, tag=123) synchronously sends array a to dest."},
    {"abort",            (PyCFunction)mpi_abort,        METH_VARARGS,
     "abort(errcode) aborts all MPI tasks."},
    {"name",             (PyCFunction)mpi_name,         METH_NOARGS,
     "name() returns the name of the processor node."},
    {"barrier",          (PyCFunction)mpi_barrier,      METH_NOARGS,
     "barrier() synchronizes all MPI tasks"},
    {"test",             (PyCFunction)mpi_test,         METH_VARARGS,
     "test(request) tests if a nonblocking communication is complete."},
    {"testall",          (PyCFunction)mpi_testall,      METH_O,
     "testall(list_of_rqs) tests if multiple nonblocking communications are complete."},
    {"wait",             (PyCFunction)mpi_wait,         METH_VARARGS,
     "wait(request) waits for a nonblocking communication to complete."},
    {"waitall",          (PyCFunction)mpi_waitall,      METH_O,
     "waitall(list_of_rqs) waits for multiple nonblocking communications to complete."},
    {"sum",              (PyCFunction)mpi_sum,
     METH_VARARGS|METH_KEYWORDS,
     "sum(a, root=-1) sums arrays, result on all tasks unless root is given."},
    {"sum_scalar",       (PyCFunction)mpi_sum_scalar,
     METH_VARARGS|METH_KEYWORDS,
     "sum_scalar(a, root=-1) sums numbers, result on all tasks unless root is given. Returns the sum."},
    {"product",          (PyCFunction)mpi_product,
     METH_VARARGS|METH_KEYWORDS,
     "product(a, root=-1) multiplies arrays, result on all tasks unless root is given."},
    {"max",              (PyCFunction)mpi_max,
     METH_VARARGS|METH_KEYWORDS,
     "max(a, root=-1) maximum of arrays, result on all tasks unless root is given."},
    {"max_scalar",       (PyCFunction)mpi_max_scalar,
     METH_VARARGS|METH_KEYWORDS,
     "max_sclar(a, root=-1) maximum of scalars, result on all tasks unless root is given. Returns the value."},
    {"min",              (PyCFunction)mpi_min,
     METH_VARARGS|METH_KEYWORDS,
     "min(a, root=-1) minimum of arrays, result on all tasks unless root is given."},
    {"min_scalar",       (PyCFunction)mpi_min_scalar,
     METH_VARARGS|METH_KEYWORDS,
     "min_scalar(a, root=-1) minimum of scalars, result on all tasks unless root is given. Returns the value."},
    {"scatter",          (PyCFunction)mpi_scatter,      METH_VARARGS,
     "scatter(src, target, root) distributes array from root task."},
    {"gather",           (PyCFunction)mpi_gather,       METH_VARARGS,
     "gather(src, root, target=None) gathers data from all tasks on root task."},
    {"all_gather",       (PyCFunction)mpi_allgather,    METH_VARARGS,
     "all_gather(src, target) gathers data from all tasks on all tasks."},
    {"alltoallv",       (PyCFunction)mpi_alltoallv,    METH_VARARGS,
     "alltoallv(sbuf, scnt, sdispl, rbuf, ...) send data from all tasks to all tasks."},
    {"broadcast",        (PyCFunction)mpi_broadcast,    METH_VARARGS,
     "broadcast(buffer, root) Broadcast data in-place from root task."},
    {"compare",          (PyCFunction)mpi_compare,      METH_VARARGS,
     "compare two communicators for identity using MPI_Comm_compare."},
    {"translate_ranks",  (PyCFunction)mpi_translate_ranks, METH_VARARGS,
     "figure out correspondence between ranks on two communicators."},
    {"get_members",      (PyCFunction)get_members,      METH_VARARGS, 0},
    {"get_c_object",     (PyCFunction)get_c_object,     METH_VARARGS, 0},
    {"new_communicator", (PyCFunction)MPICommunicator,  METH_VARARGS,
     "new_communicator(ranks) creates a new communicator."},
    {0, 0, 0, 0}
};

static PyMemberDef mpi_members[] = {
  {"size", T_INT, offsetof(MPIObject, size), 0, "Number of processors"},
  {"rank", T_INT, offsetof(MPIObject, rank), 0, "Number of this processor"},
  {"parent", T_OBJECT_EX, offsetof(MPIObject, parent), 0, "Parent communicator"},
  {0, 0, 0, 0, 0}  /* Sentinel */
};

// __new__
static PyObject *NewMPIObject(PyTypeObject* type, PyObject *args,
			      PyObject *kwds)
{
    static char *kwlist[] = {NULL};
    MPIObject* self;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
	return NULL;

    self = (MPIObject *) type->tp_alloc(type, 0);
    if (self == NULL)
	return NULL;

    mpi_ensure_initialized();

    MPI_Comm_size(MPI_COMM_WORLD, &(self->size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(self->rank));
    self->comm = MPI_COMM_WORLD;
    Py_INCREF(Py_None);
    self->parent = Py_None;
    self->members = (int*) malloc(self->size*sizeof(int));
    if (self->members == NULL)
	return NULL;
    for (int i=0; i<self->size; i++)
	self->members[i] = i;

    return (PyObject *) self;
}

// __init__ does nothing.
static int InitMPIObject(MPIObject* self, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {NULL};

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist))
    return -1;

  return 0;
}


PyTypeObject MPIType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "MPI",                     /*tp_name*/
  sizeof(MPIObject),         /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)mpi_dealloc,   /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash*/
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "MPI object",              /*tp_doc*/
  0,                         /*tp_traverse*/
  0,                         /*tp_clear*/
  0,                         /*tp_richcompare*/
  0,                         /*tp_weaklistoffset*/
  0,                         /*tp_iter*/
  0,                         /*tp_iternext*/
  mpi_methods,               /*tp_methods*/
  mpi_members,               /*tp_members*/
  0,                         /*tp_getset*/
  0,                         /*tp_base*/
  0,                         /*tp_dict*/
  0,                         /*tp_descr_get*/
  0,                         /*tp_descr_set*/
  0,                         /*tp_dictoffset*/
  (initproc)InitMPIObject,   /*tp_init*/
  0,                         /*tp_alloc*/
  NewMPIObject,              /*tp_new*/
};


static PyObject * MPICommunicator(MPIObject *self, PyObject *args)
{
  PyObject* orig_ranks;
  if (!PyArg_ParseTuple(args, "O", &orig_ranks))
    return NULL;
  // NB: int32 is NPY_LONG on 32-bit Linux and NPY_INT on 64-bit Linux!
  // First convert to NumPy array of NPY_LONG, then cast to NPY_INT, to
  // allow both 32 and 64 bit integers in the argument (except 64 on 32).
  PyArrayObject *ranks = (PyArrayObject*)PyArray_ContiguousFromAny(
					    orig_ranks, NPY_LONG, 1, 1);
  if (ranks == NULL)
    return NULL;
  PyArrayObject *iranks;
  int n = PyArray_DIM(ranks, 0);
  iranks = (PyArrayObject*)PyArray_Cast((PyArrayObject*) ranks, NPY_INT);
  Py_DECREF(ranks);
  if (iranks == NULL)
    return NULL;
  // Check that all ranks make sense
  for (int i = 0; i < n; i++)
    {
      int *x = PyArray_GETPTR1(iranks, i);
      if (*x < 0 || *x >= self->size)
	{
	  Py_DECREF(iranks);
	  PyErr_SetString(PyExc_ValueError, "invalid rank");
	  return NULL;
	}
      for (int j = 0; j < i; j++)
	{
	  int *y = PyArray_GETPTR1(iranks, j);
	  if (*y == *x)
	    {
	      Py_DECREF(iranks);
	      PyErr_SetString(PyExc_ValueError, "duplicate rank");
	      return NULL;
	    }
	}
    }
  MPI_Group group;
  MPI_Comm_group(self->comm, &group);
  MPI_Group newgroup;
  MPI_Group_incl(group, n, (int *) PyArray_BYTES(iranks), &newgroup);
  MPI_Comm comm;
  MPI_Comm_create(self->comm, newgroup, &comm); // has a memory leak!
#ifdef GPAW_MPI_DEBUG
  if (comm != MPI_COMM_NULL)
    {
      // Default Errhandler is MPI_ERRORS_ARE_FATAL
      MPI_Errhandler_set(comm, MPI_ERRORS_RETURN);
#ifdef __bgp__
      int result;
      int rank;
      MPI_Comm_rank(comm, &rank);
      MPIX_Get_property(comm, MPIDO_RECT_COMM, &result);
      if (rank == 0) {
	if(result) fprintf(stderr, "Get_property: comm is rectangular. \n");
      }
#endif
    }
#endif // GPAW_MPI_DEBUG
  MPI_Group_free(&newgroup);
  MPI_Group_free(&group);
  if (comm == MPI_COMM_NULL)
    {
      Py_DECREF(iranks);
      Py_RETURN_NONE;
    }
  else
    {
      MPIObject *obj = PyObject_NEW(MPIObject, &MPIType);
      if (obj == NULL)
	return NULL;
      MPI_Comm_size(comm, &(obj->size));
      MPI_Comm_rank(comm, &(obj->rank));
      obj->comm = comm;
      if (obj->parent == Py_None)
	Py_DECREF(obj->parent);
      obj->members = (int*) malloc(obj->size*sizeof(int));
      if (obj->members == NULL)
	return NULL;
      memcpy(obj->members, (int *) PyArray_BYTES(iranks), obj->size*sizeof(int));
      Py_DECREF(iranks);

      // Make sure that MPI_COMM_WORLD is kept alive till the end (we
      // don't want MPI_Finalize to be called before MPI_Comm_free):
      Py_INCREF(self);
      obj->parent = (PyObject*)self;
      return (PyObject*)obj;
    }
}


PyObject* globally_broadcast_bytes(PyObject *self, PyObject *args)
{
    PyObject *pybytes;
    if(!PyArg_ParseTuple(args, "O", &pybytes)){
        return NULL;
    }

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    long size;
    if(rank == 0) {
        size = PyBytes_Size(pybytes);  // Py_ssize_t --> long
    }
    MPI_Bcast(&size, 1, MPI_LONG, 0, comm);

    char *dst = (char *)malloc(size);
    if(rank == 0) {
        char *src = PyBytes_AsString(pybytes);  // Read-only
        memcpy(dst, src, size);
    }
    MPI_Bcast(dst, size, MPI_BYTE, 0, comm);

    PyObject *value = PyBytes_FromStringAndSize(dst, size);
    free(dst);
    return value;
}

#endif // PARALLEL
