#include "_gpaw.h"

int
gpaw_main()
{
    int status = -1;

    PyObject *gpaw_mod = NULL, *pymain = NULL;

    gpaw_mod = PyImport_ImportModule("gpaw");
    if(gpaw_mod == NULL) {
        status = 3;  // Basic import failure
    } else {
        pymain = PyObject_GetAttrString(gpaw_mod, "main");
    }

    if(pymain == NULL) {
        status = 4;  // gpaw.main does not exist for some reason
        //PyErr_Print();
    } else {
        // Returns Py_None or NULL (error after calling user script)
        // We already imported the Python parts of numpy.  If we want, we can
        // later attempt to broadcast the numpy C API imports, too.
        // However I don't know how many files they are, and we need to
        // figure out how to broadcast extension modules (shared objects).
        import_array1(0);
        PyObject *pyreturn = PyObject_CallFunction(pymain, "");
        status = (pyreturn == NULL);
        Py_XDECREF(pyreturn);
    }

    Py_XDECREF(pymain);
    Py_XDECREF(gpaw_mod);
    return status;
}


int
main(int argc, char **argv)
{
#ifdef GPAW_GPU
    int granted;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &granted);
    if (granted < MPI_THREAD_MULTIPLE)
        exit(1);
#else
#ifndef _OPENMP
    MPI_Init(&argc, &argv);
#else
    int granted;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &granted);
    if (granted != MPI_THREAD_MULTIPLE)
        exit(1);
#endif
#endif

#define PyChar wchar_t
    wchar_t* wargv[argc];
    wchar_t* wargv2[argc];
    for (int i = 0; i < argc; i++) {
        int n = 1 + mbstowcs(NULL, argv[i], 0);
        wargv[i] = (wchar_t*)malloc(n * sizeof(wchar_t));
        wargv2[i] = wargv[i];
        mbstowcs(wargv[i], argv[i], n);
    }

    Py_SetProgramName(wargv[0]);
    PyImport_AppendInittab("_gpaw", &moduleinit);
    Py_Initialize();
    PySys_SetArgvEx(argc, wargv, 0);

#ifdef GPAW_WITH_ELPA
    // Globally initialize Elpa library if present:
    if (elpa_init(20171201) != ELPA_OK) {
        // What API versions do we support?
        PyErr_SetString(PyExc_RuntimeError, "Elpa >= 20171201 required");
        PyErr_Print();
        return 1;
    }
#endif

    int status = gpaw_main();

    if(status != 0) {
        PyErr_Print();
    }

#ifdef GPAW_WITH_ELPA

#ifdef ELPA_API_VERSION
    // Newer Elpas define their version but older ones don't.
    int elpa_err;
    elpa_uninit(&elpa_err);
#else
    elpa_uninit();  // 2018.05.001: no errcode
#endif

#endif

    Py_Finalize();
    MPI_Finalize();

    for (int i = 0; i < argc; i++)
        free(wargv2[i]);

    return status;
}
