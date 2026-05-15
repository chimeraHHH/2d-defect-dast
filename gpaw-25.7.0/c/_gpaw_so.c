#include "_gpaw.h"

PyMODINIT_FUNC PyInit__gpaw(void)
{
    // gpaw-python needs to import arrays at the right time, so this is
    // done in gpaw_main(). For _gpaw.so, we do it here:
    import_array1(0);
    return moduleinit();
}
