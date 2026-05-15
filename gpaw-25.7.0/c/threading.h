/*  Copyright (C) 2009-2012  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. 

 * Helper macro for threading */



#ifndef __THREADED_H
#define __THREADED_H

/*
 * Partitions a range of indices among threads.
 *
 * n    the number of elements in the range
 * tn   the number of threads
 * tid  thread id
 * s    start index
 * e    end index
 */
#define SHARE_WORK(n, tn, tid, s, e) do {               \
    int q = (n) / (tn);                                 \
    int r = (n) % (tn);  /* 0 <= r < q */               \
    *(s) = q * (tid);                                   \
    *(e) = q * ((tid) + 1);                             \
    if (r > 0) {                                        \
        if (r > (tid)) {                                \
            /* Assing this thread one element more. */  \
            *(s) += (tid);                              \
            *(e) += (tid) + 1;                          \
        } else {                                        \
           *(s) += r;                                  \
	   *(e) += r;					\
        }                                               \
    }                                                   \
} while (0)

#endif  /* ! __THREADED_H */
