// This file is generated by _generate_slice_elements_c_code.py
// Do not edit it directly.


    // copyright ############################### //
    // This file is part of the Xtrack Package.  //
    // Copyright (c) CERN, 2023.                 //
    // ######################################### //

    #ifndef XTRACK_DRIFT_SLICE_RBEND_H
    #define XTRACK_DRIFT_SLICE_RBEND_H

    #include <headers/track.h>
    #include <beam_elements/elements_src/track_drift.h>


    GPUFUN
    void DriftSliceRBend_track_local_particle(
            DriftSliceRBendData el,
            LocalParticle* part0
    ) {

        double weight = DriftSliceRBendData_get_weight(el);

        #ifndef XSUITE_BACKTRACK
            double const length = weight * DriftSliceRBendData_get__parent_length(el); // m
        #else
            double const length = -weight * DriftSliceRBendData_get__parent_length(el); // m
        #endif

        START_PER_PARTICLE_BLOCK(part0, part);
            Drift_single_particle(part, length);
        END_PER_PARTICLE_BLOCK;
    }

    #endif
    