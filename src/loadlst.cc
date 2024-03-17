#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "../SDL/LST.h"

//___________________________________________________________________________________________________________________________________________________________________________________________

void run_sdl(
        cudaStream_t stream,
        bool verbose,
        std::vector<float> see_px,
        std::vector<float> see_py,
        std::vector<float> see_pz,
        std::vector<float> see_dxy,
        std::vector<float> see_dz,
        std::vector<float> see_ptErr,
        std::vector<float> see_etaErr,
        std::vector<float> see_stateTrajGlbX,
        std::vector<float> see_stateTrajGlbY,
        std::vector<float> see_stateTrajGlbZ,
        std::vector<float> see_stateTrajGlbPx,
        std::vector<float> see_stateTrajGlbPy,
        std::vector<float> see_stateTrajGlbPz,
        std::vector<int> see_q,
        std::vector<std::vector<int>> see_hitIdx,
        std::vector<unsigned int> ph2_detId,
        std::vector<float> ph2_x,
        std::vector<float> ph2_y,
        std::vector<float> ph2_z
    ) {
    std::cout << "Hi from run_sdl!" << std::endl;

    SDL::LST lst;

    lst.eventSetup();

    lst.run(
        stream,
        verbose,
        see_px,
        see_py,
        see_pz,
        see_dxy,
        see_dz,
        see_ptErr,
        see_etaErr,
        see_stateTrajGlbX,
        see_stateTrajGlbY,
        see_stateTrajGlbZ,
        see_stateTrajGlbPx,
        see_stateTrajGlbPy,
        see_stateTrajGlbPz,
        see_q,
        see_hitIdx,
        ph2_detId,
        ph2_x,
        ph2_y,
        ph2_z
    );


    // Try to look at the output info
    std::vector<std::vector<unsigned int>> hits;
    std::vector<unsigned int> len;
    std::vector<int> seedIdx;
    std::vector<short> trackCandidateType;
    hits = lst.hits();
    len = lst.len();
    seedIdx = lst.seedIdx();
    trackCandidateType = lst.trackCandidateType();

    std::cout << "Hits 0,0: " << hits[0][0] << std::endl;
    std::cout << "Len: " << len[0] << std::endl;
    std::cout << "seedIdx: " << seedIdx[0] << std::endl;
    std::cout << "trackCandidateType: " << trackCandidateType[0] << std::endl;


}

//int main() {
//    std::cout << "HELLO world from sonic main" << std::endl;
//
//    // Run the code
//    run_sdl();
//
//    return 0;
//}

