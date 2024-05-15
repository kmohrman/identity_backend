#include "LST.h"
#include <math.h> 

SDL::LST::LST() {
    TrackLooperDir_ = getenv("LST_BASE");
}

float get_pt(float px, float py) {
    return sqrt(px*px + py*py);
}

float get_eta(float px, float py, float pz) {
    float r = sqrt( px*px + py*py + pz*pz );
    float eta = 0.5 * log( (r+pz)/(r-pz) );
    return eta;
}

float get_phi(float px, float py) {
    return atan2(px,py);
}


std::vector<float> SDL::LST::readRawBuff(const void* input_buffer){

    std::vector<float> out;
    unsigned iter = 0;
    std::cout << "Here  in readRawBuff" << std::endl;
    const float * test_buffer = reinterpret_cast<const float *>(input_buffer);
    //auto test_buffer_size = sizeof(test_buffer) / sizeof(*test_buffer);
    auto test_buffer_size = sizeof(test_buffer) / sizeof(float);
    std::cout << "test_buffer_size????? " << test_buffer_size << std::endl;
    std::cout << "test_buffer_size of ????? " << sizeof(test_buffer) << std::endl;
    std::cout << "test_buffer  : " << test_buffer << std::endl;
    std::cout << "*test_buffer : " << *test_buffer << std::endl;
    std::cout << "test_buffer[0]: " << test_buffer[0] << std::endl;
    std::cout << "test_buffer[1]: " << test_buffer[1] << std::endl;
    std::cout << "test_buffer[2]: " << test_buffer[2] << std::endl;
    std::cout << "*(test_buffer+2): " << *(test_buffer + 2) << std::endl;
    std::cout << "test_buffer[3]: " << test_buffer[3] << std::endl;
    std::cout << "test_buffer[4]: " << test_buffer[4] << std::endl;
    std::cout << "test_buffer[5]: " << test_buffer[5] << std::endl;
    // Loop over the 4 phase2OTHits

    // Get the info about how many phase2OTHits we have in this event
    int itr_main = 0; // This will be the counter as we loop through the flat vector
    int itr_start; // Use this to keep track of where to start each for loop

    // The vectors we'll be filling
    std::vector<int> phase2OTHits_detId;
    std::vector<float> phase2OTHits_x;
    std::vector<float> phase2OTHits_y;
    std::vector<float> phase2OTHits_z;

    std::vector<float> pixelSeeds_px;
    std::vector<float> pixelSeeds_py;
    std::vector<float> pixelSeeds_pz;
    std::vector<float> pixelSeeds_dxy;
    std::vector<float> pixelSeeds_dz;
    std::vector<float> pixelSeeds_ptErr;
    std::vector<float> pixelSeeds_etaErr;
    std::vector<float> pixelSeeds_stateTrajGlbX;
    std::vector<float> pixelSeeds_stateTrajGlbY;
    std::vector<float> pixelSeeds_stateTrajGlbZ;
    std::vector<float> pixelSeeds_stateTrajGlbPx;
    std::vector<float> pixelSeeds_stateTrajGlbPy;
    std::vector<float> pixelSeeds_stateTrajGlbPz;
    std::vector<float> pixelSeeds_q;
    std::vector<std::vector<float>> pixelSeeds_hitIdx;


    ////////////////// Get the phase2OTHits stuff //////////////////
    int n_phase2OTHits = test_buffer[0]; itr_main++;

    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_phase2OTHits; i++){
        phase2OTHits_detId.push_back(test_buffer[i]);
        std::cout << "The phase2OTHits_detId:" <<  int (test_buffer[i]) << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_phase2OTHits; i++){
        phase2OTHits_x.push_back(test_buffer[i]);
        std::cout << "The phase2OTHits_x: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_phase2OTHits; i++){
        phase2OTHits_y.push_back(test_buffer[i]);
        std::cout << "The phase2OTHits_y: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_phase2OTHits; i++){
        phase2OTHits_z.push_back(test_buffer[i]);
        std::cout << "The phase2OTHits_z: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }

    ////////////////// Get the pixelSeeds stuff //////////////////
    int n_pixelSeeds = test_buffer[itr_main]; itr_main++;

    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_px.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_px: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_py.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_py: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_pz.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_pz: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_dxy.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_dxy: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_dz.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_dz: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_ptErr.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_ptErr: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_etaErr.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_etaErr: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_stateTrajGlbX.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_stateTrajGlbX: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_stateTrajGlbY.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_stateTrajGlbY: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_stateTrajGlbZ.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_stateTrajGlbZ: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_stateTrajGlbPx.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_stateTrajGlbPx: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_stateTrajGlbPy.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_stateTrajGlbPy: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_stateTrajGlbPz.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_stateTrajGlbPz: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        pixelSeeds_q.push_back(test_buffer[i]);
        std::cout << "The pixelSeeds_q: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }

    //// Special case for hitIdx, since extra layer of nestedness ////

    // Get the shape vector
    std::vector<float> hitIdx_sizes;
    itr_start = itr_main;
    for (int i=itr_start; i<itr_start+n_pixelSeeds; i++){
        hitIdx_sizes.push_back(test_buffer[i]);
        std::cout << "The hitIdxShape: " <<  test_buffer[i] << std::endl;
        itr_main++;
    }

    // Get the values
    //std::vector<std::vector<float>> hitIdx;
    for (const auto& hitIdx_size : hitIdx_sizes){
        std::cout << "hitIdx_size????? " << hitIdx_size << std::endl;
        std::vector<float> tmp_vec;
        itr_start = itr_main;
        for (int i=itr_start; i<itr_start+hitIdx_size; i++){
            tmp_vec.push_back(test_buffer[i]);
            std::cout << "    The hitIdx: " <<  test_buffer[i] << std::endl;
            itr_main++;
        }
        pixelSeeds_hitIdx.push_back(tmp_vec);
    }

    return out;
}
// TEST END

void SDL::LST::eventSetup() {
    static std::once_flag mapsLoaded;
    std::call_once(mapsLoaded, &SDL::LST::loadMaps, this);
    std::string path = get_absolute_path_after_check_file_exists(TrackLooperDir_ + "/data/centroid_CMSSW_12_2_0_pre2.txt");
    static std::once_flag modulesInited;
    std::call_once(modulesInited, SDL::initModules, path.c_str());
}

void SDL::LST::run(cudaStream_t stream,
                   bool verbose,
                   const std::vector<float> see_px,
                   const std::vector<float> see_py,
                   const std::vector<float> see_pz,
                   const std::vector<float> see_dxy,
                   const std::vector<float> see_dz,
                   const std::vector<float> see_ptErr,
                   const std::vector<float> see_etaErr,
                   const std::vector<float> see_stateTrajGlbX,
                   const std::vector<float> see_stateTrajGlbY,
                   const std::vector<float> see_stateTrajGlbZ,
                   const std::vector<float> see_stateTrajGlbPx,
                   const std::vector<float> see_stateTrajGlbPy,
                   const std::vector<float> see_stateTrajGlbPz,
                   const std::vector<int> see_q,
                   const std::vector<std::vector<int>> see_hitIdx,
                   const std::vector<unsigned int> ph2_detId,
                   const std::vector<float> ph2_x,
                   const std::vector<float> ph2_y,
                   const std::vector<float> ph2_z) {
    auto event = SDL::Event(stream, verbose);
    prepareInput(see_px,
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
                 ph2_z);

    event.addHitToEvent(in_trkX_,
                        in_trkY_,
                        in_trkZ_,
                        in_hitId_,
                        in_hitIdxs_);
    event.addPixelSegmentToEvent(in_hitIndices_vec0_,
                                 in_hitIndices_vec1_,
                                 in_hitIndices_vec2_,
                                 in_hitIndices_vec3_,
                                 in_deltaPhi_vec_,
                                 in_ptIn_vec_, in_ptErr_vec_,
                                 in_px_vec_, in_py_vec_, in_pz_vec_,
                                 in_eta_vec_, in_etaErr_vec_,
                                 in_phi_vec_,
                                 in_charge_vec_,
                                 in_seedIdx_vec_,
                                 in_superbin_vec_,
                                 in_pixelType_vec_,
                                 in_isQuad_vec_);
    event.createMiniDoublets();
    if (verbose) {
        printf("# of Mini-doublets produced: %d\n",event.getNumberOfMiniDoublets());
        printf("# of Mini-doublets produced barrel layer 1: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(0));
        printf("# of Mini-doublets produced barrel layer 2: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(1));
        printf("# of Mini-doublets produced barrel layer 3: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(2));
        printf("# of Mini-doublets produced barrel layer 4: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(3));
        printf("# of Mini-doublets produced barrel layer 5: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(4));
        printf("# of Mini-doublets produced barrel layer 6: %d\n",event.getNumberOfMiniDoubletsByLayerBarrel(5));
        printf("# of Mini-doublets produced endcap layer 1: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(0));
        printf("# of Mini-doublets produced endcap layer 2: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(1));
        printf("# of Mini-doublets produced endcap layer 3: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(2));
        printf("# of Mini-doublets produced endcap layer 4: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(3));
        printf("# of Mini-doublets produced endcap layer 5: %d\n",event.getNumberOfMiniDoubletsByLayerEndcap(4));
    }

    event.createSegmentsWithModuleMap();
    if (verbose) {
        printf("# of Segments produced: %d\n",event.getNumberOfSegments());
        printf("# of Segments produced layer 1-2:  %d\n",event.getNumberOfSegmentsByLayerBarrel(0));
        printf("# of Segments produced layer 2-3:  %d\n",event.getNumberOfSegmentsByLayerBarrel(1));
        printf("# of Segments produced layer 3-4:  %d\n",event.getNumberOfSegmentsByLayerBarrel(2));
        printf("# of Segments produced layer 4-5:  %d\n",event.getNumberOfSegmentsByLayerBarrel(3));
        printf("# of Segments produced layer 5-6:  %d\n",event.getNumberOfSegmentsByLayerBarrel(4));
        printf("# of Segments produced endcap layer 1:  %d\n",event.getNumberOfSegmentsByLayerEndcap(0));
        printf("# of Segments produced endcap layer 2:  %d\n",event.getNumberOfSegmentsByLayerEndcap(1));
        printf("# of Segments produced endcap layer 3:  %d\n",event.getNumberOfSegmentsByLayerEndcap(2));
        printf("# of Segments produced endcap layer 4:  %d\n",event.getNumberOfSegmentsByLayerEndcap(3));
        printf("# of Segments produced endcap layer 5:  %d\n",event.getNumberOfSegmentsByLayerEndcap(4));
    }

    event.createTriplets();
    if (verbose) {
        printf("# of T3s produced: %d\n",event.getNumberOfTriplets());
        printf("# of T3s produced layer 1-2-3: %d\n",event.getNumberOfTripletsByLayerBarrel(0));
        printf("# of T3s produced layer 2-3-4: %d\n",event.getNumberOfTripletsByLayerBarrel(1));
        printf("# of T3s produced layer 3-4-5: %d\n",event.getNumberOfTripletsByLayerBarrel(2));
        printf("# of T3s produced layer 4-5-6: %d\n",event.getNumberOfTripletsByLayerBarrel(3));
        printf("# of T3s produced endcap layer 1-2-3: %d\n",event.getNumberOfTripletsByLayerEndcap(0));
        printf("# of T3s produced endcap layer 2-3-4: %d\n",event.getNumberOfTripletsByLayerEndcap(1));
        printf("# of T3s produced endcap layer 3-4-5: %d\n",event.getNumberOfTripletsByLayerEndcap(2));
        printf("# of T3s produced endcap layer 1: %d\n",event.getNumberOfTripletsByLayerEndcap(0));
        printf("# of T3s produced endcap layer 2: %d\n",event.getNumberOfTripletsByLayerEndcap(1));
        printf("# of T3s produced endcap layer 3: %d\n",event.getNumberOfTripletsByLayerEndcap(2));
        printf("# of T3s produced endcap layer 4: %d\n",event.getNumberOfTripletsByLayerEndcap(3));
        printf("# of T3s produced endcap layer 5: %d\n",event.getNumberOfTripletsByLayerEndcap(4));
    }

    event.createQuintuplets();
    if (verbose) {
        printf("# of Quintuplets produced: %d\n",event.getNumberOfQuintuplets());
        printf("# of Quintuplets produced layer 1-2-3-4-5-6: %d\n",event.getNumberOfQuintupletsByLayerBarrel(0));
        printf("# of Quintuplets produced layer 2: %d\n",event.getNumberOfQuintupletsByLayerBarrel(1));
        printf("# of Quintuplets produced layer 3: %d\n",event.getNumberOfQuintupletsByLayerBarrel(2));
        printf("# of Quintuplets produced layer 4: %d\n",event.getNumberOfQuintupletsByLayerBarrel(3));
        printf("# of Quintuplets produced layer 5: %d\n",event.getNumberOfQuintupletsByLayerBarrel(4));
        printf("# of Quintuplets produced layer 6: %d\n",event.getNumberOfQuintupletsByLayerBarrel(5));
        printf("# of Quintuplets produced endcap layer 1: %d\n",event.getNumberOfQuintupletsByLayerEndcap(0));
        printf("# of Quintuplets produced endcap layer 2: %d\n",event.getNumberOfQuintupletsByLayerEndcap(1));
        printf("# of Quintuplets produced endcap layer 3: %d\n",event.getNumberOfQuintupletsByLayerEndcap(2));
        printf("# of Quintuplets produced endcap layer 4: %d\n",event.getNumberOfQuintupletsByLayerEndcap(3));
        printf("# of Quintuplets produced endcap layer 5: %d\n",event.getNumberOfQuintupletsByLayerEndcap(4));
    }

    event.pixelLineSegmentCleaning();

    event.createPixelQuintuplets();
    if (verbose)
        printf("# of Pixel Quintuplets produced: %d\n",event.getNumberOfPixelQuintuplets());

    event.createPixelTriplets();
    if (verbose)
        printf("# of Pixel T3s produced: %d\n",event.getNumberOfPixelTriplets());

    event.createTrackCandidates();
    if (verbose) {
        printf("# of TrackCandidates produced: %d\n",event.getNumberOfTrackCandidates());
        printf("        # of Pixel TrackCandidates produced: %d\n",event.getNumberOfPixelTrackCandidates());
        printf("        # of pT5 TrackCandidates produced: %d\n",event.getNumberOfPT5TrackCandidates());
        printf("        # of pT3 TrackCandidates produced: %d\n",event.getNumberOfPT3TrackCandidates());
        printf("        # of pLS TrackCandidates produced: %d\n",event.getNumberOfPLSTrackCandidates());
        printf("        # of T5 TrackCandidates produced: %d\n",event.getNumberOfT5TrackCandidates());
    }

    getOutput(event);
}


void SDL::LST::loadMaps() {
    // Module orientation information (DrDz or phi angles)
    std::string endcap_geom = get_absolute_path_after_check_file_exists(TrackLooperDir_ + "/data/endcap_orientation_data_CMSSW_12_2_0_pre2.txt");
    std::string tilted_geom = get_absolute_path_after_check_file_exists(TrackLooperDir_ + "/data/tilted_orientation_data_CMSSW_12_2_0_pre2.txt");
    SDL::endcapGeometry.load(endcap_geom); // centroid values added to the map
    SDL::tiltedGeometry.load(tilted_geom);

    // Module connection map (for line segment building)
    std::string mappath = get_absolute_path_after_check_file_exists(TrackLooperDir_ + "/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt");
    SDL::moduleConnectionMap.load(mappath);

    std::string pLSMapDir = TrackLooperDir_+"/data/pixelmaps_CMSSW_12_2_0_pre2_0p8minPt";

    std::string path;
    path = pLSMapDir + "/pLS_map_layer1_subdet5.txt"; SDL::moduleConnectionMap_pLStoLayer1Subdet5.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_layer2_subdet5.txt"; SDL::moduleConnectionMap_pLStoLayer2Subdet5.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_layer1_subdet4.txt"; SDL::moduleConnectionMap_pLStoLayer1Subdet4.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_layer2_subdet4.txt"; SDL::moduleConnectionMap_pLStoLayer2Subdet4.load(get_absolute_path_after_check_file_exists(path));

    path = pLSMapDir + "/pLS_map_neg_layer1_subdet5.txt"; SDL::moduleConnectionMap_pLStoLayer1Subdet5_neg.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_neg_layer2_subdet5.txt"; SDL::moduleConnectionMap_pLStoLayer2Subdet5_neg.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_neg_layer1_subdet4.txt"; SDL::moduleConnectionMap_pLStoLayer1Subdet4_neg.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_neg_layer2_subdet4.txt"; SDL::moduleConnectionMap_pLStoLayer2Subdet4_neg.load(get_absolute_path_after_check_file_exists(path));

    path = pLSMapDir + "/pLS_map_pos_layer1_subdet5.txt"; SDL::moduleConnectionMap_pLStoLayer1Subdet5_pos.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_pos_layer2_subdet5.txt"; SDL::moduleConnectionMap_pLStoLayer2Subdet5_pos.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_pos_layer1_subdet4.txt"; SDL::moduleConnectionMap_pLStoLayer1Subdet4_pos.load(get_absolute_path_after_check_file_exists(path));
    path = pLSMapDir + "/pLS_map_pos_layer2_subdet4.txt"; SDL::moduleConnectionMap_pLStoLayer2Subdet4_pos.load(get_absolute_path_after_check_file_exists(path));
}

std::string SDL::LST::get_absolute_path_after_check_file_exists(const std::string name) {
    std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
    if (not std::filesystem::exists(fullpath))
    {
        std::cout << "ERROR: Could not find the file = " << fullpath << std::endl;
        exit(2);
    }
    return std::string(fullpath.string().c_str());
}

void SDL::LST::prepareInput(const std::vector<float> see_px,
                            const std::vector<float> see_py,
                            const std::vector<float> see_pz,
                            const std::vector<float> see_dxy,
                            const std::vector<float> see_dz,
                            const std::vector<float> see_ptErr,
                            const std::vector<float> see_etaErr,
                            const std::vector<float> see_stateTrajGlbX,
                            const std::vector<float> see_stateTrajGlbY,
                            const std::vector<float> see_stateTrajGlbZ,
                            const std::vector<float> see_stateTrajGlbPx,
                            const std::vector<float> see_stateTrajGlbPy,
                            const std::vector<float> see_stateTrajGlbPz,
                            const std::vector<int> see_q,
                            const std::vector<std::vector<int>> see_hitIdx,
                            const std::vector<unsigned int> ph2_detId,
                            const std::vector<float> ph2_x,
                            const std::vector<float> ph2_y,
                            const std::vector<float> ph2_z) {
    unsigned int count = 0;
    auto n_see = see_stateTrajGlbPx.size();
    std::vector<float> px_vec;
    px_vec.reserve(n_see);
    std::vector<float> py_vec;
    py_vec.reserve(n_see);
    std::vector<float> pz_vec;
    pz_vec.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec0;
    hitIndices_vec0.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec1;
    hitIndices_vec1.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec2;
    hitIndices_vec2.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec3;
    hitIndices_vec3.reserve(n_see);
    std::vector<float> ptIn_vec;
    ptIn_vec.reserve(n_see);
    std::vector<float> ptErr_vec;
    ptErr_vec.reserve(n_see);
    std::vector<float> etaErr_vec;
    etaErr_vec.reserve(n_see);
    std::vector<float> eta_vec;
    eta_vec.reserve(n_see);
    std::vector<float> phi_vec;
    phi_vec.reserve(n_see);
    std::vector<int> charge_vec;
    charge_vec.reserve(n_see);
    std::vector<unsigned int> seedIdx_vec;
    seedIdx_vec.reserve(n_see);
    std::vector<float> deltaPhi_vec;
    deltaPhi_vec.reserve(n_see);
    std::vector<float> trkX = ph2_x;
    std::vector<float> trkY = ph2_y;
    std::vector<float> trkZ = ph2_z;
    std::vector<unsigned int> hitId = ph2_detId;
    std::vector<unsigned int> hitIdxs(ph2_detId.size());

    std::vector<int> superbin_vec;
    std::vector<int8_t> pixelType_vec;
    std::vector<short> isQuad_vec;
    std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
    const int hit_size = trkX.size();

    for (auto &&[iSeed, _] : iter::enumerate(see_stateTrajGlbPx)) {
        //ROOT::Math::PxPyPzMVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed], 0);
        //std::vector<std::vector<float>> p3LH = {{see_stateTrajGlbPx[iSeed]}, {see_stateTrajGlbPy[iSeed]}, {see_stateTrajGlbPz[iSeed]}};
        float p3LH_px = see_stateTrajGlbPx[iSeed];
        float p3LH_py = see_stateTrajGlbPy[iSeed];
        float p3LH_pz = see_stateTrajGlbPz[iSeed];
        float ptIn = get_pt(p3LH_px,p3LH_py); //p3LH.Pt();
        float eta = get_eta(p3LH_px,p3LH_py,p3LH_pz);
        float ptErr = see_ptErr[iSeed];
        //ROOT::Math::XYZVector p3LH_helper(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);

        if ((ptIn > 0.8 - 2 * ptErr)) {
            //ROOT::Math::XYZVector r3LH(see_stateTrajGlbX[iSeed], see_stateTrajGlbY[iSeed], see_stateTrajGlbZ[iSeed]);
            float r3LH_x = see_stateTrajGlbX[iSeed];
            float r3LH_y = see_stateTrajGlbY[iSeed];
            float r3LH_z = see_stateTrajGlbZ[iSeed];

            //ROOT::Math::PxPyPzMVector p3PCA(see_px[iSeed], see_py[iSeed], see_pz[iSeed], 0);
            float p3PCA_px = see_px[iSeed];
            float p3PCA_py = see_py[iSeed];
            float p3PCA_pz = see_pz[iSeed];

            //ROOT::Math::XYZVector r3PCA(calculateR3FromPCA(p3PCA, see_dxy[iSeed], see_dz[iSeed]));
            std::vector<float> r3PCA(calculateR3FromPCA(p3PCA_px, p3PCA_py, p3PCA_pz, see_dxy[iSeed], see_dz[iSeed]));
            float r3PCA_x = r3PCA[0];
            float r3PCA_y = r3PCA[1];
            float r3PCA_z = r3PCA[2];

            float pixelSegmentDeltaPhiChange = get_phi(r3LH_x-p3LH_px, r3LH_y-p3LH_py);
            float etaErr = see_etaErr[iSeed];
            float px = p3LH_px;
            float py = p3LH_py;
            float pz = p3LH_pz;

            int charge = see_q[iSeed];
            int pixtype = -1;

            if (ptIn >= 2.0) pixtype = 0;
            else if (ptIn >= (0.8 - 2 * ptErr) and ptIn < 2.0) {
                if (pixelSegmentDeltaPhiChange >= 0) pixtype =1;
                else pixtype = 2;
            }
            else continue;

            unsigned int hitIdx0 = hit_size + count;
            count++; 
            unsigned int hitIdx1 = hit_size + count;
            count++;
            unsigned int hitIdx2 = hit_size + count;
            count++;
            unsigned int hitIdx3;
            if (see_hitIdx[iSeed].size() <= 3) hitIdx3 = hitIdx2;
            else {
                hitIdx3 = hit_size + count;
                count++;
            }

            trkX.push_back(r3PCA_x);
            trkY.push_back(r3PCA_y);
            trkZ.push_back(r3PCA_z);
            trkX.push_back(get_pt(p3PCA_px,p3PCA_py));
            float p3PCA_Eta = get_eta(p3PCA_px,p3PCA_py,p3PCA_pz);
            trkY.push_back(p3PCA_Eta);
            float p3PCA_Phi = get_phi(p3PCA_px,p3PCA_py);
            trkZ.push_back(p3PCA_Phi);
            trkX.push_back(r3LH_x);
            trkY.push_back(r3LH_y);
            trkZ.push_back(r3LH_z);
            hitId.push_back(1);
            hitId.push_back(1);
            hitId.push_back(1);
            if(see_hitIdx[iSeed].size() > 3) {
                trkX.push_back(r3LH_x);
                trkY.push_back(see_dxy[iSeed]);
                trkZ.push_back(see_dz[iSeed]);
                hitId.push_back(1);
            }
            px_vec.push_back(px);
            py_vec.push_back(py);
            pz_vec.push_back(pz);

            hitIndices_vec0.push_back(hitIdx0);
            hitIndices_vec1.push_back(hitIdx1);
            hitIndices_vec2.push_back(hitIdx2);
            hitIndices_vec3.push_back(hitIdx3);
            ptIn_vec.push_back(ptIn);
            ptErr_vec.push_back(ptErr);
            etaErr_vec.push_back(etaErr);
            eta_vec.push_back(eta);
            float phi = get_phi(p3LH_px,p3LH_py);
            phi_vec.push_back(phi);
            charge_vec.push_back(charge);
            seedIdx_vec.push_back(iSeed);
            deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

            hitIdxs.push_back(see_hitIdx[iSeed][0]);
            hitIdxs.push_back(see_hitIdx[iSeed][1]);
            hitIdxs.push_back(see_hitIdx[iSeed][2]);
            bool isQuad = false;
            if(see_hitIdx[iSeed].size() > 3) {
                isQuad = true;
                hitIdxs.push_back(see_hitIdx[iSeed][3]);
            }
            float neta = 25.;
            float nphi = 72.;
            float nz = 25.;
            int etabin = (p3PCA_Eta + 2.6) / ((2*2.6)/neta);
            int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2.*3.14159265358979323846) / nphi);
            int dzbin = (see_dz[iSeed] + 30) / (2*30 / nz);
            int isuperbin = (nz * nphi) * etabin + (nz) * phibin + dzbin;
            superbin_vec.push_back(isuperbin);
            pixelType_vec.push_back(pixtype);
            isQuad_vec.push_back(isQuad);
        }
    }

    in_trkX_ = trkX;
    in_trkY_ = trkY;
    in_trkZ_ = trkZ;
    in_hitId_ = hitId;
    in_hitIdxs_ = hitIdxs;
    in_hitIndices_vec0_ = hitIndices_vec0;
    in_hitIndices_vec1_ = hitIndices_vec1;
    in_hitIndices_vec2_ = hitIndices_vec2;
    in_hitIndices_vec3_ = hitIndices_vec3;
    in_deltaPhi_vec_ = deltaPhi_vec;
    in_ptIn_vec_ = ptIn_vec;
    in_ptErr_vec_ = ptErr_vec;
    in_px_vec_ = px_vec;
    in_py_vec_ = py_vec;
    in_pz_vec_ = pz_vec;
    in_eta_vec_ = eta_vec;
    in_etaErr_vec_ = etaErr_vec;
    in_phi_vec_ = phi_vec;
    in_charge_vec_ = charge_vec;
    in_seedIdx_vec_ = seedIdx_vec;
    in_superbin_vec_ = superbin_vec;
    in_pixelType_vec_ = pixelType_vec;
    in_isQuad_vec_ = isQuad_vec;
}

//ROOT::Math::XYZVector SDL::LST::calculateR3FromPCA(const ROOT::Math::PxPyPzMVector& p3, const float dxy, const float dz) {
std::vector<float> SDL::LST::calculateR3FromPCA(float p3_x, float p3_y, float p3_z, const float dxy, const float dz) {
    const float pt = get_pt(p3_x,p3_y);
    const float p = sqrt(p3_x*p3_x + p3_y*p3_y + p3_z*p3_z);
    const float vz = dz*pt*pt/p/p;

    const float vx = -dxy*p3_y/pt - p3_x/p*p3_z/p*dz;
    const float vy =    dxy*p3_x/pt - p3_y/p*p3_z/p*dz;
    return std::vector<float> {{vx}, {vy}, {vz}};
}

void SDL::LST::getOutput(SDL::Event& event) {
    std::vector<std::vector<unsigned int>> tc_hitIdxs_;
    std::vector<unsigned int> tc_len_;
    std::vector<int> tc_seedIdx_;
    std::vector<short> tc_trackCandidateType_;

    SDL::hits& hitsInGPU = (*event.getHitsInCMSSW());
    SDL::trackCandidates& trackCandidatesInGPU = (*event.getTrackCandidatesInCMSSW());

    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
    for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
        short trackCandidateType = trackCandidatesInGPU.trackCandidateType[idx];
        std::vector<unsigned int> hit_idx = getHitIdxs(trackCandidateType, idx, trackCandidatesInGPU.hitIndices, hitsInGPU.idxs);

        tc_hitIdxs_.push_back(hit_idx);
        tc_len_.push_back(hit_idx.size());
        tc_seedIdx_.push_back(trackCandidatesInGPU.pixelSeedIndex[idx]);
        tc_trackCandidateType_.push_back(trackCandidateType);
    }

    out_tc_hitIdxs_ = tc_hitIdxs_;
    out_tc_len_ = tc_len_;
    out_tc_seedIdx_ = tc_seedIdx_;
    out_tc_trackCandidateType_ = tc_trackCandidateType_;
}

std::vector<unsigned int> SDL::LST::getHitIdxs(const short trackCandidateType, const unsigned int TCIdx, const unsigned int* TCHitIndices, const unsigned int* hitIndices) {
    std::vector<unsigned int> hits;

    unsigned int maxNHits = 0;
    if (trackCandidateType == 7) maxNHits = 14; // pT5
    else if (trackCandidateType == 5) maxNHits = 10; // pT3
    else if (trackCandidateType == 4) maxNHits = 10; // T5
    else if (trackCandidateType == 8) maxNHits = 4; // pLS

    for (unsigned int i=0; i<maxNHits; i++) {
        unsigned int hitIdxInGPU = TCHitIndices[14 * TCIdx + i];
        unsigned int hitIdx = (trackCandidateType == 8) ? hitIdxInGPU : hitIndices[hitIdxInGPU]; // Hit indices are stored differently in the standalone for pLS.

        // For p objects, the 3rd and 4th hit maybe the same,
        // due to the way pLS hits are stored in the standalone.
        // This is because pixel seeds can be either triplets or quadruplets.
        if (trackCandidateType != 4 && hits.size() == 3 && hits.back() == hitIdx) // Remove duplicate 4th hits.
            continue;

        hits.push_back(hitIdx);
    }

    return hits;
}
