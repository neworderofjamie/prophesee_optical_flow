// GeNN includes
#include "modelSpec.h"

// Model includes
#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::timestep);
    model.setName("offline_optical_flow");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    NeuronModels::SpikeSourceArray::VarValues dvsInit(
        uninitialisedVar(),     // startSpikes
        uninitialisedVar());    // endSpike

    // LIF model parameters for P population
    NeuronModels::LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        1.0);   // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -60.0,      // 0 - V
        0.0);       // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues dvsMacroPixelWeightUpdateInit(
        0.8);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues macroPixelOutputExcitatoryWeightUpdateInit(
        1.0);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues macroPixelOutputInhibitoryWeightUpdateInit(
        -0.5);     // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues macroPixelPostSynParams(
        5.0);         // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues outputExcitatoryPostSynParams(
        25.0);         // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues outputInhibitoryPostSynParams(
        50.0);         // 0 - TauSyn (ms)

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    auto *dvs = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>(
        "DVS", Parameters::inputWidth * Parameters::inputHeight,
        {}, dvsInit);
    dvs->setSpikeRecordingEnabled(true);
    model.addNeuronPopulation<NeuronModels::LIF>("MacroPixel", Parameters::macroPixelWidth * Parameters::macroPixelHeight,
                                                 lifParams, lifInit);
    auto *flow = model.addNeuronPopulation<NeuronModels::LIF>("Flow", Parameters::detectorWidth * Parameters::detectorHeight * Parameters::DetectorMax,
                                                              lifParams, lifInit);
    flow->setSpikeRecordingEnabled(true);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    auto *dvsMacroPixel = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "DVS_MacroPixel", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "DVS", "MacroPixel",
        {}, dvsMacroPixelWeightUpdateInit,
        macroPixelPostSynParams, {});

    auto *macroPixelFlowExcitatory = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "MacroPixel_Flow_Excitatory", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "MacroPixel", "Flow",
        {}, macroPixelOutputExcitatoryWeightUpdateInit,
        outputExcitatoryPostSynParams, {});

    auto *macroPixelFlowInhibitory = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "MacroPixel_Flow_Inhibitory", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "MacroPixel", "Flow",
        {}, macroPixelOutputInhibitoryWeightUpdateInit,
        outputInhibitoryPostSynParams, {});
    
    dvsMacroPixel->setMaxConnections(1);
    macroPixelFlowExcitatory->setMaxConnections(Parameters::DetectorMax);
    macroPixelFlowInhibitory->setMaxConnections(Parameters::DetectorMax);
    
}
