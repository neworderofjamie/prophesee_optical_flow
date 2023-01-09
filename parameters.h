#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    enum DetectorAxis
    {
        DetectorAxisHorizontal,
        DetectorAxisVertical,
        DetectorAxisMax,
    };

    // Order of detectors associated with each pixel
    enum Detector
    {
        DetectorLeft,
        DetectorRight,
        DetectorUp,
        DetectorDown,
        DetectorMax,
    };

    constexpr double timestep = 1.0;

    constexpr unsigned int inputWidth = 304;
    constexpr unsigned int inputHeight = 240;
    constexpr unsigned int kernelSize = 5;
    constexpr unsigned int centreWidth = 299;
    constexpr unsigned int centreHeight = 235;

    constexpr unsigned int macroPixelWidth = centreWidth / kernelSize;
    constexpr unsigned int macroPixelHeight = centreHeight / kernelSize;

    constexpr unsigned int detectorWidth = macroPixelWidth - 2;
    constexpr unsigned int detectorHeight = macroPixelHeight - 2;

    constexpr unsigned int outputScale = 12;
    constexpr unsigned int inputScale = 2;

    constexpr float flowPersistence = 0.94f;
    constexpr float spikePersistence = 0.97f;

    constexpr float outputVectorScale = 2.0f;
}
