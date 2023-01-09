// Standard C++ includes
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

// Standard C includes
#include <cassert>
#include <csignal>
#include <cstdlib>

// OpenCV includes
#include <opencv2/opencv.hpp>

// GeNN userproject includes
#include "timer.h"

// Model includes
#include "parameters.h"

// Auto-generated simulation code
#include "offline_optical_flow_CODE/definitions.h"

#ifdef _WIN32
    #include <intrin.h>
    int inline clz(unsigned int value)
    {
        unsigned long leadingZero = 0;
        if(_BitScanReverse(&leadingZero, value)) {
            return 31 - leadingZero;
        }
        else {
            return 32;
        }
    }
    // Otherwise, on *nix, use __builtin_clz intrinsic
#else
    #define clz __builtin_clz
#endif
//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
std::atomic<bool> g_Quit{false};

void signalHandler(int)
{
    g_Quit = true;
}

unsigned int getNeuronIndex(unsigned int width, unsigned int x, unsigned int y)
{
    return x + (y * width);
}

float loadEvents(const std::string &filename)
{
    // **HACK** header size
    constexpr std::streamoff headerBytes = 104;

    // Open events file and seek to end
    std::ifstream dataFile(filename, std::ifstream::binary);
    dataFile.seekg(0, std::ifstream::end);

    // Calculate event bytes
    const std::streamoff eventBytes = dataFile.tellg() - headerBytes;

    // Seek to end of header
    dataFile.seekg(headerBytes);

    // Assert that event bytes is a multiple of event size
    assert((eventBytes % (sizeof(uint32_t) * 2)) == 0);

    // Create vector of paurs of uint32_t to hold events
    std::vector<std::pair<uint32_t, uint32_t>> events(eventBytes / (sizeof(uint32_t) * 2));

    // Read events and close
    dataFile.read(reinterpret_cast<char*>(events.data()), eventBytes);
    dataFile.close();

    std::cout << "Read " << events.size() << " events" << std::endl;

    // Swap first and second in pair so events will be lexigraphically sorted by neuron not time
    for (auto &e : events) {
        std::swap(e.first, e.second);
    }
    
    // Remove negative events
    events.erase(std::remove_if(events.begin(), events.end(), 
                                [](std::pair<uint32_t, uint32_t> e)
                                {
                                    const uint32_t p = (e.first & 268435456U) >> 28;
                                    return (p == 0);
                                }), 
                                events.end());

    std::cout << events.size() << " positive polarity events" << std::endl;
    
    // Stash time of last event
    const float lastEventMs = (float)events.back().second / 1000.0f;
    
    std::cout << "Duration " << lastEventMs << "ms" << std::endl;
    
    // Loop through spikes
    std::vector<size_t> numEvents(Parameters::inputWidth * Parameters::inputHeight, 0);
    for (auto &e : events) {
        // Extract X and Y (we no longer care about polarity)
        const uint32_t x = e.first & 16383U;
        const uint32_t y = (e.first & 268419072U) >> 14;
        
        // Replace event with GeNN neuron ID
        e.first = getNeuronIndex(Parameters::inputWidth, x, y);

        // Increment event count
        numEvents[e.first]++;
    }
    
    // Calculate (exclusive) sum
    std::partial_sum(numEvents.cbegin(), numEvents.cend(), numEvents.begin());
    assert(numEvents.back() == events.size());

    // Sort events
    // **NOTE** this will be by neuron and then time
    std::sort(events.begin(), events.end());
    
    // Allocate EGP for DVS data
    allocatespikeTimesDVS(events.size());

    // Copy start and end spikes into variables
    startSpikeDVS[0] = 0;
    std::copy(numEvents.cbegin(), numEvents.cend() - 1, &startSpikeDVS[1]);
    std::copy(numEvents.cbegin(), numEvents.cend(), &endSpikeDVS[0]);

    // Convert event times into floating point milliseconds and copy into EGP
    std::transform(events.cbegin(), events.cend(), spikeTimesDVS,
                   [](std::pair<uint32_t, uint32_t> e) { return (float)e.second / 1000.0f; });
    
    // Upload event times to device
    pushspikeTimesDVSToDevice(events.size());
    
    // Return duration
    return lastEventMs;
}

void buildCentreToMacroConnection(unsigned int *rowLength, unsigned int *ind)
{
    // Calculate start and end of border on each row
    const unsigned int leftBorder = (Parameters::inputWidth - Parameters::centreWidth) / 2;
    const unsigned int rightBorder = leftBorder + Parameters::centreWidth;
    const unsigned int topBorder = (Parameters::inputHeight - Parameters::centreHeight) / 2;
    const unsigned int bottomBorder = topBorder + Parameters::centreHeight;

    // Loop through rows of pixels in centre
    unsigned int i = 0;
    for(unsigned int yi = 0; yi < Parameters::inputHeight; yi++){
        for(unsigned int xi = 0; xi < Parameters::inputWidth; xi++){
            // If we're in the centre
            if(xi >= leftBorder && xi < rightBorder && yi >= topBorder && yi < bottomBorder) {
                const unsigned int yj = (yi - topBorder) / Parameters::kernelSize;
                const unsigned int xj = (xi - leftBorder) / Parameters::kernelSize;
                
                ind[i] = getNeuronIndex(Parameters::macroPixelWidth, xj, yj);
                rowLength[i++] = 1;
            }
            else {
                rowLength[i++] = 0;
            }
        }
    }

    // Check
    assert(i == (Parameters::inputWidth * Parameters::inputHeight));
}

void buildDetectors(unsigned int *excitatoryRowLength, unsigned int *excitatoryInd,
                    unsigned int *inhibitoryRowLength, unsigned int *inhibitoryInd)
{
    // Loop through macro cells
    unsigned int iExcitatory = 0;
    unsigned int iInhibitory = 0;
    for(unsigned int yi = 0; yi < Parameters::macroPixelHeight; yi++){
        for(unsigned int xi = 0; xi < Parameters::macroPixelWidth; xi++){
            // Get index of start of row
            unsigned int sExcitatory = (iExcitatory * Parameters::DetectorMax);
            unsigned int sInhibitory = (iInhibitory * Parameters::DetectorMax);
            
            // If we're not in border region
            if(xi >= 1 && xi < (Parameters::macroPixelWidth - 1)
                && yi >= 1 && yi < (Parameters::macroPixelHeight - 1))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;

                // Add excitatory synapses to all detectors
                for(unsigned int d = 0; d < Parameters::DetectorMax; d++) {
                    excitatoryInd[sExcitatory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                                  xj + d, yj);
                }
                excitatoryRowLength[iExcitatory++] = Parameters::DetectorMax;
            }
            else {
                excitatoryRowLength[iExcitatory++] = 0;
            }


            // Create inhibitory connection to 'left' detector associated with macropixel one to right
            inhibitoryRowLength[iInhibitory] = 0;
            if(xi < (Parameters::macroPixelWidth - 2)
                && yi >= 1 && yi < (Parameters::macroPixelHeight - 1))
            {
                const unsigned int xj = (xi - 1 + 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorLeft, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'right' detector associated with macropixel one to right
            if(xi >= 2
                && yi >= 1 && yi < (Parameters::macroPixelHeight - 1))
            {
                const unsigned int xj = (xi - 1 - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorRight, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'up' detector associated with macropixel one below
            if(xi >= 1 && xi < (Parameters::macroPixelWidth - 1)
                && yi < (Parameters::macroPixelHeight - 2))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 + 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorUp, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'down' detector associated with macropixel one above
            if(xi >= 1 && xi < (Parameters::macroPixelWidth - 1)
                && yi >= 2)
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorDown, yj);
                inhibitoryRowLength[iInhibitory]++;
            }
            iInhibitory++;

        }
    }

    // Check
    assert(iExcitatory == (Parameters::macroPixelWidth * Parameters::macroPixelHeight));
    assert(iInhibitory == (Parameters::macroPixelWidth * Parameters::macroPixelHeight));
}

void displayThreadHandler(std::mutex &inputMutex, const cv::Mat &inputImage, std::mutex &outputMutex, 
                          const float (&output)[Parameters::detectorWidth][Parameters::detectorHeight][Parameters::DetectorAxisMax])
{
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", Parameters::inputWidth * Parameters::inputScale,
                     Parameters::inputHeight * Parameters::inputScale);

    // Create output image
    const unsigned int outputImageWidth = Parameters::detectorWidth * Parameters::outputScale;
    const unsigned int outputImageHeight = Parameters::detectorHeight * Parameters::outputScale;
    cv::Mat outputImage(outputImageHeight, outputImageWidth, CV_8UC3);

    while(!g_Quit){
        // Clear background
        outputImage.setTo(cv::Scalar::all(0));

        {
            std::lock_guard<std::mutex> lock(outputMutex);

            // Loop through output coordinates
            for(unsigned int x = 0; x < Parameters::detectorWidth; x++){
                for(unsigned int y = 0; y < Parameters::detectorHeight; y++){
                    const cv::Point start(x * Parameters::outputScale, y * Parameters::outputScale);
                    const cv::Point end = start + cv::Point(Parameters::outputVectorScale * output[x][y][0],
                                                            Parameters::outputVectorScale * output[x][y][1]);
                    cv::line(outputImage, start, end,
                             CV_RGB(0xFF, 0xFF, 0xFF));
                }
            }
        }

        cv::imshow("Output", outputImage);

        {
            std::lock_guard<std::mutex> lock(inputMutex);
            cv::imshow("Input", inputImage);
        }


        cv::waitKey(33);
    }
}

void applyInputSpikes(uint32_t *inputSpikes, cv::Mat &inputImage)
{
    // Loop through words
    constexpr unsigned int numWords = ((Parameters::inputWidth * Parameters::inputHeight) + 31) / 32;
    for (unsigned int w = 0; w < numWords; w++) {
        // Get word
        uint32_t spikeWord = inputSpikes[w];

        // Calculate neuron id of highest bit of this word
        unsigned int neuronID = (w * 32) + 31;

        // While bits remain
        while (spikeWord != 0) {
            // Calculate leading zeros
            const int numLZ = clz(spikeWord);

            // If all bits have now been processed, zero spike word
            // Otherwise shift past the spike we have found
            spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));

            // Subtract number of leading zeros from neuron ID
            neuronID -= numLZ;

            // Add to pixel in input image
            const auto spikeCoord = std::div((int)neuronID, (int)Parameters::inputWidth);
            inputImage.at<float>(spikeCoord.quot, spikeCoord.rem) += 1.0f;
                        
            // New neuron id of the highest bit of this word
            neuronID--;
        }
    }

    // Decay image
    inputImage *= Parameters::spikePersistence;
}

void applyOutputSpikes(uint32_t *outputSpikes, 
                       float (&output)[Parameters::detectorWidth][Parameters::detectorHeight][Parameters::DetectorAxisMax])
{
    // Loop through words
    constexpr unsigned int numWords = ((Parameters::detectorWidth * Parameters::detectorHeight * Parameters::DetectorMax) + 31) / 32;
    for(unsigned int w = 0; w < numWords; w++) {
        // Get word
        uint32_t spikeWord = outputSpikes[w];
                    
        // Calculate neuron id of highest bit of this word
        unsigned int neuronID = (w * 32) + 31;
                    
        // While bits remain
        while (spikeWord != 0) {
            // Calculate leading zeros
            const int numLZ = clz(spikeWord);

            // If all bits have now been processed, zero spike word
            // Otherwise shift past the spike we have found
            spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));

            // Subtract number of leading zeros from neuron ID
            neuronID -= numLZ;

            // Convert spike ID to x, y, detector
            const auto spikeCoord = std::div((int)neuronID, (int)Parameters::detectorWidth * Parameters::DetectorMax);
            const int spikeY = spikeCoord.quot;
            const auto xCoord = std::div(spikeCoord.rem, (int)Parameters::DetectorMax);
            const int spikeX = xCoord.quot;

            // Apply spike to correct axis of output pixel based on detector it was emitted by
            switch (xCoord.rem) {
            case Parameters::DetectorLeft:
                output[spikeX][spikeY][0] -= 1.0f;
                break;

            case Parameters::DetectorRight:
                output[spikeX][spikeY][0] += 1.0f;
                break;

            case Parameters::DetectorUp:
                output[spikeX][spikeY][1] -= 1.0f;
                break;

            case Parameters::DetectorDown:
                output[spikeX][spikeY][1] += 1.0f;
                break;
            }

            // New neuron id of the highest bit of this word
            neuronID--;
        }
    }

    // Decay output
    for(unsigned int x = 0; x < Parameters::detectorWidth; x++) {
        for(unsigned int y = 0; y < Parameters::detectorHeight; y++){
            for(unsigned int d = 0; d < Parameters::DetectorAxisMax; d++) {
                output[x][y][d] *= Parameters::flowPersistence;
            }
        } 
    }
}
}

int main()
{
    allocateMem();
    allocateRecordingBuffers(1);
    initialize();
    const float lastEventMs = loadEvents("17-03-30_12-53-58_1098500000_1158500000_td.dat");
    buildCentreToMacroConnection(rowLengthDVS_MacroPixel, indDVS_MacroPixel);
    buildDetectors(rowLengthMacroPixel_Flow_Excitatory, indMacroPixel_Flow_Excitatory,
                   rowLengthMacroPixel_Flow_Inhibitory, indMacroPixel_Flow_Inhibitory);

    initializeSparse();
    
    double step = 0.0;
    double render = 0.0;

    std::mutex inputMutex;
    cv::Mat inputImage(Parameters::inputHeight, Parameters::inputWidth, CV_32F);

    std::mutex outputMutex;
    float output[Parameters::detectorWidth][Parameters::detectorHeight][Parameters::DetectorAxisMax] = {0};
    std::thread displayThread(displayThreadHandler,
                              std::ref(inputMutex), std::ref(inputImage),
                              std::ref(outputMutex), std::ref(output));

    // Convert timestep to a duration
    const auto dtDuration = std::chrono::duration<double, std::milli>{DT};

    // Duration counters
    std::chrono::duration<double> sleepTime{0};
    std::chrono::duration<double> overrunTime{0};
    
    // Catch interrupt (ctrl-c) signals
    std::signal(SIGINT, signalHandler);
    const unsigned long long durationTimesteps = (unsigned long long)std::ceil(lastEventMs / DT);
    while(!g_Quit) {
        auto tickStart = std::chrono::high_resolution_clock::now();
        {
            TimerAccumulate timer(step);

            // Simulate
            stepTime();
            pullRecordingBuffersFromDevice();
        }

        {
            TimerAccumulate timer(render);
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                applyOutputSpikes(recordSpkFlow, output);
            }

            {
                std::lock_guard<std::mutex> lock(inputMutex);
                applyInputSpikes(recordSpkDVS, inputImage);
            }
        }

        // Get time of tick start
        auto tickEnd = std::chrono::high_resolution_clock::now();

        // If there we're ahead of real-time pause
        auto tickDuration = tickEnd - tickStart;
        if(tickDuration < dtDuration) {
            auto tickSleep = dtDuration - tickDuration;
            sleepTime += tickSleep;
            std::this_thread::sleep_for(tickSleep);
        }
        else {
            overrunTime += (tickDuration - dtDuration);
        }
        
        // Set quit flag
        if(iT == durationTimesteps) {
            g_Quit = true;
        }
    }

    // Wait for display thread to die
    displayThread.join();

    std::cout << "Ran for " << iT << " " << DT << "ms timesteps, overan for " << overrunTime.count() << "s, slept for " << sleepTime.count() << "s" << std::endl;
    std::cout << "Average step:" << (step * 1000.0) / iT << "s, Render:" << (render * 1000.0) / iT << "s" << std::endl;

    return 0;
}
