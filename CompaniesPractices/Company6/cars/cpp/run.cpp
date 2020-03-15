#include <iostream>
#include <fstream>

#include "utils.h"
#include "environment.h"
#include "track.h"

int Predict(const std::vector<double>& state) {
    return rand() % 4;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }
    std::ifstream trackFile(argv[1]);
    TTrack track = TTrack::ReadFrom(trackFile);
    TEnvironment env(track);

    bool broken = false;
    bool done = false;
    int i = 0;

    TStepResult result;

    while (!broken && !done) {
        int action = Predict(env.State());
        result = env.Step(action);
        broken = result.Broken;
        done = result.Done;
    }
    std::cout << result.Progress << std::endl;
}
