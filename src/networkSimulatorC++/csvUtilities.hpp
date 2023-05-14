#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

namespace csvUtilities {
    class CSVReader {
        public:
            CSVReader(std::string filename, char delimiter);
            int readNextLine(std::vector<bool> *inputs);
    };

    class CSVWriter {
        public:
            CSVWriter(std::string filename, char delimiter);
            int writeHeader(std::vector<std::string> columnHeaders);
            int writeNextLine(std::vector<bool> outputs);
    };
}