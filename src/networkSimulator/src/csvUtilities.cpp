#include <csvUtilities.hpp>

namespace csvUtilities {
    class csvReader {
        public:
            std::ifstream file;
            char delimiter;

            csvReader(std::string filename, char delimiter){
                this->file = std::ifstream(filename);
                this->delimiter = delimiter;
            }

            int readNextLine(std::vector<bool> *inputs){
                std::vector<bool> row;
                std::string line, word;
                bool aux;
                while (std::getline(this->file, line)) {
                {
                    std::stringstream ss(line);

                    while (std::getline(ss, word, this->delimiter))
                        if (word == "1")
                            aux = true;
                        else if (word == "0")
                            aux = false;
                        row.push_back(aux);
                    *inputs = row;
                    return 1;   
                }
                return 0;   
            }
        }
    };

    class csvWriter {
        public:
            std::ofstream file;
            char delimiter;

            csvWriter(std::string filename, char delimiter){
                this->file = std::ofstream(filename);
                this->delimiter = delimiter;
            }

            int writeHeader(std::vector<std::string> columnHeaders){
                for (int i = 0; i < columnHeaders.size(); i++)
                    this->file << columnHeaders[i] << this->delimiter;
                this->file << std::endl;
                return 1;
            }

            int writeNextLine(std::vector<bool> outputs){
                for (int i = 0; i < outputs.size(); i++)
                    this->file << outputs[i] << this->delimiter;
                this->file << std::endl;
                return 1;
            }
    };
}