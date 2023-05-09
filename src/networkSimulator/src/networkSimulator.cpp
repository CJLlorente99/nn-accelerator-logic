#include <mockturtle/mockturtle.hpp>
#include <mockturtle/algorithms/simulation.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <csvUtilities.hpp>

using namespace std;

int main(int argc, char** argv){
    // Parse arguments
    string aigerFilename = argv[1];
    string inputFilename = argv[2];
    string outputFilename = argv[3];

    // Create CSV parser and load input file
    csvUtilities::CSVReader csvParser(inputFilename, ',');

    // Create CSV writer
    csvUtilities::CSVWriter csvWriter(outputFilename, ',');

    // Load aiger file
    mockturtle::aig_network aig;
    lorina::read_aiger(aigerFilename, mockturtle::aiger_reader( aig ) );

    // Simulate all inputs
    vector<bool> assignment( aig.num_pis() );
    vector<string> columnHeaders;
    for ( auto i = 0u; i < aig.num_pos(); ++i){
        columnHeaders.push_back(fmt::format("N{}", i));
    }
    csvWriter.writeHeader(columnHeaders);

    while(csvParser.readNextLine(&assignment)){
        mockturtle::default_simulator<bool> sim( assignment );
        const auto values = mockturtle::simulate<bool>( aig, sim );
        csvWriter.writeNextLine(values);
    }
}

