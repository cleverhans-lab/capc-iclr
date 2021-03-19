#include "emp-sh2pc/emp-sh2pc.h"
#include <fstream>
#include <iostream>
#include <iterator>
using namespace emp;

const int BITLENGTH = 64;
const int MOD_LENGTH = 48;//should be *less* than BITLENGTH
Integer mod(const Integer& a) {
		return a & Integer(BITLENGTH, (1<<MOD_LENGTH) - 1, PUBLIC);
}

int argmax_sum_histogram(std::vector<int> & data, int party) {
    std::vector<Integer> A;
    std::vector<Integer> B;
    
    for(auto v : data) {
        A.push_back(Integer(BITLENGTH, v, ALICE));
    }
    for(auto v : data) {
        B.push_back(Integer(BITLENGTH, v, BOB));
    }
    
    std::vector<Integer> hist;
    
    for(int i=0; i < data.size(); ++i) {
        Integer sum = A[i]+B[i];
        //hist.push_back(sum.reveal<uint32_t>(BOB));
        hist.push_back(sum);
    }
    Integer index(BITLENGTH, 0, PUBLIC);
    Integer max_value = mod(hist[0]);
    for(int i = 1; i < hist.size(); ++i) {
        Integer value = mod(hist[i]);
        Bit greater = value > max_value;
        index = index.select(greater, Integer(BITLENGTH, i, PUBLIC));
        max_value = max_value.select(greater, value);
    }
    int res = index.reveal<uint64_t>(PUBLIC); 
    return res;
}

void fileToVector(const std::string fileName, std::vector<int> &inp){
	std::ifstream inputFile(fileName, std::ios_base::app);
	if (inputFile){
		int value;
		while(inputFile >> value){
			inp.push_back(value);
		}
	}else{
		std::cout << "Cannot open file" << fileName << std::endl;
	}
	std::cout << "done opening file: " << fileName << std::endl;
}
    

int main(int argc, char** argv) {
    //read documentation for argmax_1, it follows the same convention except party 2 (BOB) needs to have the output file passed in
    //
    int port, party;
    parse_party_and_port(argv, &party, &port);
    NetIO *io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port);

    setup_semi_honest(io, party);
    std::vector<int> data;

    fileToVector(argv[3], data);

    int results = argmax_sum_histogram(data, party);
    
    if(party == BOB) {
        std::ofstream out(argv[4]);
        //for(int i=0; i<results.size(); ++i) {
            out << results << std::endl;
        //}
        out.close();
    }
    delete io;
}
