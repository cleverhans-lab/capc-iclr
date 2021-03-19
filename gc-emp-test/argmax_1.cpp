#include "emp-sh2pc/emp-sh2pc.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <iterator>
using namespace std;
using namespace emp;

const int BITLENGTH = 64;
const int MOD_LENGTH = 38;//should be *less* than BITLENGTH
// TODO: tune the mod_length and add tests for the boundaries in the python code
int party;

Integer mod(const Integer& a) {
	return a & Integer(BITLENGTH, (1<<MOD_LENGTH) - 1, PUBLIC);
}
//	argmax((alice+bob) mod 2^MOD_LENGTH )
long long argmax(vector<long long> & data, vector<long long> &rr) {
	vector<Integer> alice;
	vector<Integer> bob;
	if(party == ALICE) {
		for(auto v : data){
			alice.push_back(Integer(BITLENGTH, v + 8388608, ALICE));
			//cout << v << endl;
			rr.push_back(-v);
		}
		for(int i = 0; i < data.size(); ++i)
			bob.push_back(Integer(BITLENGTH, 0, BOB));
	} else {
		for(int i = 0; i < data.size(); ++i)
			alice.push_back(Integer(BITLENGTH, 0, ALICE));
		for(auto v : data)
			bob.push_back(Integer(BITLENGTH, v + 8388608, BOB));
	}

	Integer index(BITLENGTH, 0, PUBLIC);
	Integer max_value = mod(alice[0] + bob[0]);
	cout << "value: " << max_value.reveal<uint64_t>(PUBLIC) << endl;
	for(int i = 1; i < data.size(); ++i) {
		Integer value = mod(alice[i] + bob[i]);
		cout << "value: " << value.reveal<uint64_t>(PUBLIC) << endl;
		Bit greater = value > max_value;
		index = index.select(greater, Integer(BITLENGTH, i, PUBLIC));
		max_value = max_value.select(greater, value);
	}
	long long res = index.reveal<uint64_t>(PUBLIC);
	return res;
}

void fileToVector(const string fileName, vector<long long> &inp){
	ifstream inputFile(fileName, std::ios_base::app);
	if (inputFile){
		long long value;
		while(inputFile >> value){
			inp.push_back(value);		
		}
	}else{
		cout << "Cannot open file" << fileName << endl; 
	}
	cout << "done opening file: " << fileName << endl;
}

int main(int argc, char** argv) {
	// USAGE: party number, port, input file, output file
	// if party == 1 (ALICE) pass noise file name and output file name
	// if party == 2 (BOB) pass logits file name and no output file name needed.
	int port;
	parse_party_and_port(argv, &party, &port);
	NetIO * io = new NetIO(party==ALICE ? nullptr : "127.0.0.1", port);

	auto prot = setup_semi_honest(io, party);
	prot->set_batch_size(1024*1024);//set it to number of bits in BOB's input

	//vector<int> noise;// = {0, 1, 2};
	//vector<int> logits;// = {0, 3, 1};

	//fileToVector("logits1.txt", logits);
	//fileToVector("noise1.txt", noise);

	//cout << "vectors opened" << endl;
	int res;

	if(party == ALICE) {
		vector<long long> noise;
		//logits = {0, 3, 1};
		fileToVector(argv[3], noise);
		//for (int i=0; i<noise.size();i++){
		//	cout << noise[i] << endl;
		//}
		vector<long long > rr;
		res = argmax(noise, rr);
		cout << "argmax: " << res << endl;
		//for(int i=0; i < rr.size(); i++){
		//	cout << rr[i] << endl;
		//}
		rr[res] = rr[res] + 1;
		ofstream out(argv[4]);
		std::copy(rr.begin(), rr.end(),
				         std::ostream_iterator<int>(out, "\n"));
		//for(int i=rr.size()-1;i>=0;i--)
		//	    out<<rr[i]<<"\n";
		cout << "done writing to output" << endl;
		//if (out.is_open()){
		//	out << res;
		//}
		delete io;
	
	}
     	else {
		vector<long long> logits;
		vector<long long> temp;
		//noise = {0, 1, 2};
		fileToVector(argv[3], logits);
		res = argmax(logits, temp);
	}
	//out[res] = out[res] + 1;
	//cout << "writing to output" << endl;
	//ofstream out ("output1.txt");
	//if (out.is_open())
	//{
	//	out << res;
	//}
	//delete io;
}
