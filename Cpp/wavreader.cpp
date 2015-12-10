# include "/usr/local/include/aquila/global.h"
# include "/usr/local/include/aquila/source/WaveFile.h"
# include "/usr/local/include/aquila/source/generator/SineGenerator.h"
# include "/usr/local/include/aquila/transform/Mfcc.h"
# include <cmath>
# include <vector>
# include <cstdlib>
# include <algorithm>
# include <cstddef>
# include <iostream>
# include <iterator>
# include <functional>

using namespace std;

template<typename T>
vector<T> windowfunc(vector<T>& a, int x){
	return a;
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in){
	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);
	double delta = (end - start) / (num - 1);

	std::vector<double> linspaced(num - 1);
	for (int i = 0; i < num; ++i){
		linspaced[i] = start + delta * i;
	}
	linspaced.push_back(end);
	return linspaced;
}

template<typename T>
vector<vector<T> > tile(vector<T> signal, int rows, int cols){
	vector< vector<T> > vec;
	for (int i = 0; i < rows; ++i){
		vector<int> v;
		vec.push_back(v);
		for (int j = 0; j < cols; ++j){
			vec[i].insert(vec[i].begin(), signal.begin(), signal.end());
		}
	}
	return vec;
}

template<typename T>
vector<vector<T> > Transpose(const vector<vector<T> > data) {
    vector<vector<T> > result(data[0].size(), vector<T>(data.size()));
    for (int i = 0; i < data[0].size(); i++) 
        for (int j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

template<typename T>
void MultiplyingMatrix(const vector<vector<T> >& A, const vector<vector<T> >& B, vector< vector<T> > &C){
    int vrows;
    int vcols;
    vrows = A.size();
    vcols = A[1].size();
 
    for (int i = 0; i < vrows; i++){
        for (int j = 0; j < vcols; j++){
            for (int rwcl = 0; rwcl < vcols; rwcl++){
              C[i][j] += (A[i][rwcl] * B[rwcl][j]);
            }
         }
    }
}

template <typename T>
vector<vector<T> > operator+(const vector<vector<T> >& a, const vector<vector<T> >& b){
    assert(a.size() == b.size());

    vector<vector<T> > result;
    result.reserve(a.size());

    transform(a.begin(), a.end(), b.begin(), back_inserter(result), plus<T>());
    return result;
}

template<typename T>
vector< vector<T> > frameSignal(vector<T>& signal, int flen, int fstep){
	int slen = signal.size();
	int framelen = static_cast<int>(flen);
	int framestep = static_cast<int>(fstep);
	int numframes;
	int z = 0;
	int padlen;
	vector< vector<T> > indices, frames, win, result;

	T zero = static_cast<T>(z);
	 
	if(slen < framelen)
		numframes = 1;
	else
		numframes = 1 + static_cast<int>(ceil((1.0 * slen - framelen) / framestep));
	padlen = static_cast<int>((numframes - 1) * framestep + framelen);
	
	signal.resize(padlen);

	indices = tile(linspace(0, framelen, framelen), numframes, 1) + Transpose(tile(linspace(0, numframes * framestep, numframes), framelen, 1));
	
	for (int i = 0; i < indices.size(); ++i){
		vector<T> f;
		frames.push_back(f);
		for (int j = 0; j < indices[0].size(); ++j){
			frames[i].push_back(signal[j]);
		}
	}

    win = tile(windowfunc(signal, framelen),numframes,1);
    
   	MultiplyingMatrix(frames, win, result); 
	
	return result;

}

double hzToMel(double hertz) {
	return (2595 * log10(1 + hertz / 700.0));
}

double melToHz(double mel){
	return 700 * pow(10, (mel / 2595.0) - 1);
}

template<typename T>
vector< vector <T> > getFilterBanks(float nfilt, float nfft = 512, float samplerate = 16000, float lowfreq = 0, float highfreq = 0) { 	//Pass half of sampling rate as var highfreq
	double highFrequency = samplerate / 2;
	double lowMel, highMel;
	vector<double> melPoints;
	int i;

	lowMel = hzToMel(lowfreq);
	highMel = hzToMel(highFrequency);

	melPoints = linspace(lowMel, highMel, (nfilt + 2));

	i = 0;
	vector<T> bin;
	for(double d : melPoints){
		bin.push_back(floor((nfft + 1) * melToHz(d) / samplerate));
		i++;
	}

	vector< vector <T> > fbank[(int)nfilt][(int)nfft/(2 + 1)];
	for(int j = 0 ; j < (int)nfilt ; j++)
		for(int k = 0 ; k < (int)(nfft/(2 + 1)) ; k++)
			fbank[j][k] = 0;

	for(int j = 0 ; j < (int)nfilt ; j++){
		for(int k = (int)bin[j] ; k < (int)bin[j + 1] ; k++)
			fbank[j][k] = (k - bin[j]) / (bin[j + 1] - bin[j]);
		for(int k = (int)bin[j + 1] ; k < (int)bin[j + 2] ; k++)
			fbank[j][k] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1]);
	}

	return fbank;
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		cout << "Usage: wavReader <wav file>\n";
		exit(0);
	}

	Aquila::WaveFile wav(argv[1]);
	cout << "\nWav file name: "    << wav.getFilename();
	cout << "\nLength: "           << wav.getAudioLength()        << " ms";
	cout << "\nSample frequency: " << wav.getSampleFrequency()    << " Hz";
	cout << "\nChannels: "         << wav.getChannelsNum();
	cout << "\nByte rate: "        << wav.getBytesPerSec() / 1024 << " kB/s";
	cout << "\nBits per sample: "  << wav.getBitsPerSample()      << "b\n";
	cout << endl;

	int sampleRate = wav.getSampleFrequency();
	vector<double> doubleWav;
	for(auto it = wav.begin() ; it != wav.end() ; it++){
		doubleWav.push_back(*it);
	}

	/*const Aquila::FrequencyType sampleFrequency = wav.getSampleFrequency();
	Aquila::SineGenerator input(sampleFrequency);
	input.setAmplitude(5).setFrequency(64).generate(1024);

	Aquila::Mfcc mfcc(input.getSamplesCount());
	auto mfccValues = mfcc.calculate(input);
	cout << "MFCC coefficients: \n";
	std::copy(
	    std::begin(mfccValues),
	    std::end(mfccValues),
	    std::ostream_iterator<double>(std::cout, " ")
	);
	cout << "\n";*/
	vector< vector<double> > filterBanks = getFilterBanks((float)26, (float)512, (float)sampleRate, (float)0, (float)0);	
}