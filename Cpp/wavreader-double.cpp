# include <aquila/global.h>
# include <aquila/source/WaveFile.h>
# include <aquila/source/generator/SineGenerator.h>
# include <aquila/transform/Mfcc.h>
# include <aquila/transform/FftFactory.h>
# include <cmath>
# include <vector>
# include <cstdlib>
# include <algorithm>
# include <cstddef>
# include <iostream>
# include <iterator>
# include <functional>
# include <cassert>

using namespace std;

vector<double> windowfunc(vector<double>& a, int x){
	return a;
}

double rootMeanSquare(vector<double> energy){
	double squaredAverage = 0;
	for(int i = 0 ; i < energy.size() ; i++){
		squaredAverage += energy[i] * energy[i];
	}
	return pow((squaredAverage/energy.size()), 0.5);
}

std::vector<double> linspace(double start_in, double end_in, int num_in){
	cout << "In linspace" << endl;
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

vector<vector<double> > tile(vector<double> signal, int rows, int cols){
	cout <<"In tile." << endl;
	vector< vector<double> > vec;
	for (int i = 0; i < rows; ++i){
		vector<double> v;
		vec.push_back(v);
		cout << "Pushing back i: " << i << " " << sizeof(vec) << endl;
		for (int j = 0; j < cols ; ++j){
			vec[i].insert(vec[i].begin(), signal.begin(), signal.end());
		}
	}
	return vec;
}

vector<vector<double> > Transpose(vector<vector<double> > data) {
	cout << "Transposing, yo." << endl;
    vector<vector<double> > result(data[0].size(), vector<double>(data.size()));
    for (int i = 0; i < data[0].size(); i++) 
        for (int j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

void MultiplyingMatrix(vector<vector<double> >& A, vector<vector<double> >& B, vector< vector<double> > &C){
	cout << "Multiplying Matrices" << endl;
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

vector<vector<double> > add_mat(vector<vector<double> > mat1, vector<vector<double> > mat2){
	cout << "Adding Matrices" << endl;
	for (int i = 0; i < mat1.size() - 1; ++i)
	{
		for(int j = 0; j < mat1[0].size() - 1; j++){
			mat1[i][j] += mat2[i][j];
		}
	}
	return mat1;
}

vector< vector<double> > frameSignal(vector<double>& signal, int flen, int fstep){
	cout << "Inside frameSignal." << endl;
	int slen = signal.size();
	int framelen = static_cast<int>(flen);
	int framestep = static_cast<int>(fstep);
	int numframes;
	//int z = 0;
	int padlen;
	vector< vector<double> > indices, frames, win, result;

	//double zero = static_cast<double>(z);
	 
	if(slen < framelen)
		numframes = 1;
	else
		numframes = 1 + static_cast<int>(ceil((1.0 * slen - framelen) / framestep));
	padlen = static_cast<int>((numframes - 1) * framestep + framelen);
	
	signal.resize(padlen);

	indices = add_mat(tile(linspace(0, framelen, framelen), numframes, 1), Transpose(tile(linspace(0, numframes * framestep, numframes), framelen, 1)));
	
	for (int i = 0; i < indices.size(); ++i){
		vector<double> f;
		frames.push_back(f);
		for (int j = 0; j < indices[0].size(); ++j){
			frames[i].push_back(signal[j]);
		}
	}

    win = tile(windowfunc(signal, framelen),numframes,1);
    
    for (int i = 0; i < frames.size(); ++i){
		vector<double> v;
		result.push_back(v);
		for (int j = 0; j < win[1].size(); ++j){
			result[i].push_back(0);
		}
	}

   	MultiplyingMatrix(frames, win, result); 
	
	return result;

}

double hzToMel(double hertz) {
	return (2595 * log10(1 + hertz / 700.0));
}

double melToHz(double mel){
	return 700 * pow(10, (mel / 2595.0) - 1);
}

vector< vector <double> > getFilterBanks(float nfilt, float nfft, float samplerate, float lowfreq, double highfreq) { 	//Pass half of sampling rate as var highfreq
	double highFrequency = highfreq;
	double lowMel, highMel;
	vector<double> melPoints;
	int i;

	lowMel = hzToMel(lowfreq);
	highMel = hzToMel(highFrequency);

	melPoints = linspace(lowMel, highMel, (nfilt + 2));

	i = 0;
	vector<double> bin;
	for(double d : melPoints){
		bin.push_back(floor((nfft + 1) * melToHz(d) / samplerate));
		i++;
	}

	vector< vector <double> > fbank;

	for (int i = 0; i < int(nfilt); ++i){
		vector<double> v;
		fbank.push_back(v);
		for (int j = 0; j < (int)(nfft/2) + 1; ++j){
			fbank[i].push_back(0);
		}
	}

	for(int j = 0 ; j < int(nfilt) ; j++)
		for(int k = 0 ; k < (int)(nfft / 2) + 1 ; k++)
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

	/*const Aquila::Frequencydoubleype sampleFrequency = wav.getSampleFrequency();
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

	cout << "First hurdle." << endl;
	
	vector< vector <double> > framedSignal = frameSignal(doubleWav, (25 * sampleRate / 1000), (10 * sampleRate / 1000));
	vector< vector <double> > filterBanks = getFilterBanks(float(26), float(512), float(wav.getSampleFrequency()), float(0), float(0));

	/*Aquila::Mfcc mfcc(wav.getSamplesCount());
	vector< vector <double> > mfccValues;
	for(int i = 0 ; i < framedSignal.size() ; i++){
		vector<double> tempMfcc = mfcc.calculate(framedSignal[i]);
		mfccValues.push_back(tempMfcc);
	}

	for(int i = 0 ; i < mfccValues.size() ; i++){
		cout << endl;
		for(int j = 0 ; j < mfccValues[i].size() ; j++){
			cout << mfccValues[i][j] << " ";
		}
	}*/

	return 0;
}