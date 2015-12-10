# include <iostream>
# include <aquila/global.h>
# include <aquila/source/WaveFile.h>

using namespace std;

int main(int argc, char *argv[]) {
	Aquila::WaveFile wavFile("../sounds/Women/Pallavi/pallavi-angry.wav");

	cout << "\nWav file name: " << wavFile.getFilename();
	cout << "\nLength: " << wavFile.getAudioLength() << " ms";
	cout << "\nSample frequency: " << wavFile.getSampleFrequency() << " Hz";
	cout << "\nChannels: " << wavFile.getChannelsNum();
	cout << "\nByte rate: " << wavFile.getBytesPerSec() / 1024 << " kB/s";
	cout << "\nBits per sample: " << wavFile.getBitsPerSample() << "b\n";
	cout << endl;

	return 0;
}