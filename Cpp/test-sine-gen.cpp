# include <iostream>
# include <cmath>
# include <vector>

int main() {
	std::vector<double> testSineSignal (20000);

	for (int i = 0 ; i < testSineSignal.size() ; i++) {
		testSineSignal[i] = std::sin(1000 * (2 * 3.14) * i / 44100);
		std::cout << testSineSignal[i] << " ";
	}
	return 0;
}