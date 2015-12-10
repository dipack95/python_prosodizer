# include <iostream>
# include <vector>
# include <iterator>
# include <stdexcept>

int pushIntoIt(std::vector<double>::iterator outputIterator, int start_in, int end_in, int num) {
	double delta = (double)(end_in - start_in) / (num - 1);
	for (int i = 0 ; i < num ; i++) {
		if (*outputIterator == 0) {
			*outputIterator = start_in + (delta * i);
			outputIterator++;
		} else {
			throw std::invalid_argument("Length of output container is lesser than number of elements specified.");
			return 0;
		}
	}
	return 1;
}

std::vector<double>::iterator pushInto(std::vector<double>::iterator outputIterator, int start_in, int end_in, int num) {
	double delta = (double)(end_in - start_in) / (num - 1);
	std::vector<double> output;
	output.reserve(num);
	for (int i = 0 ; i < num ; i++) {
		output[i] = start_in + (delta * i);
		std::cout << output[i] << std::endl;
	}
	std::vector<double>::iterator v = output.begin();
	return output.begin();
}

int main(int argc, char const *argv[]) {
	std::vector<double> v;
	std::vector<double>::iterator it;
	it = v.begin();
	std::vector<double>::iterator i = pushInto(it, 7, 15, 10);
	std::cout << "\nLegit array!\n";
	for(; *i ; i++){
		std::cout << *i << std::endl;
	}
	return 0;
}
