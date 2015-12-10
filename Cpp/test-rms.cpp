# include <cmath>
# include <iostream>
# include <vector>
# include <functional>
# include <numeric>
# include <algorithm>
# include <iterator>

using namespace std;

double rootMeanSquare(std::vector<double> energy){
	double squaredAverage = 0;
	for(int i = 0 ; i < energy.size() ; i++){
		squaredAverage += energy[i] * energy[i];
	}
	return pow((squaredAverage/energy.size()), 0.5);
}


double rootMeanSquare(std::vector<double>::iterator energy, int len){
	double squaredAverage = 0;
	int size = 0;
	for(; size < len ; energy++, size++){
		squaredAverage += (*energy) * (*energy);
	}
	return pow((squaredAverage/size), 0.5);
}

int main(){
	std::vector<double> testArray (1000);
	int n(0);
	generate(testArray.begin(), testArray.end(), [&n] { return n++ ;});
	auto result = rootMeanSquare(testArray.begin(), std::distance(testArray.begin(), testArray.end()));
	auto result2 = rootMeanSquare(testArray);
	cout << result << " " << result2;
	return 0; 
}