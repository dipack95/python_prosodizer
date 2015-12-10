# include <boost/numeric/ublas/matrix.hpp>
# include <boost/numeric/ublas/io.hpp>
# include <boost/numeric/ublas/matrix_proxy.hpp>
# include <boost/range/irange.hpp>
# include <aquila/global.h>
# include <aquila/source/generator/SineGenerator.h>
# include <aquila/transform/FftFactory.h>
# include <vector>
# include <stdexcept>
# include <iterator>
# include <algorithm>	

template<typename TMatrix>
using MatrixRow =  boost::numeric::ublas::matrix_row<TMatrix>;

template <typename TValue>
using matrix_t =  boost::numeric::ublas::matrix<TValue>;

template <typename InputIterator, typename OutputContainer, size_t N>
OutputContainer tile(const InputIterator& begin, const InputIterator& end, OutputContainer& retVal, const int(&dimensions)[N]) {

	auto size = std::distance(begin, end);
	for (int i = 0; i < dimensions[0]; i++ ) {
		MatrixRow<OutputContainer> row (retVal, i);
		for (int j = 0; j < dimensions[1]; j++) {
			std::copy(begin, end, row.begin() + (j * size));
		}
	}

	return retVal;
}

template<typename TValue, typename InputContainer>
matrix_t<TValue> frameSignal(InputContainer signal, TValue fLength, TValue fStep) {

	int frameLength = static_cast<int>(round(fLength));
	int frameStep = static_cast<int>(round(fStep));

	auto signalLength = std::distance(signal.begin(), signal.end());

	int numOfFrames, dimensions[2];
	std::vector<double> indicesValues;

	matrix_t<TValue> indices, frames, temp1, temp2;

	if (signalLength <= frameLength) {
		numOfFrames = 1;
	}
	else {
		numOfFrames = 1 + int(ceil((1.0 * signalLength - frameLength) / frameStep));
	}

	/* Resizing the signal to standard signal size. Eg: 500 signal length will be resized to 512. */
	int padLength = static_cast<int>(((numOfFrames - 1) * frameStep + frameLength));
	signal.resize(padLength);

	/* Creating two temporary matrices. */
	auto matrixOneRange = boost::irange(0, frameLength, 1);
	auto matrixTwoRange = boost::irange(0, (numOfFrames * frameStep), frameStep);

	auto matrixOneRangeSize = std::distance(matrixOneRange.begin(), matrixOneRange.end());
	dimensions[0] = numOfFrames, dimensions[1] = 1;
	temp1.resize(dimensions[0], matrixOneRangeSize * dimensions[1]);

	tile(matrixOneRange.begin(), matrixOneRange.end(), temp1, dimensions);

	auto matrixTwoRangeSize = std::distance(matrixTwoRange.begin(), matrixTwoRange.end());
	dimensions[0] = frameLength, dimensions[1] = 1;
	temp2.resize(dimensions[0], matrixTwoRangeSize * dimensions[1]);

	tile(matrixTwoRange.begin(), matrixTwoRange.end(), temp2, dimensions);

	/* Adding the two matrices. */
	indices.resize(numOfFrames, frameLength);
	indices = temp1 + boost::numeric::ublas::trans(temp2);

	for (auto it1 = indices.begin1() ; it1 != indices.end1() ; ++it1) {
		for (auto it2 = it1.begin() ; it2 != it1.end() ; ++it2) {
			indicesValues.push_back(*it2);
		}
	}

	frames.resize(numOfFrames, frameLength);
	auto it3 = indicesValues.begin();
	for (auto it1 = frames.begin1() ; it1 != frames.end1() ; ++it1) {
		for (auto it2 = it1.begin() ; it3 != indicesValues.end() && it2 != it1.end() ; ++it2, ++it3) {
			*it2 = signal[*it3];
		}
	}

	matrix_t<TValue> window (numOfFrames, frameLength, 1);

	/* Multiplies each element of frames with its corresponding element in window.
	 * The dimensions of the frames and window matrices MUST BE THE SAME.
	 */
	for (auto i = 0, k = 0 ; i < frames.size1() && k < window.size1() ; i++, k++) {
		for (auto j = 0, l = 0 ; j < frames.size2() && l < window.size2() ; j++, l++) {
			frames(i, j) *= window(k, l);
		}
	}

	return frames;
}

template<typename InputContainer, typename InputType>
InputContainer computePower(InputContainer frames, InputType nfft) {
	const std::size_t SIZE = nfft;

	/* Creating the FFT of size nfft to use with the values
	 * from the input boost matrix.
	 */
	auto generatedFft = Aquila::FftFactory::getFft(SIZE);
	InputContainer powerComputedMatrix (frames.size1(), nfft);

	/* Taking each row from the input boost matrix,
	 * and generating the FFT from each row.
	 * The output values are pushed into another boost matrix
	 * which is returned as the output of this function.
	 */
	for (unsigned int i = 0 ; i < frames.size1() ; i++) {
		boost::numeric::ublas::matrix_row< InputContainer > matrixRow (frames, i);

		std::vector<double> matrixRowVector (frames.size2());
		std::copy(matrixRow.begin(), matrixRow.end(), matrixRowVector.begin());

		Aquila::SpectrumType generatedSpectrum = generatedFft -> fft(&matrixRowVector[0]);

		std::copy(powerComputedMatrix.begin1(), powerComputedMatrix.end1(), generatedSpectrum.begin());
	}

	return powerComputedMatrix;
}

int main() {
	double arg1 = 25.0 * (44100.0 / 1000.0), arg2 = 10.0 * (44100.0 / 1000.0);
	std::vector<double> testSineSignal (20000);
	for (unsigned int i = 0 ; i < testSineSignal.size() ; i++) {
		testSineSignal[i] = std::sin(1000 * (2 * 3.14) * i / 44100);
	}
	boost::numeric::ublas::matrix<double> framedSineSignal = frameSignal(testSineSignal, floor(arg1), floor(arg2));
	boost::numeric::ublas::matrix<double> powerComputedMatrix = computePower(framedSineSignal, double(512));

	std::cout << "Power Computed Matrix: " << powerComputedMatrix;
	return 0;
}