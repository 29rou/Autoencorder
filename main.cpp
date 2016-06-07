#include "autoencorder.h"
#define mid 301
#define im 60

void autoencorder(const std::array<uint8_t, im * im + 1> img) {
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	std::uniform_real_distribution<> randc(-1,1);
	double w_enc[im * im + 1][mid] = {};
	double w_dcd[mid][im * im] = {};
	std::array<double, mid> midimg; midimg.back() = 1;
	std::array<double, im * im> output= {};
	double tr = 0.1;
	for (int i = 0; i < mid - 1; i++) {
		for (int j = 0; j < (im * im + 1); j++) {
			w_enc[j][i] = randc(mt);
		}
	}
	for (int i = 0; i < im * im; i++) {
		for (int j = 0; j < mid; j++) {
			w_dcd[j][i] = randc(mt);
		}
	}
	for (int s = 1; s <= 100;s++) {
		//tr -= 1.0 / 20.0;
		for (int i = 0; i < mid - 1; i++) {
			double sum = 0;
			for (int j = 0; j < (im * im + 1); j++) {
				sum += w_enc[j][i] * img.at(j);
			}
			//std::cout << sum << std::endl;
			midimg.at(i) = 1.0 / (1.0 + expl(-sum));
		}
		for (int i = 0; i < im * im; i++) {
			double sum = 0;
			for (int j = 0; j < mid; j++) {
				sum += w_dcd[j][i] * midimg.at(j);
			}
			//std::cout << sum << std::endl;
			output.at(i) = 1.0 / (1.0 + expl(-sum));
		}
		for (int i = 0; i < mid; i++) {
			for (int j = 0; j < im * im; j++) {
				w_dcd[i][j] += tr * (img.at(j) - output.at(j))*(output.at(j))*(1 - output.at(j))*(midimg.at(i));
			}
		}
		for (int i = 0; i < im * im + 1; i++) {
			for (int j = 0; j < mid; j++) {
				double sum = 0;
				for (int k = 0; k < im*im; k++) {
					sum += (img.at(k) - output.at(k))*(1-output.at(k))*(output.at(k))*w_dcd[j][k] * (midimg.at(j))*(1 - midimg.at(j))*(img.at(i));
				}
				w_enc[i][j] += tr*sum;
			}
		}
	}
	for (int i = 0; i < mid-1; i++) {
		double sum = 0;
		for (int j = 0; j < (im * im + 1); j++) {
			sum += w_enc[j][i] * img.at(j);
		}
		//std::cout << sum << std::endl;
		midimg.at(i) = 1.0 / (1.0 + expl(-sum));
	}
	for (int i = 0; i < im * im; i++) {
		double sum = 0;
		for (int j = 0; j < mid; j++) {
			sum += w_dcd[j][i] * midimg.at(j);
		}
		std::cout << sum << std::endl;
		output.at(i) = 1.0 / (1.0 + expl(-sum));
	}

	std::cout << "Finish" << std::endl;
	using namespace cv;
	Mat src_img = Mat::zeros(Size(im, im), CV_8UC1);
	auto binary = [](double x)->uint8_t {
		if(roundl(x)>0)return 255;
		return 0;
	};
	for (int i = 0; i < im; i++) //i is height y 
	{
		for (int j = 0; j < im; j++)//j is width x
		{
			std::cout << img.at(i * im + j)<<" "<< (uint8_t)roundl(output.at(i * im + j)) << "\n";
			src_img.at<uchar>(i, j) = binary(output.at(i * im + j));
		}
	}
	resize(src_img, src_img, Size(300, 300));
	std::cout << "redy";
	cv::imshow("test2", src_img);
	cv::waitKey(0);
}

int main() {
	using namespace cv;
	std::array<uint8_t, im * im + 1> img = {}; img.back() = 1;
	Mat src_img = imread(".\\test.png");
	cvtColor(src_img, src_img, CV_RGB2GRAY);
	threshold(src_img, src_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
	cv::imshow("test1", src_img);
	cv::waitKey(1);
	resize(src_img, src_img, Size(im, im));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < im; i++) //i is height y 
	{
		for (int j = 0; j < im; j++)//j is width x
		{
			img[i * im + j] = (uint8_t)src_img.at<uchar>(i, j);
		}
	}
	autoencorder(img);
}