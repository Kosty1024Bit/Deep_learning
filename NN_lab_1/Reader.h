#pragma once

#include <string>
#include <fstream>
#include <vector>

using namespace std;

inline int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

inline void ReadData(const char* fileName, float **data){
	ifstream file(fileName, ios::binary);
	if (file.is_open()) {
		int magicNumber = 0;
		int numberImages = 0;
		int numberRows = 0;
		int numberCols = 0;

		file.read((char*)&magicNumber, sizeof(int));
		magicNumber = ReverseInt(magicNumber);

		if (magicNumber != 2051) {
			printf("\nInvalid MNIST image file! \n");
			exit(1);
		}

		file.read((char*)&numberImages, sizeof(int));
		numberImages = ReverseInt(numberImages);

		file.read((char*)&numberRows, sizeof(int));
		numberRows = ReverseInt(numberRows);

		file.read((char*)&numberCols, sizeof(int));
		numberCols = ReverseInt(numberCols);

		for (int i = 0; i < numberImages; i++) {
			data[i][0] = 1;
			int k = 1;
			for (int r = 0; r < numberRows; r++) {
				for (int c = 0; c < numberCols; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(unsigned char));
					data[i][k] = (float)pixel / 255.0;
					k++;
				}
			}
		}
	}
	else {
		printf("\nError opening file! \n");
		exit(1);
	}
}


inline void ReadLabels(const char* filename, float *label)
{
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magicNumber = 0;
		int numberItems = 0;

		file.read((char*)&magicNumber, sizeof(int));
		magicNumber = ReverseInt(magicNumber);

		if (magicNumber != 2049) {
			printf("Invalid MNIST label file! \n");
			exit(1);
		}

		file.read((char*)&numberItems, sizeof(int));
		numberItems = ReverseInt(numberItems);
		for (int i = 0; i < numberItems; i++) {
			unsigned char l = 0;
			file.read((char*)&l, sizeof(unsigned char));
			label[i] = (float)l;
		}
	}
	else {
		printf("\nError opening file! \n");
		exit(1);
	}
}