#include <iostream>
using namespace std;

#include "Layer.h"
#include "Reader.h"

void main(){
	const uint num_of_epochs = 10;			//10

	const uint num_of_out_neuron = 10;		//10
	const uint num_of_hiden_neuron = 100;		//100
	const uint num_of_input_neuron = 28*28;		//28 * 28

	const uint size_train_data = 60000;			//60000
	const uint size_test_data = 10000;		//10000

	const float learning_rate = 0.01;
	
	float** train_data = new float*[size_train_data];
	for (int i = 0; i < size_train_data; i++) train_data[i] = new float[num_of_input_neuron];
	float* train_labels = new float[size_train_data];

	ReadData("train-images.idx3-ubyte", train_data);
	ReadLabels("train-labels.idx1-ubyte", train_labels);
	
	float** test_data = new float*[size_test_data];
	for (int i = 0; i < size_test_data; i++) test_data[i] = new float[num_of_input_neuron];
	float* test_labels = new float[size_test_data];
	
	ReadData("t10k-images.idx3-ubyte", test_data);
	ReadLabels("t10k-labels.idx1-ubyte", test_labels);
	
	layer output(num_of_out_neuron, learning_rate);
	layer hiden(num_of_hiden_neuron, output.num_of_neurom, learning_rate);
	layer input(num_of_input_neuron, hiden.num_of_neurom, learning_rate);

	hiden.init_rand();
	input.init_rand();

	cout << "training..." << endl;
	for (uint epoch = 0; epoch < num_of_epochs; epoch++) {
		cout << "epoch number " << epoch + 1<< endl;
		for (uint num_data = 0; num_data < size_train_data; num_data++) {

			input.init_input_layer(train_data[num_data], num_of_input_neuron);
			hiden.clear_neuron();
			output.clear_neuron();

			hiden.compute(input);

			hiden.tanh_af();
		
			output.compute(hiden);
			
			output.softmax_af();	

			if (num_data == 9999) getchar();
			
			output.compute_out_errors(hiden, 1);

			hiden.compute_errors(output);

			input.updata_weight(hiden);

			hiden.updata_weight(output);
		}
	
		//mix data
		for (int i = 0; i < size_train_data; i++){
			int tmp1 = rand() % size_train_data;
			int tmp2 = rand() % size_train_data;

			swap(train_data[tmp1], train_data[tmp2]);
			swap(train_labels[tmp1], train_labels[tmp2]);
		}
	}
	
	float count_true_answers = 0.0f;
	float accuracy = 0.0;
	double cross_entrophy = 0.0;
	
	cout << "testing..." << endl;
	for (uint num_data = 0; num_data < size_test_data; num_data++) {
		
		input.init_input_layer(test_data[num_data], num_of_input_neuron);
		hiden.clear_neuron();
		output.clear_neuron();
		
		hiden.compute(input);
		output.compute(hiden);

		cross_entrophy += output.cross_entropy(test_labels[num_data]);

		if (output.neurom_out[(int)test_labels] == 1) count_true_answers++;
	}

	accuracy = count_true_answers / size_test_data;
	
	cout << "Cross-entrophy: " << cross_entrophy << endl;
	cout << "Accuracy: " << accuracy << endl;

	getchar();
	getchar();
	getchar();
}

