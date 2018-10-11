#pragma once

#pragma once

#include <iostream>
#include <iomanip>

typedef unsigned int uint;

class layer {
public:
	float *neuron;
	float *neuron_out;
	float *errors;

	float *weight;

	uint num_of_weight;
	uint num_of_neuron;

	float learning_rate;

	layer(uint _size_this_layer, uint _size_next_layer, const float _learning_rate) {
		this->num_of_neuron = _size_this_layer;
		this->num_of_weight = _size_this_layer * _size_next_layer;

		this->neuron = new float[this->num_of_neuron];
		this->neuron_out = new float[this->num_of_neuron];
		this->errors = new float[this->num_of_neuron];
		for (int i = 0; i < num_of_neuron; i++) {
			neuron[i] = 0;
			neuron_out[i] = 0;
			errors[i] = 0;
		}

		this->weight = new float[this->num_of_weight];

		this->learning_rate = _learning_rate;
	}

	layer(uint _size_this_layer, const float _learning_rate) {
		this->num_of_neuron = _size_this_layer;
		this->num_of_weight = _size_this_layer;

		this->neuron = new float[this->num_of_neuron];
		this->neuron_out = new float[this->num_of_neuron];
		this->errors = new float[this->num_of_neuron];
		for (int i = 0; i < num_of_neuron; i++) {
			neuron[i] = 0;
			neuron_out[i] = 0;
			errors[i] = 0;
		}

		this->weight = new float[this->num_of_weight];
		this->weight = this->neuron;

		this->learning_rate = _learning_rate;
	}

	void init_rand() {
		for (int i = 0; i < num_of_weight; i++) weight[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
	}

	void init_input_layer(float *_data, uint _data_size) {
		if (_data_size != num_of_neuron) std::cout << "Error: mismatch of the number of input signals" << std::endl;
		else {
			for (int i = 0; i < num_of_neuron; i++) neuron[i] = _data[i];
		}
		neuron_out = neuron;
	}

	void compute(layer _previous_layer) {
		for (int m = 0; m < num_of_neuron; m++) {
#pragma omp parallel for//
			for (int n = 0; n < _previous_layer.num_of_neuron; n++) {
				neuron[m] += _previous_layer.neuron_out[n] * _previous_layer.weight[_previous_layer.num_of_neuron * m + n];
			}
		}
	}

	void compute_out_errors(layer _previous_layer, float _label) {
		float *true_answer = new float[num_of_neuron];
		for (int i = 0; i < num_of_neuron; i++) true_answer[i] = 0.0f;
		true_answer[(int)_label] = 1.0f;

		float sum = 0.0f;
		float d = 0.0f;

		for (int i = 0; i < num_of_neuron; i++) errors[i] = true_answer[i] - neuron_out[i];
	}

	void compute_errors(layer _next_layer) {
		for (uint i = 0; i < num_of_neuron; i++) {
			for (int j = 0; j < _next_layer.num_of_neuron; j++) {
				errors[i] += weight[num_of_neuron * j + i] * _next_layer.errors[j];
			}
		}
	}

	void updata_weight(layer _next_layer) {
		float func = 0;
		for (uint i = 0; i < _next_layer.num_of_neuron; i++) {
			func = _next_layer.errors[i] * (2 / (cosh(2 * _next_layer.neuron[i]) + 1));
			for (int j = 0; j < num_of_neuron; j++) {
				weight[_next_layer.num_of_neuron * i + j] += func * neuron_out[j] * learning_rate;
			}
		}
	}

	void clear_neuron() {
		for (int i = 0; i < num_of_neuron; i++) {
			neuron[i] = 0;
			neuron_out[i] = 0;
			errors[i] = 0;
		}
	}

	void softmax_af() {
		float sum_exp = 0;
		float *exp_neurom = new float[num_of_neuron];

		for (int i = 0; i < num_of_neuron; i++) {
			exp_neurom[i] = exp(neuron[i]);
			sum_exp += exp_neurom[i];
		}
		for (int i = 0; i < num_of_neuron; i++) neuron_out[i] = exp_neurom[i] / sum_exp;

		delete[] exp_neurom;
	}

	void tanh_af() {
		for (int i = 0; i < num_of_neuron; i++) neuron_out[i] = tanh(neuron[i]);
	}

	float cross_entropy(float _label) {

		float *true_answer = new float[num_of_neuron];
		for (uint i = 0; i < num_of_neuron; i++) true_answer[i] = 0.0f;
		true_answer[(int)_label] = 1.0f;

		float sum = 0;

		for (uint i = 0; i < num_of_neuron; i++) sum += log(neuron[i]) * true_answer[i];


		return sum;
	}

};