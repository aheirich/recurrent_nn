param one_hot_encoding_width;
param compressed_input_width;

# layer 0
param rows_0 := one_hot_encoding_width;
param columns_0 := compressed_input_width;
param layer_0_width := one_hot_encoding_width;
var a0{i in 1..layer_0_width};

param layer_0_weights{i in 1..rows_0, j in 1..columns_0};

# layer 1
param layer_1_width := compressed_input_width;
var a1{i in 1..layer_1_width};
var z1{i in 1..layer_1_width};

# range constraints
subject to rangemax1{i in 1..layer_1_width}: z1[i] <= 10;
subject to rangemin1{i in 1..layer_1_width}: z1[i] >= -10;

# compute preactivations
subject to preactivation1{i in 1..layer_1_width}:
z1[i] = sum{j in 1..layer_0_width} (layer_0_weights[j, i] * a0[j])
;

# compute Relu activations
subject to activation1{i in 1..layer_1_width}:
a1[i] = z1[i] * (tanh(100.0*z1[i]) + 1) * 0.5;


# layer 2
param rows_2;
param columns_2;
param layer_2_width := rows_2 - layer_1_width;
var a2{i in 1..layer_2_width};
var z2{i in 1..layer_2_width};

param layer_2_weights{i in 1..rows_2, j in 1..columns_2};

param layer_2_bias{i in 1..rows_2};

# range constraints
subject to rangemax2{i in 1..layer_2_width}: z2[i] <= 10;
subject to rangemin2{i in 1..layer_2_width}: z2[i] >= -10;

# compute preactivations
subject to preactivation2{i in 1..layer_2_width}:
z2[i] = sum{j in 1..layer_1_width} (layer_2_weights[j, i] * a1[j])
+ sum{j in 1+layer_1_width..rows_2} (layer_2_weights[j, i] * a2[j - layer_1_width])
+ layer_2_bias[i]
;

# compute Relu activations
subject to activation2{i in 1..layer_2_width}:
a2[i] = z2[i] * (tanh(100.0*z2[i]) + 1) * 0.5;


# layer 3
param rows_3;
param columns_3;
param layer_3_width := rows_3 - layer_2_width;
var a3{i in 1..layer_3_width};
var z3{i in 1..layer_3_width};

param layer_3_weights{i in 1..rows_3, j in 1..columns_3};

param layer_3_bias{i in 1..rows_3};

# range constraints
subject to rangemax3{i in 1..layer_3_width}: z3[i] <= 10;
subject to rangemin3{i in 1..layer_3_width}: z3[i] >= -10;

# compute preactivations
subject to preactivation3{i in 1..layer_3_width}:
z3[i] = sum{j in 1..layer_2_width} (layer_3_weights[j, i] * a2[j])
+ sum{j in 1+layer_2_width..rows_3} (layer_3_weights[j, i] * a3[j - layer_2_width])
+ layer_3_bias[i]
;

# compute Relu activations
subject to activation3{i in 1..layer_3_width}:
a3[i] = z3[i] * (tanh(100.0*z3[i]) + 1) * 0.5;


# layer 4
param rows_4;
param columns_4;
param layer_4_width;
var a4{i in 1..layer_4_width};
var z4{i in 1..layer_4_width};

param layer_4_weights{i in 1..rows_4, j in 1..columns_4};

param layer_4_bias{i in 1..rows_4};

# range constraints
subject to rangemax4{i in 1..layer_4_width}: z4[i] <= 10;
subject to rangemin4{i in 1..layer_4_width}: z4[i] >= -10;

# compute preactivations
subject to preactivation4{i in 1..layer_4_width}:
z4[i] = sum{j in 1..layer_3_width} (layer_3_weights[i, j] * a3[j])
+ layer_4_bias[i]
;

# compute Relu activations
subject to activation4{i in 1..layer_4_width}:
a4[i] = z4[i] * (tanh(100.0*z4[i]) + 1) * 0.5;

