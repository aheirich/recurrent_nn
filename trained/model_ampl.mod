
# network size

param embedding_layer_width;
param input_width;
param recurrent_layer1_size;
param recurrent_layer1_width;
param recurrent_layer2_size;
param recurrent_layer2_width;
param output_width;
param projection_layer_width;

# weights and biases

param embed{i in 1..embedding_layer_width, j in 1..input_width};

param Winphid1{i in 1..recurrent_layer1_width, j in 1..recurrent_layer1_size};

param Bhid1{i in 1..recurrent_layer1_size};

param Whid12{i in 1..recurrent_layer2_width, j in 1..recurrent_layer2_size};

param Bhid2{i in 1..recurrent_layer2_size};

param proj{i in 1..output_width, j in 1..projection_layer_width};

param Bproj{i in 1..projection_layer_width};


# preactivation

var z_embed{i in 1..embedding_layer_width};
var z_layer1{i in 1..recurrent_layer1_width};
var z_layer2{i in 1..recurrent_layer2_width};
var z_projection{i in 1..projection_layer_width};

# activation

var a_embed{i in 1..embedding_layer_width};
var a_layer1{i in 1..recurrent_layer1_width};
var a_layer2{i in 1..recurrent_layer2_width};
var a_projection{i in 1..projection_layer_width};

