import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

# Freeze graph and write it to frozen_linear_regression.pb
freeze_graph.freeze_graph(input_graph='linear_regression.pbtxt',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='linear_regression.ckpt',
                          output_node_names='y_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='frozen_linear_regression.pb',
                          clear_devices=True,
                          initializer_nodes='',
                          variable_names_blacklist='')

# Read frozen graph, optimize it and write it to optimized_frozen_linear_regression.pb

# input_graph_def contains all the data from the pb file, converted into a String
input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_linear_regression.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def=input_graph_def,
                                                                     input_node_names=['x'],
                                                                     output_node_names=['y_output'],
                                                                     placeholder_type_enum=tf.float32.as_datatype_enum)

f = tf.gfile.FastGFile(name='optimized_frozen_linear_regression.pb', mode='w')

f.write(file_content=output_graph_def.SerializeToString())