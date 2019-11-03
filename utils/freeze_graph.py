import os, argparse

import tensorflow as tf
from tensorflow.contrib.rnn import *

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names, precision_mode='FP32'):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the useful nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the frozen graph." % len(output_graph_def.node))

        trt_graph = tf.contrib.tensorrt.create_inference_graph(
            input_graph_def=output_graph_def,
            outputs=output_node_names.split(","),
            max_batch_size=1,
            max_workspace_size_bytes=2500000000,
            precision_mode=precision_mode)

        with tf.gfile.FastGFile(absolute_model_dir + "/tensor_rt.pb", 'wb') as f:
            f.write(trt_graph.SerializeToString())

        trt_engine_nodes = len(
            [1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
        print("{} trt_engine_nodes in TensorRT graph".format(trt_engine_nodes))
        all_nodes = len([1 for n in trt_graph.node])
        print("{} nodes in TensorRT graph".format(all_nodes))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./weights", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="softmax_output",
                        help="The name of the output nodes, comma separated.")
    parser.add_argument("--precision_mode", type=str, default="FP32", required=False, help="Percision model for TensorRT optimization")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names, args.precision_mode)
