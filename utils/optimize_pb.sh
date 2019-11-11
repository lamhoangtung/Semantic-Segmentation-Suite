# Install bazel, clone tensorflow repo, run `./configure` then

# For MobileUnet
# --transforms='
# add_default_attributes
# strip_unused_nodes(type=float, shape="1,299,299,3")
# remove_nodes(op=Identity, op=CheckNumerics)
# fold_constants(ignore_errors=true)
# fold_batch_norms
# fold_old_batch_norms
# quantize_weights
# quantize_nodes
# strip_unused_nodes
# sort_by_execution_order'

# For DenseASPP
# --transforms='
# add_default_attributes
# strip_unused_nodes(type=float, shape="1,299,299,3")
# remove_nodes(op=Identity, op=CheckNumerics)
# fold_batch_norms
# strip_unused_nodes
# sort_by_execution_order
# obfuscate_names'

# bazel build tensorflow/tools/graph_transforms:transform_graph
~/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=frozen_model.pb \
--out_graph=optimized.pb \
--inputs='Placeholder' \
--outputs='softmax_output' \
--transforms='
add_default_attributes
strip_unused_nodes(type=float, shape="1,299,299,3")
remove_nodes(op=Identity, op=CheckNumerics)
merge_duplicate_nodes
remove_control_dependencies
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
strip_unused_nodes
sort_by_execution_order
obfuscate_names'