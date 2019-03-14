import tensorflow as tf

output_node_names = ['output/Softmax']

def main():
    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph("model.meta")

        # Load weights
        saver.restore(sess, './checkpoints/model.ckpt')

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names
            )

        # Save the frozen graph
        with open('frozen_graph.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


if __name__ == '__main__':
    main()
