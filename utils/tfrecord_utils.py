import tensorflow as tf

def image_example(X, Y, num_instances):
    '''
    Creates an image example.
    X: numpy ndarray: the input image data
    Y: numpy ndarray: corresponding label information, can be an ndarray, integer, float, etc

    Returns: tf.train.Example with the following features:
        dim0, dim1, dim2, ..., dimN, X, Y, X_dtype, Y_dtype

    '''
    feature = {}
    feature['X'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[X.tobytes()]))
    feature['Y'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[Y.tobytes()]))
    feature['num_instances'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[num_instances]))

    return tf.train.Example(features=tf.train.Features(feature=feature))



def parse_bag(record, instance_size, num_labels):
    features = {'X': tf.FixedLenFeature([], tf.string),
                'Y': tf.FixedLenFeature([], tf.string),
                'num_instances': tf.FixedLenFeature([], tf.int64)}

    image_features = tf.parse_single_example(record, features=features)

    x = tf.decode_raw(image_features.get('X'), tf.float16)
    x = tf.reshape(x, (image_features.get('num_instances'), *instance_size, 1))
    x = tf.cast(x, tf.float32)

    y = tf.decode_raw(image_features.get('Y'), tf.uint8)
    y = tf.reshape(y, (num_labels, ))
    y = tf.cast(y, tf.float32)

    return x, y

def image_seg_example(X, Y, ):
    '''
    Creates an image example.
    X: numpy ndarray: the input image data
    Y: numpy ndarray: corresponding label information, can be an ndarray, integer, float, etc

    Returns: tf.train.Example with the following features:
        dim0, dim1, dim2, ..., dimN, X, Y, X_dtype, Y_dtype

    '''
    feature = {}
    feature['X'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[X.tobytes()]))
    feature['Y'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[Y.tobytes()]))

    return tf.train.Example(features=tf.train.Features(feature=feature))



def parse_seg_bag(record, instance_size, ):
    features = {'X': tf.FixedLenFeature([], tf.string),
                'Y': tf.FixedLenFeature([], tf.string),
                }

    image_features = tf.parse_single_example(record, features=features)

    x = tf.decode_raw(image_features.get('X'), tf.float64)
    x = tf.reshape(x, (1, *instance_size, 1))
    x = tf.cast(x, tf.float32)

    y = tf.decode_raw(image_features.get('Y'), tf.float32)
    y = tf.reshape(y, (1, *instance_size, 1))
    y = tf.cast(y, tf.float32)

    return x, y
