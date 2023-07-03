import collections
try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer as glorot_normal
from utils import *


class Model(collections.namedtuple("Model", ["model_name",
                                             'model_dir', 'embedding_upload_hook', 'high_param'])):
    def __new__(cls,
                model_name,
                model_dir,
                embedding_upload_hook=None,
                high_param=None
                ):
        return super(Model, cls).__new__(
            cls,
            model_name,
            model_dir,
            embedding_upload_hook,
            high_param
        )

    def get_model_fn(self):
        def model_fn(features, labels, mode, params):
            pass

        return model_fn

    def get_estimator(self):
        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={}
        )

        # add gauc

        return estimator


# ranking
class DCN(Model):
    def get_model_fn(self):

        def model_fn(features, labels, mode, params):
            is_train = (mode == tf.estimator.ModeKeys.TRAIN)

            my_head = tf.estimator.BinaryClassHead()

            user_feature_embeddings = []
            item_feature_embeddings = []

            user_features = ['uid', 'gender', 'bal']
            item_features = ['item']

            for feature in user_features:
                feature_emb = tf.feature_column.input_layer(features, params['feature_columns'][feature])
                user_feature_embeddings.append(feature_emb)

            for feature in item_features:
                feature_emb = tf.feature_column.input_layer(features, params['feature_columns'][feature])
                item_feature_embeddings.append(feature_emb)

            input = tf.concat(item_feature_embeddings + user_feature_embeddings, axis=1, name='deep')

            # DNN part
            for unit in params['hidden_units']:
                dnn_net = tf.layers.dense(input, units=unit, activation=tf.nn.relu)
                dnn_net = tf.layers.batch_normalization(dnn_net)
                dnn_net = tf.layers.dropout(dnn_net)

            # Cross net part
            cross_out = CrossNet(layer_num=int(self.high_param['cross_num']),
                                 parameterization=self.high_param['cross_parameterization'], l2_reg=1e-5)(input)
            final_out = tf.concat([dnn_net, cross_out], axis=1)
            final_logit = tf.keras.layers.Dense(my_head.logits_dimension, use_bias=False)(final_out)

            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.005)

            def _train_op_fn(loss):
                tf.summary.scalar('loss', loss)
                return optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

            return my_head.create_estimator_spec(
                features=features,
                mode=mode,
                labels=labels,
                logits=final_logit,
                train_op_fn=_train_op_fn
            )

        return model_fn

    def get_estimator(self):
        # 商品id类特征
        def get_categorical_hash_bucket_column(key, hash_bucket_size, dimension, dtype):
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=hash_bucket_size, dtype=dtype
            )
            return tf.feature_column.embedding_column(categorical_column, dimension=dimension)

        # 连续值类特征（差异较为明显）
        def get_bucketized_column(key, boundaries, dimension):
            bucketized_column = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(key), boundaries)
            return tf.feature_column.embedding_column(bucketized_column, dimension=dimension)

        # 层级分类类型特征(num_bucket需要按实际赋值)
        def get_categorical_identity_column(key, num_buckets, dimension, default_value=0):
            identity_column = tf.feature_column.categorical_column_with_identity(key, num_buckets=num_buckets,
                                                                                 default_value=default_value)
            return tf.feature_column.embedding_column(identity_column, dimension=dimension)

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=4, dtype=tf.int64),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=4, dtype=tf.int64),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=4, dimension=1, dtype=tf.int64)
        }

        all_feature_column = {}
        all_feature_column.update(cnt_feature_columns)

        # weight_column = tf.feature_column.numeric_column('weight')

        hidden_layers = [512, 256, 128]

        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={
                'hidden_units': hidden_layers,
                'feature_columns': all_feature_column,
                # 'weight_column': weight_column,
            })

        return estimator


