from __future__ import print_function
import json
from tqdm import tqdm
from w2v_model import *
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from config import model_path


def init_keras_config():
    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    session = K.tf.Session(config=config)
    K.set_session(session)


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1, keepdims=True)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)(seq)

    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h

    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


class Attention(Layer):
    """多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))


def get_k_inter(x, n=6):
    seq, k1, k2 = x
    k_inter = [K.round(k1 * a + k2 * (1 - a)) for a in np.arange(n) / (n - 1.)]
    k_inter = [seq_gather([seq, k]) for k in k_inter]
    k_inter = [K.expand_dims(k, 1) for k in k_inter]
    k_inter = K.concatenate(k_inter, 1)
    return k_inter


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """

    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))

    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


class Evaluate(Callback):
    def __init__(self, train_model, EMAer, dev_data, spoer, subject_model, object_model):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
        self.train_model = train_model
        self.EMAer = EMAer
        self.dev_data = dev_data
        self.spoer = spoer
        self.subject_model = subject_model
        self.object_model = object_model

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        self.EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            self.train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        self.EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
                self.stage == 0 and epoch > 10 and
                (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            self.train_model.load_weights('best_model.weights')
            self.EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    def evaluate(self):
        from data_loader import extract_items
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(self.dev_data)):
            R = set(extract_items(d['text'], spoer=self.spoer, subject_model=self.subject_model,
                                  object_model=self.object_model))
            T = set(d['spo_list'])
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'spo_list_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            F.write(s + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C


def model():
    from data_loader import num_classes, char_size, maxlen, char2id

    t1_in = Input(shape=(None,))
    t2_in = Input(shape=(None, word_size))
    s1_in = Input(shape=(None,))
    s2_in = Input(shape=(None,))
    k1_in = Input(shape=(1,))
    k2_in = Input(shape=(1,))
    o1_in = Input(shape=(None, num_classes))
    o2_in = Input(shape=(None, num_classes))
    pres_in = Input(shape=(None, 2))
    preo_in = Input(shape=(None, num_classes * 2))

    t1, t2, s1, s2, k1, k2, o1, o2, pres, preo = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in, pres_in, preo_in
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)

    # -------------------------------------------------
    # -------------------------------------------------
    pid = Lambda(position_id)(t1)
    position_embedding = Embedding(maxlen, char_size, embeddings_initializer='zeros')
    pv = position_embedding(pid)

    t1 = Embedding(len(char2id) + 2, char_size)(t1)  # 0: padding, 1: unk
    t2 = Dense(char_size, use_bias=False)(t2)  # 词向量也转为同样维度
    t = Add()([t1, t2, pv])  # 字向量、词向量、位置向量相加
    t = Dropout(0.25)(t)
    t = Lambda(lambda x: x[0] * x[1])([t, mask])
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 1)
    t_dim = K.int_shape(t)[-1]

    pn1 = Dense(char_size, activation='relu')(t)
    pn1 = Dense(1, activation='sigmoid')(pn1)
    pn2 = Dense(char_size, activation='relu')(t)
    pn2 = Dense(1, activation='sigmoid')(pn2)

    h = Attention(8, 16)([t, t, t, mask])
    h = Concatenate()([t, h, pres])
    h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
    ps1 = Dense(1, activation='sigmoid')(h)
    ps2 = Dense(1, activation='sigmoid')(h)
    ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])
    ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])

    subject_model = Model([t1_in, t2_in, pres_in], [ps1, ps2])  # 预测subject的模型

    t_max = Lambda(seq_maxpool)([t, mask])
    pc = Dense(char_size, activation='relu')(t_max)
    pc = Dense(num_classes, activation='sigmoid')(pc)
    # -------------------------------------------------
    # -------------------------------------------------
    k = Lambda(get_k_inter, output_shape=(6, t_dim))([t, k1, k2])
    k = Bidirectional(CuDNNGRU(t_dim))(k)
    k1v = position_embedding(Lambda(position_id)([t, k1]))
    k2v = position_embedding(Lambda(position_id)([t, k2]))
    kv = Concatenate()([k1v, k2v])
    k = Lambda(lambda x: K.expand_dims(x[0], 1) + x[1])([k, kv])

    h = Attention(8, 16)([t, t, t, mask])
    h = Concatenate()([t, h, k, pres, preo])
    h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
    po = Dense(1, activation='sigmoid')(h)
    po1 = Dense(num_classes, activation='sigmoid')(h)
    po2 = Dense(num_classes, activation='sigmoid')(h)
    po1 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po1, pc, pn1])
    po2 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po2, pc, pn2])

    object_model = Model([t1_in, t2_in, k1_in, k2_in, pres_in, preo_in], [po1, po2])  # 输入text和subject，预测object及其关系

    train_model = Model([t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in, pres_in, preo_in],
                        [ps1, ps2, po1, po2])

    s1 = K.expand_dims(s1, 2)
    s2 = K.expand_dims(s2, 2)

    s1_loss = K.binary_crossentropy(s1, ps1)
    s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
    s2_loss = K.binary_crossentropy(s2, ps2)
    s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

    o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
    o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
    o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
    o2_loss = K.sum(o2_loss * mask) / K.sum(mask)

    loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(1e-3))
    train_model.summary()
    return train_model, subject_model, object_model


def train():
    from data_trans import process
    process()
    from data_loader import train_data, SpoSearcher, dev_data, DataGenerator

    init_keras_config()
    train_model, subject_model, object_model = model()

    EMAer = ExponentialMovingAverage(train_model)
    EMAer.inject()

    spoer = SpoSearcher(train_data)
    train_D = DataGenerator(train_data)

    evaluator = Evaluate(train_model, EMAer=EMAer, dev_data=dev_data, spoer=spoer, subject_model=subject_model,
                         object_model=object_model)

    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=120,
                              callbacks=[evaluator]
                              )


def test(test_data):
    """输出测试结果
    """
    from data_loader import train_data, SpoSearcher, extract_items

    train_model, subject_model, object_model = model()
    EMAer = ExponentialMovingAverage(train_model)
    EMAer.inject()

    spoer = SpoSearcher(train_data)

    orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
    F = open('test_pred.json', 'w')
    for d in tqdm(iter(test_data)):
        R = set(extract_items(d['text'], spoer=spoer, subject_model=subject_model, object_model=object_model))
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, spo + ('', ''))) for spo in R
            ]
        }, ensure_ascii=False)
        F.write(s.encode('utf-8') + '\n')
    F.close()


def predict(sentence_list):
    from data_loader import train_data, SpoSearcher, extract_items

    train_model, subject_model, object_model = model()
    train_model.load_weights(model_path)

    spoer = SpoSearcher(train_data)

    EMAer = ExponentialMovingAverage(train_model)
    EMAer.inject()

    orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
    for sent in tqdm(iter(sentence_list)):
        R = set(extract_items(sent, spoer=spoer, subject_model=subject_model, object_model=object_model))
        spo_list = [
            dict(zip(orders, spo + ('', ''))) for spo in R
        ]
        yield sent, spo_list
