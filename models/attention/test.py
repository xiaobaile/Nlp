import tensorflow as tf
import numpy as np
import os
from six.moves import xrange
from models.attention import datautil
from models.attention import seq2seq_model

_buckets = []
convo_hist_limit = 1
max_source_length = 0
max_target_length = 0

flags = tf.flags
FLAGS = flags.FLAGS

tf.reset_default_graph()

_buckets = [(3, 3), (5, 5), (10, 10)]  # , (20, 20)]# ["40,10","50,15"]
max_train_data_size = 0  # (0: no limit)

data_dir = "datacn/"

dropout = 1.0
grad_clip = 5.0
batch_size = 60
hidden_size = 14
num_layers = 2
learning_rate = 0.5
lr_decay_factor = 0.99

checkpoint_dir = "data/checkpoints/"

hidden_size = 100
checkpoint_dir = "fanyichina/checkpoints/"
data_dir = "fanyichina/"
_buckets = [(20, 20), (40, 40), (50, 50), (60, 60)]


def get_fan_yi_info():
    vocab_en, rev_vocab_en = datautil.initialize_vocabulary(os.path.join(datautil.data_dir, datautil.vocabulary_file_en))
    vocab_size_en = len(vocab_en)
    print("vocab_size", vocab_size_en)

    vocab_ch, rev_vocab_ch = datautil.initialize_vocabulary(os.path.join(datautil.data_dir, datautil.vocabulary_file_ch))
    vocab_size_ch = len(vocab_ch)
    print("vocab_size_ch", vocab_size_ch)

    return vocab_size_en, vocab_size_ch, vocab_en, rev_vocab_ch


def main():
    vocab_size_en, vocab_size_ch, vocab_en, rev_vocab_ch = get_fan_yi_info()

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print("checkpoint_dir is {0}".format(checkpoint_dir))

    with tf.Session() as sess:
        model = create_model(sess, True, vocab_size_en, vocab_size_ch)
        print(_buckets)
        model.batch_size = 1

        conversation_history = []
        while True:
            prompt = "请输入: "
            sentence = input(prompt)
            conversation_history.append(sentence.strip())
            conversation_history = conversation_history[-convo_hist_limit:]

            token_ids = list(reversed(
                datautil.sentence_to_ids(" ".join(conversation_history), vocab_en, normalize_digits=True, is_ch=False)))
            print(token_ids)
            bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            # TODO implement beam search
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if datautil.EOS_ID in outputs:
                outputs = outputs[:outputs.index(datautil.EOS_ID)]
                convo_output = " ".join(datautil.ids2texts(outputs, rev_vocab_ch))
                conversation_history.append(convo_output)
                print(convo_output)
            else:
                print("can not translation！")


def create_model(session, forward_only, from_vocab_size, to_vocab_size):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        from_vocab_size,  # from
        to_vocab_size,  # to
        _buckets,
        hidden_size,
        num_layers,
        dropout,
        grad_clip,
        batch_size,
        learning_rate,
        lr_decay_factor,
        forward_only=forward_only,
        dtype=tf.float32)

    print("model is ok")

    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt is not None:
        model.saver.restore(session, ckpt)
        print("Reading model parameters from {0}".format(ckpt))
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model


if __name__ == "__main__":
    main()
