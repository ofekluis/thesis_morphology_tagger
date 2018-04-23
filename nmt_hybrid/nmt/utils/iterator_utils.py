# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# WORD_MAX_LEN is used for padding for character tensors later on.
WORD_MAX_LEN=50
def pad_tensor(t,n,sym):
    """
    args:
        t- a tensor to pad
        n- a size up to which to pad the tensor
        sym- padding symbol
    return:
        a sym padded n sized tensor
    """
    # NOTE: This specifically for rank 1 i.e. vector tensors, can be easily
    # changed.
    s = tf.shape(t)[0]
    paddings = [[0,0],[0,n-s]]
    return tf.pad([t], paddings, 'CONSTANT', constant_values=sym)[0]

# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                            ("initializer", "source", "source_char", "target_input",
                                "target_output", "source_sequence_length",
                                "target_sequence_length", "word_len"))):
    pass


def get_infer_iterator(src_dataset,
        src_vocab_table,
        batch_size,
        eos,
        src_max_len=None,
        src_char_vocab_table=None):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_char_eos_id = tf.cast(src_char_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    char = not(src_char_vocab_table is None)
    if src_max_len:
        src_dataset = src_dataset.map(lambda src: src[:src_max_len])
    if char:
        # Convert the word strings to ids
        # the tf.map_fn might seem a bit more involved but it basically just a
        # nested loop converting every char of every word in every sentence into
        # its id.
        src_dataset = src_dataset.map(
            lambda src: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                tf.map_fn(lambda word: pad_tensor(tf.cast(src_char_vocab_table.lookup(
                tf.string_split([word], delimiter="").values), tf.int32),
                WORD_MAX_LEN, src_char_eos_id), src , tf.int32, infer_shape=False),
                         tf.map_fn(lambda word: tf.size(tf.string_split([word], delimiter="").values),src, tf.int32)
                         ))
        # Add in the word counts.
        src_dataset = src_dataset.map(lambda src, src_char, src_char_len: (src,
                                            src_char , tf.size(src),
                            src_char_len))
    else:
        # Convert the word strings to ids
        src_dataset = src_dataset.map(
            lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
        # Add in the word counts.
        src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x, char):
        if char:
            return x.padded_batch(
                    batch_size,
                    # The entry is the source line rows;
                    # this has unknown-length vectors.    The last entry is
                    # the source row size; this is a scalar.
                    padded_shapes=(
                            tf.TensorShape([None]),    # src
                            tf.TensorShape([None, None]),    # src_char
                            tf.TensorShape([]),    # src_len
                            tf.TensorShape([None])),    # src_char_len

                    # Pad the source sequences with eos tokens.
                    # (Though notice we don't generally need to do this since
                    # later on we will be masking out calculations past the true sequence.
                    padding_values=(
                            src_eos_id,    # src
                            src_char_eos_id,     # src_char
                            0,    # src_len -- unused
                            0))    # src_char_len -- unused


        else:
            return x.padded_batch(
                    batch_size,
                    # The entry is the source line rows;
                    # this has unknown-length vectors.    The last entry is
                    # the source row size; this is a scalar.
                    padded_shapes=(
                            tf.TensorShape([None]),    # src
                            tf.TensorShape([])),    # src_len
                    # Pad the source sequences with eos tokens.
                    # (Though notice we don't generally need to do this since
                    # later on we will be masking out calculations past the true sequence.
                    padding_values=(
                            src_eos_id,    # src
                            0))    # src_len -- unused

    batched_dataset = batching_func(src_dataset, char)
    batched_iter = batched_dataset.make_initializable_iterator()
    src_char_ids = None
    word_len = None
    if char:
        (src_ids, src_char_ids, src_seq_len, word_len) = batched_iter.get_next()
    else:
        (src_ids, src_seq_len) = batched_iter.get_next()
    return BatchedInput(
            initializer=batched_iter.initializer,
            source=src_ids,
            source_char=src_char_ids,
            target_input=None,
            target_output=None,
            source_sequence_length=src_seq_len,
            target_sequence_length=None,
            word_len=word_len)


def get_iterator(src_dataset,
                                 tgt_dataset,
                                 src_vocab_table,
                                 tgt_vocab_table,
                                 batch_size,
                                 sos,
                                 eos,
                                 random_seed,
                                 num_buckets,
                                 src_max_len=None,
                                 tgt_max_len=None,
                                 num_parallel_calls=4,
                                 output_buffer_size=None,
                                 skip_count=None,
                                 num_shards=1,
                                 shard_index=0,
                                 reshuffle_each_iteration=True,
                                 src_char_vocab_table=None):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    char = not(src_char_vocab_table is None)
    if char:
        src_char_eos_id = tf.cast(src_char_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
            output_buffer_size, random_seed, reshuffle_each_iteration)

    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (
                    tf.string_split([src]).values, tf.string_split([tgt]).values),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src[:src_max_len], tgt),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src, tgt[:tgt_max_len]),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Convert the word strings to ids.    Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    if char:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                    tf.map_fn(lambda word: pad_tensor(tf.cast(src_char_vocab_table.lookup(
                    tf.string_split([word], delimiter="").values), tf.int32),
                    WORD_MAX_LEN, src_char_eos_id), src , tf.int32, infer_shape=False),
                    tf.map_fn(lambda word: tf.size(tf.string_split([word], delimiter="").values),src,dtype=tf.int32),
                    tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, src_char, src_char_len, tgt: (src, src_char,
                                        src_char_len,
                                        tf.concat(([tgt_sos_id], tgt), 0),
                                        tf.concat((tgt, [tgt_eos_id]), 0)
                                        ),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        # Add in sequence lengths.
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, src_char, src_char_len, tgt_in, tgt_out: (
                                                        src, src_char, tgt_in, tgt_out,
                    src_char_len, tf.size(src), tf.size(tgt_in)),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    else:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                                    tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src,
                                                    tf.concat(([tgt_sos_id], tgt), 0),
                                                    tf.concat((tgt, [tgt_eos_id]), 0)),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
        # Add in sequence lengths.
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt_in, tgt_out: (
                        src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x, char=False):
        if char:
            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.    The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                        tf.TensorShape([None]),    # src
                        tf.TensorShape([None, None]),    # src_char
                        tf.TensorShape([None]),    # tgt_input
                        tf.TensorShape([None]),    # tgt_output
                        tf.TensorShape([None]),     # src_char_len
                        tf.TensorShape([]),    # src_len
                        tf.TensorShape([])),    # tgt_len
                # Pad the source and target sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                        src_eos_id,    # src
                        src_char_eos_id,    # src_char
                        tgt_eos_id,    # tgt_input
                        tgt_eos_id,    # tgt_output
                        0,    # src_char_len -- unused
                        0,    # src_len -- unused
                        0))    # tgt_len -- unused
        else:

            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.    The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                        tf.TensorShape([None]),    # src
                        tf.TensorShape([None]),    # tgt_input
                        tf.TensorShape([None]),    # tgt_output
                        tf.TensorShape([]),    # src_len
                        tf.TensorShape([])),    # tgt_len
                # Pad the source and target sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                        src_eos_id,    # src
                        tgt_eos_id,    # tgt_input
                        tgt_eos_id,    # tgt_output
                        0,    # src_len -- unused
                        0))    # tgt_len -- unused

    if num_buckets > 1:

        def key_func(*inputs):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.    Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(inputs[-2] // bucket_width, inputs[-1] // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data,char)

        batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(
                        key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset, char = char)
    batched_iter = batched_dataset.make_initializable_iterator()
    src_char_ids=None
    word_len = None
    if char:
        (src_ids, src_char_ids, tgt_input_ids, tgt_output_ids, word_len, src_seq_len,
        tgt_seq_len) = (batched_iter.get_next())
    else:
        (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
        tgt_seq_len) = (batched_iter.get_next())
    return BatchedInput(
            initializer=batched_iter.initializer,
            source=src_ids,
            source_char=src_char_ids,
            target_input=tgt_input_ids,
            target_output=tgt_output_ids,
            source_sequence_length=src_seq_len,
            target_sequence_length=tgt_seq_len,
            word_len=word_len)
