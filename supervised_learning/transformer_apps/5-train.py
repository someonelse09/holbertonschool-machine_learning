#!/usr/bin/env python3
"""
Train transformer model for machine translation
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule for transformer
    """

    def __init__(self, dm, warmup_steps=4000):
        """
        Initialize custom schedule

        Args:
            dm: dimensionality of the model
            warmup_steps: number of warmup steps
        """
        super(CustomSchedule, self).__init__()

        self.dm = dm
        self.dm = tf.cast(self.dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Calculate learning rate for given step

        Args:
            step: current training step

        Returns:
            learning rate
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Create and train a transformer model for machine translation

    Args:
        N: number of blocks in encoder and decoder
        dm: dimensionality of the model
        h: number of heads
        hidden: number of hidden units in fully connected layers
        max_len: maximum number of tokens per sequence
        batch_size: batch size for training
        epochs: number of epochs to train for

    Returns:
        trained transformer model
    """
    # Load dataset
    dataset = Dataset(batch_size, max_len)

    # Get vocabulary sizes (vocab_size + 2 for start and end tokens)
    input_vocab = dataset.tokenizer_pt.vocab_size + 2
    target_vocab = dataset.tokenizer_en.vocab_size + 2

    # Create transformer model
    transformer = Transformer(
        N, dm, h, hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len
    )

    # Create learning rate schedule
    learning_rate = CustomSchedule(dm)

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    # Loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    def loss_function(real, pred):
        """
        Calculate loss ignoring padding tokens

        Args:
            real: real target values
            pred: predicted values

        Returns:
            loss value
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

    @tf.function
    def train_step(inp, tar):
        """
        Single training step

        Args:
            inp: input tensor
            tar: target tensor

        Returns:
            None
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        # Create masks
        enc_mask, combined_mask, dec_mask = create_masks(
            inp, tar_inp
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inp, tar_inp, True, enc_mask, combined_mask,
                dec_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(
            loss, transformer.trainable_variables
        )
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # Training loop
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        batch_num = 0
        for (inp, tar) in dataset.data_train:
            train_step(inp, tar)

            if batch_num % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch_num}: '
                    f'Loss {train_loss.result()}, '
                    f'Accuracy {train_accuracy.result()}'
                )

            batch_num += 1

        print(
            f'Epoch {epoch + 1}: '
            f'Loss {train_loss.result()}, '
            f'Accuracy {train_accuracy.result()}'
        )

    return transformer
