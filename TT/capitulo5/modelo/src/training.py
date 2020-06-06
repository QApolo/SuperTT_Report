@tf.function
def train_step(img_tensor, target):
  loss = 0
  loss_counter = 0
  loss_before = tf.constant(0.0)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor, True)
      tf.debugging.assert_all_finite(features, 'the features have exploded')

      hidden = initial_state(features)
      carry = initial_carry(features)
      out = initial_out(features)
      dec_input = tf.constant([BEGIN] * target.shape[0])

      hidden_and_carry = [hidden, carry]

      B = tf.zeros((features.shape[0], features.shape[1], 1))

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, out, attention_weights, hidden_and_carry = decoder(features, hidden_and_carry, dec_input, out, B)
          B += attention_weights

          real = get_num_token_end(target[:, i])

          loss += loss_function(real, predictions)
          tf.debugging.assert_all_finite(loss, 'the loss has exploded')

          # is_done = tf.where(tf.less(real, end_tensor), is_done, end_tensor)
          if loss != loss_before:
            loss_counter += 1
            loss_before = loss

          # using teacher forcing
          dec_input = real

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + initial_state.trainable_variables + initial_carry.trainable_variables + initial_out.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  for g in gradients:
    tf.debugging.assert_all_finite(g, 'the gradients have exploded')

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  train_loss(loss)
  # train_accuracy(y_train, predictions)

  return loss, total_loss, loss_counter



start = time.time()
total_loss = 0
total_val_loss = 0

for (batch, (img_tensor, target)) in enumerate(dataset):
    # img_tensor = noise(img_tensor, training=True)
    batch_loss, t_loss, loss_counter = train_step(img_tensor, target)
    total_loss += t_loss

    print ('Epoch {} Batch {} Loss {:.6f}'.format(
      epoch, batch, batch_loss.numpy() / loss_counter))
