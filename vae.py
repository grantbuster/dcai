# -*- coding: utf-8 -*-
"""
Extreme variational autoencoder (ExVae).
"""
import os
import shutil
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tempfile
import tensorflow as tf
import tensorflow_probability as tfp
from zipfile import ZipFile


class Vae(tf.keras.Model):
    """Extreme variational autoencoder."""

    def __init__(self, data, latent_dim=2, encoder=None, decoder=None,
                 kernel_size=3, strides=(1, 1), kl_weight=0.01, optimizer=None,
                 learning_rate=1e-4, validation_split=0.1,
                 standardize_data=True, mean=None, std=None):
        """
        Parameters
        ----------
        data : np.ndarray
            3D array of source data for training/validation/test.
            Shape is: (n_obs, hourly_timeseries, n_features). This is assumed
            to be (n_obs, 24*7 hour per week, dry bulb and wet bulb
            temperatures for 73 sites)
        latent_dim : int
            Number of dimensions in the latent space.
        encoder : None | tf.keras.Sequential
            Optional pre-trained keras encoder model.
        decoder : None | tf.keras.Sequential
            Optional pre-trained keras decoder model.
        kernel_size : int
            Conv kernel size.
        strides : int
            Conv stride length for the encoder. Decoder stride is fixed.
        kl_weight : float
            Weight of the KL-divergence term (also sometimes referred
            to as beta)
        optimizer : None | tf.keras.optimizer
            Optional pre-trained keras optimizer model.
        learning_rate : float
            Learning rate for the optimizer
        validation_split : float
            Fraction of data to hold out for validation
        train_percentile : int | float
            Extremeness metric percentile below which training data is gathered
            from. For example, if this is 60, then only weeks with extremeness
            below the 60th percentile of the full input dataset will be used
            for training.
        standardize_data : bool
            Flag to standarize data
        mean : None | float | list | np.ndarray
            Mean value of the training data to be used for standardizing
            the training/validation/test datasets. Should only be input
            if pre-trained encoder+decoder models are being used
        std : None | float | list | np.ndarray
            Standard dev of the training data to be used for standardizing
            the training/validation/test datasets. Should only be input
            if pre-trained encoder+decoder models are being used
        """

        super(Vae, self).__init__()

        self.data = data
        self.latent_dim = latent_dim
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self._mean = np.array(mean) if isinstance(mean, list) else mean
        self._std = np.array(std) if isinstance(std, list) else std
        self._standardize_data = standardize_data
        self._encoder = encoder
        self._decoder = decoder
        self.kernel_size = kernel_size
        self.strides = strides
        self.kl_weight = kl_weight
        self._optimizer = optimizer
        self.train_history = pd.DataFrame()

        self.val_index = np.random.choice(
            np.arange(len(self.data)),
            size=int(self.validation_split * len(self.data)),
            replace=False)

        self.train_index = np.array(list(set(np.arange(len(self.data)))
                                         - set(self.val_index)))

        self._set_mean_std()

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    @property
    def encoder(self):
        """Get the encoder model

        Returns
        -------
        tf.keras.sequential
        """
        if self._encoder is None:
            self._encoder = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(
                     filters=16, kernel_size=self.kernel_size,
                     strides=self.strides, activation='relu',
                     padding='valid', input_shape=(32, 32, 1)),
                 tf.keras.layers.Conv2D(
                     filters=32, kernel_size=self.kernel_size,
                     strides=self.strides, activation='relu',
                     padding='valid'),
                 tf.keras.layers.Conv2D(
                     filters=64, kernel_size=self.kernel_size,
                     strides=self.strides, activation='relu',
                     padding='valid'),
                 tf.keras.layers.Flatten(),
                 tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
                ])

        return self._encoder

    @property
    def decoder(self):
        """Get the decoder model

        Returns
        -------
        tf.keras.sequential
        """
        if self._decoder is None:
            strides = 1
            self._decoder = tf.keras.Sequential(
                [tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                 tf.keras.layers.Dense(units=26*26*16,
                                       activation=tf.nn.relu),
                 tf.keras.layers.Reshape(target_shape=(26, 26, 16)),
                 tf.keras.layers.Conv2DTranspose(
                     filters=64, kernel_size=self.kernel_size,
                     strides=strides, padding='valid', activation='relu'),
                 tf.keras.layers.Conv2DTranspose(
                     filters=32, kernel_size=self.kernel_size,
                     strides=strides, padding='valid', activation='relu'),
                 tf.keras.layers.Conv2DTranspose(
                     filters=16, kernel_size=self.kernel_size,
                     strides=strides, padding='valid', activation='relu'),
                 tf.keras.layers.Conv2DTranspose(
                     filters=1, kernel_size=self.kernel_size,
                     strides=1, padding='same', activation='linear'),
                ])

        return self._decoder

    def _set_mean_std(self):
        """Set the data standardization parameters (mean and std)"""
        train_x = self.data[self.train_index]
        if self._standardize_data and self._mean is None:
            self._mean = train_x.mean()
            self._std = train_x.std()
        elif not self._standardize_data and self._mean is None:
            self._mean = 0
            self._std = 1

    @property
    def train_x(self):
        """Get standardized training dataset below the training percentile.

        Returns
        -------
        np.ndarray
        """
        train_x = self.data[self.train_index]
        train_x = self.standardize(train_x)
        return train_x

    @property
    def val_x(self):
        """Get standardized validation dataset (sampled from the full source
        dataset without the training percentile).

        Returns
        -------
        np.ndarray
        """
        val_x = self.data[self.val_index]
        val_x = self.standardize(val_x)
        return val_x

    @property
    def loss_stats(self):
        """Get a dataframe of common loss statistics (MAE, MBE)

        Returns
        -------
        pd.DataFrame
        """
        dsets = {'train': self.train_x,
                 'validation': self.val_x,
                 }
        loss_stats = pd.DataFrame()

        for label, x in dsets.items():
            mean, _ = self.encode(x)
            out = self.decode(mean)
            x = self.un_standardize(x)
            out = self.un_standardize(out)
            mae = np.mean(np.abs(out - x))
            mbe = np.mean(out - x)
            mse = np.mean((out - x)**2)

            loss_stats.at[label, 'MAE'] = mae
            loss_stats.at[label, 'MBE'] = mbe
            loss_stats.at[label, 'MSE'] = mse

        return loss_stats

    def standardize(self, data):
        """Standardize data

        Parameters
        ----------
        data : np.ndarray
            Array with same dimensionality as source data with physical units

        Returns
        -------
        data : np.ndarray
            Array with same dimensionality as source data with
            mean=0 and stdev=1
        """
        return (data - self._mean) / self._std

    def un_standardize(self, data):
        """Un-Standardize data

        Parameters
        ----------
        data : np.ndarray
            Array with same dimensionality as source data with
            mean=0 and stdev=1

        Returns
        -------
        data : np.ndarray
            Array with same dimensionality as source data with physical units
        """
        return data * self._std + self._mean

    def sample(self, eps=None, standardize=False, numpy=True,
               apply_sigmoid=True, to_image=True):
        """Generate a decoded output by randomly sampling the latent space

        Parameters
        ----------
        eps : None | tf.Tensor
            Optional samples from the latent space.

        Returns
        -------
        out : tf.Tensor
            Decoded output x.
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        out = self.decode(eps, standardize=standardize, numpy=numpy,
                          apply_sigmoid=apply_sigmoid, to_image=to_image)
        return out

    def encode(self, x):
        """Encode physical data x into the latent space

        Parameters
        ----------
        x : np.ndarray
            Physical data to encode in latent space. Should be standardized.

        Returns
        -------
        mean : np.ndarray
            Mean value in latent space of the input data x
        logvar : np.ndarray
            Log of the variance in latent space of the input data x
        """
        mean, logvar = tf.split(self.encoder(x.copy()),
                                num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, standardize=True, numpy=False, apply_sigmoid=False, to_image=True):
        """Decode a latent space sample z into a physical output x

        Parameters
        ----------
        z : np.ndarray
            Array of latent space variables.
        standardize : bool
            Flag to return x in mean=0 stdev=1 standardized format
            or un-standardized to physical units.
        numpy : bool
            Flag to output numpy array instead of tensor.

        Returns
        -------
        out : np.ndarray | tf.Tensor
            Decoded transformation of the latent space input z
        """

        out = self.decoder(z)

        if apply_sigmoid and isinstance(apply_sigmoid, (int, float)):
            out = self.sigmoid(out, weight=apply_sigmoid)
        elif apply_sigmoid:
            out = self.sigmoid(out)

        if not standardize:
            out = self.un_standardize(out)

        if numpy:
            out = out.numpy()

        if to_image:
            out = np.round(out[:, :, :, 0] * 255).astype(np.uint8)

        return out

    def _reparameterize(self, mean, logvar):
        """Re-parameterize the encoded mean and log-variance by applying
        gaussian noise. This is a trick to add variance while allowing for back
        propagation"""
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @staticmethod
    def sigmoid(x, weight=1):
        return 1 / (1 + tf.math.exp(-weight*x))

    @staticmethod
    def log_normal_pdf(z, mean, logvar, raxis=1):
        """Log of the normal pdf

        Parameters
        ----------
        z : np.ndarray | tf.Tensor
            Data sample from the latent space.
        mean : np.ndarray | tf.Tensor | float
            Mean value of the gaussian.
        logvar : np.ndarray | tf.Tensor | float
            Log of the variance of the gaussian.
        raxis : int
            Axis to reduce the output across.

        Returns
        -------
        out : np.ndarray | tf.Tensor | float
            Log of the normal probability of z given the gaussian with
            mean and logvar.
        """
        log2pi = tf.math.log(2. * np.pi)
        out = -.5 * ((z - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
        out = tf.reduce_sum(out, axis=raxis)
        return out

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self._reparameterize(mean, logvar)
        x_logit = self.decode(z, numpy=False, apply_sigmoid=False,
                              to_image=False)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        loss = -tf.reduce_mean(logpx_z + self.kl_weight * (logpz - logqz_x))
        return loss, tf.reduce_mean(logpx_z), tf.reduce_mean(logpz - logqz_x)

    @staticmethod
    def make_batches(x, n_batch=16, shuffle=True):
        """Make lists of unique data batches by splitting x along the
        1st data dimension.

        Parameters
        ----------
        x : np.ndarray
            Feature data for training
        n_batch : int
            Number of times to update the NN weights per epoch. The training
            data will be split into this many batches and the NN will train on
            each batch, update weights, then move onto the next batch.
        shuffle : bool
            Flag to randomly subset the validation data from x and y.

        Returns
        -------
        x_batches : list
            List of ND arrays that are split subsets of x.
            ND matches input dimension. Length of list is n_batch.
        """

        L = x.shape[0]
        if shuffle:
            i = np.random.choice(L, replace=False, size=(L,))
            assert len(set(i)) == L
        else:
            i = np.arange(L)

        batch_indexes = np.array_split(i, n_batch)

        x_batches = [x[j] for j in batch_indexes]

        return x_batches

    def train(self, n_epoch=1, n_batch=1):
        """Train the encoder and decoder models on data x and apply
        gradient descent to update model weights.
        """

        for i in range(n_epoch):

            batches = self.make_batches(self.train_x, n_batch)

            for x_batch in batches:
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(x_batch)[0]

                gradients = tape.gradient(loss, self.trainable_variables)
                zipped = zip(gradients, self.trainable_variables)
                self.optimizer.apply_gradients(zipped)

            train_loss, train_rec, train_kl = self.compute_loss(self.train_x)
            val_loss, val_rec, val_kl = self.compute_loss(self.val_x)

            self.train_history.at[i, 'train_loss'] = train_loss.numpy()
            self.train_history.at[i, 'train_rec_loss'] = train_rec.numpy()
            self.train_history.at[i, 'train_kl_loss'] = train_kl.numpy()
            self.train_history.at[i, 'val_loss'] = val_loss.numpy()
            self.train_history.at[i, 'val_rec_loss'] = val_rec.numpy()
            self.train_history.at[i, 'val_kl_loss'] = val_kl.numpy()

            print('Epoch #{}, train loss: {:.3f}, validation loss: {:.3f}'
                  .format(i, train_loss, val_loss))

    def test_plot(self, x, epoch=0, save_dir='plots', show=True, save=True,
                  plot_channel=0):
        """Test the model by encoding and decoding test data and plotting
        the true vs. reconstructed data

        Parameters
        ----------
        x : np.ndarray
            Training, validation, or test data to encode/decode and
            plot next to the reconstructed output. Must have 8 observations.
        epoch : int
            Training epoch number to tag the image filename.
        save_dir : str
            Relative directory to save images to.
        show : bool
            Flag to show image.
        save : bool
            Flag to save image file.
        plot_channel : int
            Data channel (last axis) to plot
        """

        assert x.shape[0] == 12
        mean, logvar = self.encode(x)
        x_out = self.decode(mean)
        fig = plt.figure(figsize=(15, 5))

        for i in range(x_out.shape[0]):
            plt.subplot(3, 4, i + 1)
            plt.plot(x[i, :, plot_channel])
            plt.plot(x_out[i, :, plot_channel])
            plt.axis('off')

        if save:
            fp = './{}/ExVae_epoch_{:04d}.png'.format(save_dir, epoch)
            save_dir = os.path.dirname(os.path.abspath(fp))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(fp)

        if show:
            plt.show()

        plt.close()

    @property
    def model_params(self):
        """Get a namespace of model parameters for save/load

        Returns
        -------
        dict
        """

        mean = self._mean
        std = self._std
        if isinstance(self._mean, np.ndarray):
            mean = self._mean.tolist()
            std = self._std.tolist()

        out = {'mean': mean,
               'std': std,
               'latent_dim': self.latent_dim,
               'kernel_size': self.kernel_size,
               'strides': self.strides,
               'kl_weight': self.kl_weight,
               'validation_split': self.validation_split,
               'learning_rate': self.learning_rate,
               'standardize_data': self._standardize_data,
               }
        return out

    def save(self, fpath):
        """Save the model to a zip filename

        Parameters
        ----------
        fpath : str
            Filepath to save the model zip file.
        """
        fpath = os.path.abspath(fpath)
        if not fpath.endswith('.zip'):
            fpath += '.zip'

        with tempfile.TemporaryDirectory() as td:
            temp_path = os.path.join(td, os.path.basename(fpath))
            fps = [os.path.join(td, 'encoder_model'),
                   os.path.join(td, 'decoder_model'),
                   os.path.join(td, 'params.json'),
                   ]

            self.encoder.save(fps[0])
            self.decoder.save(fps[1])
            with open(fps[2], 'w') as f:
                json.dump(self.model_params, f, indent=2)

            with ZipFile(temp_path, 'w') as zf:
                for folder, _, fns in os.walk(td):
                    for fn in fns:
                        if fn != os.path.basename(fpath):
                            fp = os.path.join(folder, fn)
                            target = fp.replace(td, '')
                            zf.write(fp, target)

            shutil.copy(temp_path, fpath)

    @classmethod
    def load(cls, data, fpath, **kwargs):
        """Load a zip archived ExVae models

        Parameters
        ----------
        data : np.ndarray
            3D array of source data for training/validation/test.
            Shape is: (n_obs, 1_week_hourly, n_features)
        fpath : str
            Zip file from which to load model
        kwargs : dict
            Optional initialization namespace arguments to override the
            saved model parameters
        """
        fpath = os.path.abspath(fpath)
        assert fpath.endswith('.zip')

        with tempfile.TemporaryDirectory() as td:
            temp_path = os.path.join(td, os.path.basename(fpath))
            fps = [os.path.join(td, 'encoder_model'),
                   os.path.join(td, 'decoder_model'),
                   os.path.join(td, 'params.json'),
                   ]
            shutil.copy(fpath, temp_path)

            with ZipFile(temp_path, 'r') as zf:
                zf.extractall(td)

            encoder = tf.keras.models.load_model(fps[0])
            decoder = tf.keras.models.load_model(fps[1])
            with open(fps[2], 'r') as f:
                params = json.load(f)

        params.update(kwargs)
        model = cls(data, encoder=encoder, decoder=decoder, **params)
        return model
