import os
import sys
import time
import models_GAN as models
import numpy as np
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import matplotlib.pylab as plt

# Utils
sys.path.append("../utils")
import general_utils
import data_utils

def inverse_normalization(X):

    return (X + 1.) / 2.

def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    generator = kwargs["generator"]
    model_name = kwargs["model_name"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    img_dim = kwargs["img_dim"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    noise_scale = kwargs["noise_scale"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)

    # Load and rescale data
    # if dset == "celebA":
    #     X_real_train = data_utils.load_celebA(img_dim, image_dim_ordering)
    # if dset == "mnist":
    #     X_real_train, _, _, _ = data_utils.load_mnist(image_dim_ordering)
    # img_dim = X_real_train.shape[-3:]
    img_dim = (3,64,64)
    noise_dim = (100,)

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)

        # Load generator model
        generator_model = models.load("generator_%s" % generator,
                                      noise_dim,
                                      img_dim,
                                      bn_mode,
                                      batch_size,
                                      dset=dset,
                                      use_mbd=use_mbd)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          noise_dim,
                                          img_dim,
                                          bn_mode,
                                          batch_size,
                                          dset=dset,
                                          use_mbd=use_mbd)
        #load the weights here
        for e in range(200,355,5):
            gen_weights_path = os.path.join('../../CNN/gen_weights_epoch%s.h5' % (e))
            # gen_weight_file  = h5py.File(gen_weights_path, 'r')
            file_path_to_save_img = 'GeneratedImages1234/Epoch_%s/'%(e)
            os.mkdir(file_path_to_save_img)
            # generate images
            generator_model.load_weights(gen_weights_path)
            generator_model.compile(loss='mse', optimizer=opt_discriminator)
            noise_z = np.random.normal(scale=0.5, size=(32, noise_dim[0]))
            X_generated = generator_model.predict(noise_z)
            # print('Epoch%s.png' % (i))
            X_gen = inverse_normalization(X_generated)
            for img in range(X_gen.shape[0]):
                ret = X_gen[img].transpose(1,2,0)
                fig = plt.figure(frameon=False)
                fig.set_size_inches(64,64)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(ret, aspect='normal')
                fig.savefig(file_path_to_save_img+'retina_%s.png' % (img), dpi=1)
                plt.clf()
                plt.close()

            # Xg = X_gen[:8]
            # Xr = X_gen[8:]
            #
            # if image_dim_ordering == "tf":
            #     X = np.concatenate((Xg, Xr), axis=0)
            #     list_rows = []
            #     for i in range(int(X.shape[0] / 4)):
            #         Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            #         list_rows.append(Xr)
            #
            #     Xr = np.concatenate(list_rows, axis=0)
            #
            # if image_dim_ordering == "th":
            #     X = np.concatenate((Xg, Xr), axis=0)
            #     list_rows = []
            #     for i in range(int(X.shape[0] / 4)):
            #         Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            #         list_rows.append(Xr)
            #
            #     Xr = np.concatenate(list_rows, axis=1)
            #     Xr = Xr.transpose(1,2,0)
            #
            # if Xr.shape[-1] == 1:
            #     plt.imshow(Xr[:, :, 0], cmap="gray")
            # else:
            #     plt.imshow(Xr)
            # plt.savefig(file_path_to_save_img+'Epoch%s.png' % (e))
            # plt.clf()
            # plt.close()


        # generator_model.load_weights('gen_weights_epoch245.h5')
        # generator_model.compile(loss='mse', optimizer=opt_discriminator)
        # discriminator_model.trainable = False
        #
        # DCGAN_model = models.DCGAN(generator_model,
        #                            discriminator_model,
        #                            noise_dim,
        #                            img_dim)
        #
        # loss = ['binary_crossentropy']
        # loss_weights = [1]
        # DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)
        #
        # discriminator_model.trainable = True
        # discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)


        # noise_z = np.random.normal(scale=0.5, size=(32, noise_dim[0]))
        # X_generated = generator_model.predict(noise_z)
        #
        # X_gen = inverse_normalization(X_generated)
        #
        # Xg = X_gen[:8]
        # Xr = X_gen[8:]
        #
        # if image_dim_ordering == "tf":
        #     X = np.concatenate((Xg, Xr), axis=0)
        #     list_rows = []
        #     for i in range(int(X.shape[0] / 4)):
        #         Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
        #         list_rows.append(Xr)
        #
        #     Xr = np.concatenate(list_rows, axis=0)
        #
        # if image_dim_ordering == "th":
        #     X = np.concatenate((Xg, Xr), axis=0)
        #     list_rows = []
        #     for i in range(int(X.shape[0] / 4)):
        #         Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
        #         list_rows.append(Xr)
        #
        #     Xr = np.concatenate(list_rows, axis=1)
        #     Xr = Xr.transpose(1,2,0)
        #
        # if Xr.shape[-1] == 1:
        #     plt.imshow(Xr[:, :, 0], cmap="gray")
        # else:
        #     plt.imshow(Xr)
        # plt.savefig("current_batch.png")
        # plt.clf()
        # plt.close()

        # gen_loss = 100
        # disc_loss = 100
        #
        # # Start training
        # print("Start training")
        # k = 0
        # for e in range(nb_epoch):
        #     # Initialize progbar and batch counter
        #     progbar = generic_utils.Progbar(epoch_size)
        #     batch_counter = 1
        #     start = time.time()
        #
        #     for X_real_batch in data_utils.gen_batch(X_real_train, batch_size):
        #
        #         # Create a batch to feed the discriminator model
        #         X_disc, y_disc = data_utils.get_disc_batch(X_real_batch,
        #                                                    generator_model,
        #                                                    batch_counter,
        #                                                    batch_size,
        #                                                    noise_dim,
        #                                                    noise_scale=noise_scale,
        #                                                    label_smoothing=label_smoothing,
        #                                                    label_flipping=label_flipping)
        #
        #         # Update the discriminator
        #         disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
        #
        #         # Create a batch to feed the generator model
        #         X_gen, y_gen = data_utils.get_gen_batch(batch_size, noise_dim, noise_scale=noise_scale)
        #
        #         # Freeze the discriminator
        #         discriminator_model.trainable = False
        #         gen_loss = DCGAN_model.train_on_batch(X_gen, y_gen)
        #         # Unfreeze the discriminator
        #         discriminator_model.trainable = True
        #
        #         batch_counter += 1
        #         progbar.add(batch_size, values=[("D logloss", disc_loss),
        #                                         ("G logloss", gen_loss)])
        #
        #         # Save images for visualization
        #         if batch_counter % 100 == 0:
        #             data_utils.plot_generated_batch(X_real_batch, generator_model,
        #                                             batch_size, noise_dim, image_dim_ordering,k)
        #             k = k +1
        #         if batch_counter >= n_batch_per_epoch:
        #             break
        #
        #     print("")
        #     print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
        #
        #     if e % 5 == 0:
        #         gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%s.h5' % (model_name, e))
        #         generator_model.save_weights(gen_weights_path, overwrite=True)
        #
        #         disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%s.h5' % (model_name, e))
        #         discriminator_model.save_weights(disc_weights_path, overwrite=True)
        #
        #         DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
        #         DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
