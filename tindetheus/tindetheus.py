# MIT License
#
# Copyright (c) 2017, 2018 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The facenet implementation has been hard coded into tindetheus. This has been
# hardcoded into tindetheus for the following reasons: 1) there is no setup.py
# for facenet yet. 2) to prevent changes to facenet from breaking tindetheus.
#
# facenet is used to align the database, crop the faces in database, and
# to calculate the embeddings for the database. I've included the copyright
# from facenet below. The specific code that is in this file from facenet
# is within the like_or_dislike_users(self, users) function.

# facenet was created by David Sandberg and is available at
# https://github.com/davidsandberg/facenet with the following MIT license:

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# import os
# import sys
#
# scriptpath = "../../"
# sys.path.append(os.path.abspath(scriptpath))
# import test
# print("importing parent file")
# print(test.func())


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input


import sys
import os
import shutil

import pandas as pd


import matplotlib.pyplot as plt
import imageio
import numpy as np
try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve

import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

import facenet.src.facenet as facenet
import tensorflow as tf




from skimage.transform import resize
from skimage import img_as_ubyte

from facenet.src.align import detect_face
import random
from time import sleep





# add version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()


def clean_temp_images():
    # delete the temp_images dir
    shutil.rmtree('temp_images')
    os.makedirs('temp_images')


def clean_temp_images_aligned():
    # delete the temp images aligned dir
    shutil.rmtree('temp_images_aligned')


def download_url_photos(urls, userID, is_temp=False):
    # define a function which downloads the pictures of urls
    count = 0
    image_list = []
    if is_temp is True:
        os.makedirs('temp_images/temp')
    for url in urls:
        if is_temp is True:
            image_list.append('temp_images/temp/'+userID+'.'+str(count)+'.jpg')
        else:
            image_list.append('temp_images/'+userID+'.'+str(count)+'.jpg')
        urlretrieve(url, image_list[-1])
        count += 1
    return image_list


def move_images_temp(image_list, userID):
    # move images from temp folder to al_database
    count = 0
    database_loc = []
    for i, j in enumerate(image_list):
        new_fname = 'al_database/'+userID+'.'+str(count)+'.jpg'
        try:
            os.rename(j, new_fname)
        except:
            print('WARNING: unable to save file, it may already exist!',
                  'file: ' + new_fname)
        database_loc.append(new_fname)
        count += 1
    return database_loc


def show_images(images, holdon=False, title=None, nmax=49):
    # use matplotlib to display profile images
    n = len(images)
    if n > nmax:
        n = nmax
        n_col = 7
    else:
        n_col = 3
    if n % n_col == 0:
        n_row = n // n_col
    else:
        n_row = n // 3 + 1
    if title is None:
        plt.figure()
    else:
        plt.figure(title)
    plt.tight_layout()
    for j, i in enumerate(images):
        if j == nmax:
            print('\n\nToo many images to show... \n\n')
            break
        temp_image = imageio.imread(i)
        if len(temp_image.shape) < 3:
            # needs to be converted to rgb
            temp_image = to_rgb(temp_image)
        plt.subplot(n_row, n_col, j+1)
        plt.imshow(temp_image)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)

    if holdon is False:
        plt.show(block=False)
        plt.pause(0.1)


def calc_avg_emb():
    # a function to create a vector of n average embeddings for each
    # tinder profile
    # get the embeddings per profile
    labels = np.load('labels.npy')
    # label_strings = np.load('label_strings.npy')
    embeddings = np.load('embeddings.npy')
    image_list = np.load('image_list.npy')

    # determine the n dimension of embeddings
    n_emb = embeddings.shape[1]

    # find the maximum number of images in a profile
    split_image_list = []
    profile_list = []
    for i in image_list:
        split_image_list.append(i.split('.')[1])
        # split_image_list.append(i.replace('/', '.').split('.'))
        profile_list.append(i.split('.')[0])

    # convert profile list to pandas index
    pl = pd.Index(profile_list)
    pl_unique = pl.value_counts()

    # get the summar statics of pl
    pl_describe = pl_unique.describe()
    print('Summary statistics of profiles with at least one detectable face')
    print(pl_describe)
    number_of_profiles = int(pl_describe[0])
    # number_of_images = int(pl_describe[-1])

    # convert the embeddings to a data frame
    eb = pd.DataFrame(embeddings, index=pl)
    dislike = pd.Series(labels, index=pl)
    # if dislike == 1 it means I disliked the person!

    # create a blank numpy array for embeddings
    new_embeddings = np.zeros((number_of_profiles, n_emb))
    new_labels = np.zeros(number_of_profiles)
    for i, j in enumerate(pl_unique.index):
        temp = eb.loc[j]

        # if a profile has more than one face it will be a DataFrame
        # else the profile will be a Series
        if isinstance(temp, pd.DataFrame):
            # get the average of each column
            temp_embeddings = np.mean(temp.values, axis=0)
        else:
            temp_embeddings = temp.values

        # save the new embeddings
        new_embeddings[i] = temp_embeddings

        # Save the profile label, 1 for dislike, 0 for like
        new_labels[i] = dislike[j].max()

    # save the files
    np.save('embeddings_avg_profile.npy', new_embeddings)
    np.save('labels_avg_profile.npy', new_labels)
    return new_embeddings, new_labels


def calc_avg_emb_temp(embeddings):
    # a function to create a vector of n average embeddings for each
    # in the temp_images_aligned folder
    # embeddings = np.load('temp_embeddings.npy')
    # determine the n dimension of embeddings
    n_emb = embeddings.shape[1]
    # calculate the average embeddings
    new_embeddings = np.zeros((1, n_emb))
    new_embeddings[0] = np.mean(embeddings, axis=0)
    return new_embeddings


def fit_log_reg(X, y):
    # fits a logistic regression model to your data
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)
    print('Train size: ', len(X))
    train_score = model.score(X, y)
    print('Training accuracy', train_score)
    ypredz = model.predict(X)
    cm = confusion_matrix(y, ypredz)
    # tn, fp, fn, tp = cm.ravel()
    tn, _, _, tp = cm.ravel()

    # true positive rate When it's actually yes, how often does it predict yes?
    recall = float(tp) / np.sum(cm, axis=1)[1]
    # Specificity: When it's actually no, how often does it predict no?
    specificity = float(tn) / np.sum(cm, axis=1)[0]

    print('Recall/ Like accuracy', recall)
    print('specificity/ Dislike accuracy', specificity)

    # save the model
    joblib.dump(model, 'log_reg_model.pkl')


def like_or_dislike():
    # define a function to get like or dislike input
    likeOrDislike = '0'
    while likeOrDislike != 'j' and likeOrDislike != 'l' \
            and likeOrDislike != 'f' and likeOrDislike != 's':

        likeOrDislike = input()
        if likeOrDislike == 'j' or likeOrDislike == 'f':
            return 'Dislike'
        elif likeOrDislike == 'l' or likeOrDislike == 's':
            return 'Like'
        else:
            print('you must enter either l or s for like,'
                  ' or j or f for dislike')
            likeOrDislike = input()




    def like_or_dislike_users(self, users):
        # automatically like or dislike users based on your previously trained
        # model on your historical preference.

        # facenet settings from export_embeddings....
        data_dir = 'temp_images_aligned'
        embeddings_name = 'temp_embeddings.npy'
        # labels_name = 'temp_labels.npy'
        # labels_strings_name = 'temp_label_strings.npy'
        is_aligned = True
        image_size = 160
        margin = 44
        gpu_memory_fraction = 1.0
        image_batch = 1000
        prev_user = None
        for user in users:
            clean_temp_images()
            urls = user.get_photos(width='640')
            image_list = download_url_photos(urls, user.id,
                                             is_temp=True)
            # align the database
            tindetheus_align.main(input_dir='temp_images',
                                  output_dir='temp_images_aligned')
            # export the embeddings from the aligned database

            train_set = facenet.get_dataset(data_dir)
            image_list_temp, label_list = facenet.get_image_paths_and_labels(train_set)  # noqa: E501

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")  # noqa: E501
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # noqa: E501
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")  # noqa: E501

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list_temp)
            print('Number of images: ', nrof_images)
            batch_size = image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size
            else:
                nrof_batches = (nrof_images // batch_size) + 1
            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches - 1:
                    n = nrof_images
                else:
                    n = i*batch_size + batch_size
                # Get images for the batch
                if is_aligned is True:
                    images = facenet.load_data(image_list_temp[i*batch_size:n],  # noqa: E501
                                                False, False,
                                                image_size)
                else:
                    images = load_and_align_data(image_list_temp[i*batch_size:n],  # noqa: E501
                                                    image_size, margin,
                                                    gpu_memory_fraction)
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                # Use the facenet model to calculate embeddings
                embed = self.sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            # export embeddings and labels
            label_list = np.array(label_list)

            np.save(embeddings_name, emb_array)

            if emb_array.size > 0:
                # calculate the n average embedding per profiles
                X = calc_avg_emb_temp(emb_array)
                # evaluate on the model
                yhat = self.model.predict(X)

                if yhat[0] == 1:
                    didILike = 'Like'
                    # check to see if this is the same user as before
                    if prev_user == user.id:
                        clean_temp_images_aligned()
                        print('\n\n You have already liked this user!!! \n \n')
                        print('This typically means you have used all of your'
                              ' free likes. Exiting program!!! \n\n')
                        self.likes_left = -1
                        return
                    else:
                        prev_user = user.id
                else:
                    didILike = 'Dislike'
            else:
                # there were no faces in this profile
                didILike = 'Dislike'
            print('**************************************************')
            print(user.name, user.age, didILike)
            print('**************************************************')

            dbase_names = move_images_temp(image_list, user.id)

            if didILike == 'Like':
                print(user.like())
                self.likes_left -= 1
            else:
                print(user.dislike())
            userList = [user.id, user.name, user.age, user.bio,
                        user.distance_km, user.jobs, user.schools,
                        user.get_photos(width='640'), dbase_names,
                        didILike]
            self.al_database.append(userList)
            np.save('al_database.npy', self.al_database)
            clean_temp_images_aligned()


    def like(self):
        # like and dislike Tinder profiles using your trained logistic
        # model. Note this requires that you first run tindetheus browse to
        # build a database. Then run tindetheus train to train a model.

        # load the pretrained model
        self.model = joblib.load('log_reg_model.pkl')

        while self.likes_left > 0:
            try:
                users = self.session.nearby_users()
                self.like_or_dislike_users(users)
            except RecsTimeout:
                    self.search_distance += 5
                    self.session.profile.distance_filter += 5
                    self.like()


def align(input_dir='database', output_dir='database_aligned', image_size=182,
         margin=44, gpu_memory_fraction=1, random_order=False):
    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store some git revision info in a text file in the log directory
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    cwd = os.getcwd()
    facenet.store_revision_info(cwd, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(input_dir)
    print(input_dir)
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  # noqa: E501
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                          log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # Add a random key to the filename to allow alignment using multiple
    # processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)  # noqa: E501

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir,
                                               filename+'.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = imageio.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)  # noqa: E501

                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])  # noqa: E501
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0]+det[:, 2])/2-img_center[1], (det[:, 1]+det[:, 3])/2-img_center[0]])  # noqa: E501
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)  # noqa: E501
                                index = np.argmax(bounding_box_size - offset_dist_squared*2.0)  # noqa: E501  # some extra weight on the centering
                                det = det[index, :]
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = resize(cropped, (image_size, image_size),
                                            mode='constant')
                            nrof_successfully_aligned += 1
                            # convert image to uint8 before saving
                            imageio.imwrite(output_filename,
                                            img_as_ubyte(scaled))
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))  # noqa: E501
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)  # noqa: E501




try:
    xrange = range
except:
    range = xrange


def embeding(model_dir='20170512-110547', data_dir='database_aligned',
                 is_aligned=True, image_size=160, margin=44, gpu_memory_fraction=1.0,
                 image_batch=1000, embeddings_name='embeddings.npy',
                 labels_name='labels.npy', labels_strings_name='label_strings.npy',
                 return_image_list=False):
        train_set = facenet.get_dataset(data_dir)
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        # fetch the classes (labels as strings) exactly as it's done in get_dataset
        path_exp = os.path.expanduser(data_dir)
        classes = [path for path in os.listdir(path_exp)
                   if os.path.isdir(os.path.join(path_exp, path))]
        # get the label strings
        label_strings = [name for name in classes if
                         os.path.isdir(os.path.join(path_exp, name))]

        with tf.Graph().as_default():

            with tf.Session() as sess:

                # Load model
                facenet.load_model(model_dir)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")  # noqa: E501
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # noqa: E501
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Run forward pass to calculate embeddings
                nrof_images = len(image_list)
                print('Number of images: ', nrof_images)
                batch_size = image_batch
                if nrof_images % batch_size == 0:
                    nrof_batches = nrof_images // batch_size
                else:
                    nrof_batches = (nrof_images // batch_size) + 1
                print('Number of batches: ', nrof_batches)
                embedding_size = embeddings.get_shape()[1]
                emb_array = np.zeros((nrof_images, embedding_size))
                start_time = time.time()

                for i in range(nrof_batches):
                    if i == nrof_batches - 1:
                        n = nrof_images
                    else:
                        n = i * batch_size + batch_size
                    # Get images for the batch
                    if is_aligned is True:
                        images = facenet.load_data(image_list[i * batch_size:n],
                                                   False, False, image_size)
                    else:
                        images = load_and_align_data(image_list[i * batch_size:n],
                                                     image_size, margin,
                                                     gpu_memory_fraction)
                    feed_dict = {images_placeholder: images,
                                 phase_train_placeholder: False}
                    # Use the facenet model to calculate embeddings
                    embed = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array[i * batch_size:n, :] = embed
                    print('Completed batch', i + 1, 'of', nrof_batches)

                run_time = time.time() - start_time
                print('Run time: ', run_time)

                #   export embeddings and labels
                label_list = np.array(label_list)

                np.save(embeddings_name, emb_array)
                if emb_array.size > 0:
                    labels_final = (label_list) - np.min(label_list)
                    np.save(labels_name, labels_final)
                    label_strings = np.array(label_strings)
                    np.save(labels_strings_name, label_strings[labels_final])
                    np.save('image_list.npy', image_list)
                if return_image_list:
                    np.save('validation_image_list.npy', image_list)
                    return image_list, emb_array


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  # noqa: E501
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        print(image_paths[i])
        img = imageio.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet,
                                                    onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = resize(cropped, (image_size, image_size))
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images





def saveDatabse(userList):

    if not os.path.exists('temp_images'):
        os.makedirs('temp_images')
    if not os.path.exists('database/like'):
        os.makedirs('database/like')
    if not os.path.exists('database/dislike'):
        os.makedirs('database/dislike')
    if not os.path.exists('al_database'):
        os.makedirs('al_database')

    # attempt to load database
    try:
        database = list(np.load('database.npy'))
        print('You have browsed', len(database), 'Tinder profiles.')
    except:
        database = []

    # attempt to load an auto liked or disliked database
    try:
        al_database = list(np.load('al_database.npy'))
        print('You have automatically liked or disliked ',
              len(al_database), 'Tinder profiles.')
    except:
        al_database = []

    #didILike =  'Like' #Dislike #like_or_dislike()


    #userList = ["test3", "Leeta", "20", "booooo is meeeeeeee", "50km", "employed", "apj23", didILike]


    database.append(userList)
    np.save('database.npy', database)

    return "true"


def trainModel():

    align()#tindetheus align main
    # export the embeddings from the aligned database
    embeding("20170512-110547")#exportEmbedding
    # calculate the n average embedding per profiles
    X, y = calc_avg_emb()
    # fit and save a logistic regression model to the database
    fit_log_reg(X, y)












        # print('... Loading the facenet model ...')
        # print('... be patient this may take some time ...')
        # with tf.Graph().as_default():
        #     with tf.Session() as sess:
        #         # pass the tf session into client object
        #         my_sess = client(facebook_token, args.distance, args.model_dir,
        #                          likes_left=args.likes, tfsess=sess)
        #         # Load the facenet model
        #         facenet.load_model(my_sess.model_dir)
        #         print('Facenet model loaded successfully!!!')
        #         # automatically like users
        #         my_sess.like()
        #


def predictModel():

    print('\n\nAttempting to validate the dataset...\n\n')
    valdir = 'validation'
    # align the validation dataset

    align(input_dir=valdir, output_dir=valdir + '_aligned')
        # export embeddings
        # y is the image list, X is the embedding_array
    image_list, emb_array = embeding(model_dir="20170512-110547",  # noqa: E501
                                        data_dir=valdir+'_aligned',
                                        image_batch=1000,
                                        embeddings_name='val_embeddings.npy',
                                        labels_name='val_labels.npy',
                                        labels_strings_name='val_label_strings.npy',  # noqa: E501
                                        return_image_list=True)




        # print(image_list)
        # convert the image list to a numpy array to take advantage of
        # numpy array slicing
    image_list = np.array(image_list)
    boolMe = 1
    print('\n\nEvaluating trained model\n \n')
    model = joblib.load('log_reg_model.pkl')
    yhat = model.predict(emb_array)
        # print(yhat)
        # 0 should be dislike, and 1 should be like
        # if this is backwards, there is probablly a bug...
    # dislikes = yhat == 0
    # likes = yhat == 1
    return yhat
        #  print(yhat)
        # show_images(image_list[dislikes], holdon=True, title='Dislike')
        # print('\n\nGenerating plots...\n\n')
        # plt.title('Dislike')
        #
        # show_images(image_list[likes], holdon=True, title='Like')
        # plt.title('Like')
        #
        # cols = ['Image name', 'Model prediction (0=Dislike, 1=Like)']
        # results = np.array((image_list, yhat)).T
        # print('\n\nSaving results to validation.csv\n\n')
        # my_results_DF = pd.DataFrame(results, columns=cols)
        # my_results_DF.to_csv('validation.csv')
        #
        # plt.show()




####################################################### F L A S K #######################################


from flask import Flask, request,jsonify, make_response
import json


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response



app = Flask(__name__)



@app.route('/')
def index():
    return "Started python server"



@app.route('/browseData', methods=['POST', "OPTIONS"])
def postdata():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        data = request.get_json()
        print(data)

        folderName = ''
        if data['User_List'][len(data['User_List'])-1] == "Like":
            folderName = 'like/'
        else:
            folderName = 'dislike/'
        urlretrieve(data['Image_Url'], 'database/'+folderName+data['User_List'][0]+'.jpg')
        if saveDatabse(data['User_List']) == "true":
            return _corsify_actual_response(jsonify(data))



@app.route('/trainData', methods=['POST', "OPTIONS"])
def postdatatrain():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        data = request.get_json()
        print(data)
        align()  # tindetheus align main
        # export the embeddings from the aligned database
        embeding("20170512-110547")  # exportEmbedding
        # calculate the n average embedding per profiles
        X, y = calc_avg_emb()
        # fit and save a logistic regression model to the database
        fit_log_reg(X, y)

        return _corsify_actual_response(jsonify(data))



@app.route('/swipeData', methods=['POST', "OPTIONS"])
def postdataswipe():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        data = request.get_json()
        print(data)
        os.remove('validation_aligned/like/test.png')
        urlretrieve(data['Image_Url'], 'validation/like/test.jpg')
        likeDislike = predictModel()
        # os.remove('validation/like/test.jpg')

        print(likeDislike)
        data['predict'] = likeDislike[0]
        print(data)
        return _corsify_actual_response(jsonify(data))




if __name__ == "__main__":
    app.run(port=5000)



